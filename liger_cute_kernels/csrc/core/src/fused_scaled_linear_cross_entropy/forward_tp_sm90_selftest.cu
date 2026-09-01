// ═══════════════════════════════════════════════════════════════════════════
// Internal multi-PE self test for the tensor-parallel forward path.
//
// Compiled out unless LIGER_FSLCE_FORWARD_TP_SM90_SELFTEST is defined, so the
// core library's recursive .cu glob picks this file up as an empty translation
// unit and no shared CMake/test file has to change. Build and run with:
//
//   nvcc -std=c++17 -arch=sm_90a -O3 --expt-relaxed-constexpr \
//     -c fused_linear_scaled_cross_entropy_forward.cu -o forward.o <includes>
//   nvcc -std=c++17 -arch=sm_90a -O3 --expt-relaxed-constexpr -rdc=true \
//     -c forward_reduce.cu -o forward_reduce.o <includes>
//   nvcc -std=c++17 -arch=sm_90a -O3 --expt-relaxed-constexpr -rdc=true \
//     -DLIGER_FSLCE_FORWARD_TP_SM90_SELFTEST \
//     forward_tp_sm90_selftest.cu forward_reduce.o forward.o \
//     -I<this dir> -I<core include> -I<core src/moe> \
//     -I$CUTLASS_HOME/include -I$NVSHMEM_HOME/include \
//     -L$NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device -lcuda -o tp_selftest
//   nvshmrun -n 4 ./tp_selftest
//
// Every PE owns a contiguous shard of one global vocabulary. Each PE rebuilds
// the *whole* weight matrix from the shared seed and checks the reduced
// NLL / LSE / entropy against a full-vocabulary CPU reference, so a wrong
// global max, a missed rescale or a dropped target contribution all show up.
// ═══════════════════════════════════════════════════════════════════════════

#if defined(LIGER_FSLCE_FORWARD_TP_SM90_SELFTEST)

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "buffer_pool.cuh"
#include "forward_gemm_sm90.cuh"
#include "forward_reduce.cuh"
#include "workspace.cuh"

namespace {

namespace fslce = liger::fused_scaled_linear_cross_entropy;

using Element = __nv_bfloat16;

int g_pe = 0;
int g_n_pes = 1;
int g_failures = 0;

#define CUDA_OK(expr)                                                        \
	do {                                                                     \
		cudaError_t status = (expr);                                         \
		if (status != cudaSuccess) {                                         \
			std::printf("[pe %d] CUDA error %s at %s:%d: %s\n",              \
				g_pe, #expr, __FILE__, __LINE__,                             \
				cudaGetErrorString(status));                                 \
			std::exit(1);                                                    \
		}                                                                    \
	} while (0)

float next_random(std::uint64_t& state) {
	state = state * 6364136223846793005ULL + 1442695040888963407ULL;
	std::uint32_t bits = static_cast<std::uint32_t>(state >> 33);
	return static_cast<float>(bits % 2048) / 1024.0f - 1.0f;
}

// Deterministic from (seed, index) so every PE reconstructs the same global
// tensors without any communication.
float element_at(std::uint64_t seed, std::size_t index, float scale) {
	std::uint64_t state = seed ^ (index * 0x9E3779B97F4A7C15ULL);
	state = state * 6364136223846793005ULL + 1442695040888963407ULL;
	state ^= state >> 29;
	state = state * 0xBF58476D1CE4E5B9ULL;
	state ^= state >> 32;
	std::uint32_t bits = static_cast<std::uint32_t>(state >> 33);
	return (static_cast<float>(bits % 2048) / 1024.0f - 1.0f) * scale;
}

struct ReferenceRow {
	float lse;
	float nll;
	float entropy;
	float row_max;
	float sum;
	float expectation;
};

// Full-vocabulary CPU reference, reproducing the kernel's FP16 staging round.
std::vector<ReferenceRow> reference(
		const std::vector<float>& x,
		const std::vector<float>& weight_global,
		const std::vector<std::int64_t>& target,
		int tokens,
		int hidden,
		int global_vocab,
		std::int64_t ignore_index,
		float inverse_temperature) {
	std::vector<ReferenceRow> rows(tokens);
	std::vector<float> scaled(global_vocab);
	for (int m = 0; m < tokens; ++m) {
		float row_max = -1.0e38f;
		for (int v = 0; v < global_vocab; ++v) {
			float dot = 0.0f;
			for (int k = 0; k < hidden; ++k) {
				dot += x[static_cast<std::size_t>(m) * hidden + k] *
					weight_global[static_cast<std::size_t>(v) * hidden + k];
			}
			scaled[v] = __half2float(__float2half(dot)) * inverse_temperature;
			row_max = fmaxf(row_max, scaled[v]);
		}

		double sum = 0.0;
		double weighted = 0.0;
		for (int v = 0; v < global_vocab; ++v) {
			double e = std::exp(
				static_cast<double>(scaled[v]) - static_cast<double>(row_max));
			sum += e;
			weighted += e * static_cast<double>(scaled[v]);
		}
		float lse = row_max + static_cast<float>(std::log(sum));

		bool ignored = target[m] == ignore_index;
		rows[m].expectation = static_cast<float>(weighted / sum);
		rows[m].row_max = row_max;
		rows[m].sum = static_cast<float>(sum);
		rows[m].lse = lse;
		rows[m].nll = ignored ? 0.0f : lse - scaled[target[m]];
		rows[m].entropy = ignored
			? 0.0f
			: lse - static_cast<float>(weighted / sum);
	}
	return rows;
}

void expect_near(
		const char* label, int row, float actual, float expected,
		float absolute, float relative) {
	float tolerance = absolute + relative * fabsf(expected);
	if (!(fabsf(actual - expected) <= tolerance)) {
		if (g_failures < 20) {
			std::printf(
				"[pe %d]   FAIL %-8s row %4d: got %14.6f want %14.6f "
				"(tol %.6f)\n",
				g_pe, label, row, actual, expected, tolerance);
		}
		++g_failures;
	}
}

template <bool ReturnEntropy>
void run_case(
		const std::string& name,
		int tokens,
		int hidden,
		int local_vocab,
		float temperature,
		std::uint64_t seed) {
	const int global_vocab = local_vocab * g_n_pes;
	const std::int64_t vocab_start =
		static_cast<std::int64_t>(g_pe) * local_vocab;
	const float scale = 4.0f / std::sqrt(static_cast<float>(hidden));

	// Rebuild the global tensors identically on every PE.
	std::vector<float> host_x(static_cast<std::size_t>(tokens) * hidden);
	for (std::size_t i = 0; i < host_x.size(); ++i) {
		host_x[i] = element_at(seed, i, scale);
	}
	std::vector<float> host_w(
		static_cast<std::size_t>(global_vocab) * hidden);
	for (std::size_t i = 0; i < host_w.size(); ++i) {
		host_w[i] = element_at(seed ^ 0xABCDEF01ULL, i, scale);
	}
	// Round through BF16 so host and device see identical operands.
	for (auto& v : host_x) v = __bfloat162float(__float2bfloat16(v));
	for (auto& v : host_w) v = __bfloat162float(__float2bfloat16(v));

	std::vector<std::int64_t> host_target(tokens);
	std::uint64_t target_state = seed ^ 0x5DEECE66DULL;
	for (int m = 0; m < tokens; ++m) {
		if (m % 13 == 5) {
			host_target[m] = -100;  // ignore_index
		} else {
			next_random(target_state);
			host_target[m] = static_cast<std::int64_t>(
				(target_state >> 17) % static_cast<std::uint64_t>(global_vocab));
		}
	}

	// Device operands: X is replicated, W is this PE's contiguous shard.
	std::vector<Element> device_x_host(host_x.size());
	for (std::size_t i = 0; i < host_x.size(); ++i) {
		device_x_host[i] = __float2bfloat16(host_x[i]);
	}
	std::vector<Element> device_w_host(
		static_cast<std::size_t>(local_vocab) * hidden);
	for (std::size_t i = 0; i < device_w_host.size(); ++i) {
		device_w_host[i] = __float2bfloat16(
			host_w[static_cast<std::size_t>(vocab_start) * hidden + i]);
	}

	Element* device_x = nullptr;
	Element* device_w = nullptr;
	std::int64_t* device_target = nullptr;
	CUDA_OK(cudaMalloc(&device_x, device_x_host.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_w, device_w_host.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_target, tokens * sizeof(std::int64_t)));
	CUDA_OK(cudaMemcpy(device_x, device_x_host.data(),
		device_x_host.size() * sizeof(Element), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_w, device_w_host.data(),
		device_w_host.size() * sizeof(Element), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_target, host_target.data(),
		tokens * sizeof(std::int64_t), cudaMemcpyHostToDevice));

	float* device_out = nullptr;
	CUDA_OK(cudaMalloc(&device_out, 3 * tokens * sizeof(float)));
	CUDA_OK(cudaMemset(device_out, 0xff, 3 * tokens * sizeof(float)));

	fslce::ForwardTpParamsSm90<90> params;
	params.gemm.x = device_x;
	params.gemm.weight = device_w;
	params.gemm.target = device_target;
	params.gemm.tokens = tokens;
	params.gemm.hidden = hidden;
	params.gemm.local_vocab = local_vocab;
	params.gemm.vocab_start = vocab_start;
	params.gemm.ignore_index = -100;
	params.gemm.inverse_temperature = 1.0f / temperature;
	params.nll = device_out + 0 * tokens;
	params.lse = device_out + 1 * tokens;
	params.entropy = device_out + 2 * tokens;
	params.team_handle =
		static_cast<std::int64_t>(NVSHMEM_TEAM_WORLD);

	fslce::fused_linear_scaled_cross_entropy_forward<
		ReturnEntropy, 90>(params, nullptr);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	std::vector<float> host_out(3 * tokens);
	CUDA_OK(cudaMemcpy(host_out.data(), device_out,
		host_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

	fslce::ForwardTpWorkspace probe =
		fslce::reserve_forward_tp_workspace<ReturnEntropy>(tokens);
	std::vector<float> host_global_max(tokens);
	std::vector<float> host_reduced(
		static_cast<std::size_t>(probe.fields) * tokens);
	CUDA_OK(cudaMemcpy(host_global_max.data(), probe.global_max,
		tokens * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_OK(cudaMemcpy(host_reduced.data(), probe.reduced,
		host_reduced.size() * sizeof(float), cudaMemcpyDeviceToHost));

	std::vector<ReferenceRow> expected = reference(
		host_x, host_w, host_target, tokens, hidden, global_vocab,
		params.gemm.ignore_index, params.gemm.inverse_temperature);

	int before = g_failures;
	int stage_max_bad = 0;
	int stage_sum_bad = 0;
	for (int m = 0; m < tokens; ++m) {
		if (fabsf(host_global_max[m] - expected[m].row_max) >
			2.0e-3f + 1.0e-3f * fabsf(expected[m].row_max)) {
			++stage_max_bad;
		}
		float reduced_sum = host_reduced[0 * tokens + m];
		if (fabsf(reduced_sum - expected[m].sum) >
			1.0e-2f + 5.0e-3f * fabsf(expected[m].sum)) {
			++stage_sum_bad;
		}
	}
	if (stage_max_bad || stage_sum_bad) {
		std::printf("[pe %d] STAGE %-28s max_bad=%d sum_bad=%d of %d rows\n",
			g_pe, name.c_str(), stage_max_bad, stage_sum_bad, tokens);
	}
	for (int m = 0; m < tokens; ++m) {
		expect_near("lse", m, host_out[1 * tokens + m],
			expected[m].lse, 3.0e-3f, 2.0e-3f);
		expect_near("nll", m, host_out[0 * tokens + m],
			expected[m].nll, 3.0e-3f, 2.0e-3f);
		if constexpr (ReturnEntropy) {
			expect_near("entropy", m, host_out[2 * tokens + m],
				expected[m].entropy, 5.0e-3f, 5.0e-3f);
			// Backward recovers the softmax-weighted mean logit from the two
			// saved output tensors as (lse - entropy); check that identity
			// holds wherever the gradient is non-zero.
			if (host_target[m] != params.gemm.ignore_index) {
				expect_near("lse-entropy", m,
					host_out[1 * tokens + m] - host_out[2 * tokens + m],
					expected[m].expectation, 5.0e-3f, 5.0e-3f);
			}
		}
		if (g_failures > before + 20) break;
	}

	if (g_pe == 0) {
		std::printf("%-40s %s  (PEs=%d M=%d H=%d Vloc=%d Vglobal=%d T=%.2f "
			"entropy=%d)\n",
			name.c_str(),
			g_failures == before ? "PASS" : "FAIL",
			g_n_pes, tokens, hidden, local_vocab, global_vocab, temperature,
			static_cast<int>(ReturnEntropy));
	}

	// Keep every PE on the same case boundary before pooled symmetric
	// reduction buffers are reused by the next launch.
	nvshmem_barrier_all();

	CUDA_OK(cudaFree(device_x));
	CUDA_OK(cudaFree(device_w));
	CUDA_OK(cudaFree(device_target));
	CUDA_OK(cudaFree(device_out));
}

}  // namespace

int main() {
	nvshmem_init();
	g_pe = nvshmem_my_pe();
	g_n_pes = nvshmem_n_pes();
	CUDA_OK(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

	cudaDeviceProp properties = {};
	CUDA_OK(cudaGetDeviceProperties(&properties, nvshmem_team_my_pe(
		NVSHMEMX_TEAM_NODE)));
	if (properties.major != 9) {
		if (g_pe == 0) {
			std::printf("skipping: requires Hopper (sm90), found sm_%d%d\n",
				properties.major, properties.minor);
		}
		nvshmem_finalize();
		return 0;
	}

	if (g_pe == 0) {
		std::printf("tensor-parallel forward over %d PEs\n", g_n_pes);
	}
	// Capacities are immutable once allocated, so configure for the largest
	// shape used below before any launch.
	constexpr int kMaxTokens = 1024;
	constexpr int kMaxHidden = 192;
	constexpr int kMaxLocalVocab = 2560;
	fslce::configure_backward_tp_symmetric(
		kMaxTokens,
		kMaxHidden,
		kMaxLocalVocab,
		1,
		1,
		static_cast<std::int64_t>(NVSHMEM_TEAM_WORLD));
	fslce::configure_forward_tp_workspace(kMaxTokens, kMaxLocalVocab);

	try {
		run_case<false>("tp aligned / no entropy", 256, 128, 640, 1.0f, 101);
		run_case<true>("tp aligned / entropy", 256, 128, 640, 1.0f, 102);
		run_case<true>("tp ragged M/H/V", 200, 104, 517, 0.8f, 103);
		run_case<false>("tp temperature / no entropy", 384, 192, 901, 1.4f, 104);
		run_case<true>("tp large split", 1024, 128, 2560, 1.0f, 105);
	} catch (const std::exception& error) {
		std::printf("[pe %d] exception: %s\n", g_pe, error.what());
		nvshmem_finalize();
		return 1;
	}

	// Make every rank's verdict visible before PE 0 prints the summary.
	// Pooled, not raw nvshmem_malloc: this module never runs its own symmetric
	// allocation lifecycle.
	auto& pool = liger::global_buffer_pool();
	int* local_slot = static_cast<int*>(
		pool.get_symmetric("fslce_tp_selftest_verdict_local", sizeof(int)));
	int* total_slot = static_cast<int*>(
		pool.get_symmetric("fslce_tp_selftest_verdict_total", sizeof(int)));
	CUDA_OK(cudaMemcpy(local_slot, &g_failures, sizeof(int),
		cudaMemcpyHostToDevice));
	nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, total_slot, local_slot, 1);
	int host_total = 0;
	CUDA_OK(cudaMemcpy(&host_total, total_slot, sizeof(int),
		cudaMemcpyDeviceToHost));

	if (g_pe == 0) {
		std::printf("%s: %d comparison failures across %d PEs\n",
			host_total == 0 ? "OK" : "FAILED", host_total, g_n_pes);
	}
	fslce::reset_fslce_tp_configuration();
	liger::global_buffer_pool().clear();
	nvshmem_finalize();
	return host_total == 0 ? 0 : 1;
}

#endif  // LIGER_FSLCE_FORWARD_TP_SM90_SELFTEST
