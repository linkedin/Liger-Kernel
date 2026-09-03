// ═══════════════════════════════════════════════════════════════════════════
// Internal SM90 forward GEMM self test.
//
// Compiled out unless LIGER_FSLCE_FORWARD_SM90_SELFTEST is defined, so the
// core library's recursive .cu glob picks this file up as an empty translation
// unit and no shared CMake/test file has to change. Build it standalone with:
//
//   nvcc -std=c++17 -arch=sm_90a -O3 --expt-relaxed-constexpr \
//     -DLIGER_FSLCE_FORWARD_SM90_SELFTEST \
//     forward_gemm_sm90_selftest.cu \
//     -I<this dir> -I<core include> -I$CUTLASS_HOME/include -o selftest
//
// Checks the fused kernel's local (max, sum, lse, target, weighted) statistics
// against a CPU reference that reproduces the same staging precision: the
// reference rounds every logit through FP16 exactly like the N80 SMEM slices,
// then applies the temperature scaling.
// ═══════════════════════════════════════════════════════════════════════════

#if defined(LIGER_FSLCE_FORWARD_SM90_SELFTEST)

#include "forward_gemm_mainloop_sm90.cuh"

#include "forward_gemm_sm90.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace fslce = liger::fused_scaled_linear_cross_entropy;

using Element = cutlass::bfloat16_t;
using ElementLogit = cutlass::half_t;

int g_failures = 0;

#define CUDA_OK(expr)                                                        \
	do {                                                                     \
		cudaError_t status = (expr);                                         \
		if (status != cudaSuccess) {                                         \
			std::printf("CUDA error %s at %s:%d: %s\n",                      \
				#expr, __FILE__, __LINE__, cudaGetErrorString(status));      \
			std::exit(1);                                                    \
		}                                                                    \
	} while (0)

float next_random(std::uint64_t& state) {
	state = state * 6364136223846793005ULL + 1442695040888963407ULL;
	std::uint32_t bits = static_cast<std::uint32_t>(state >> 33);
	return static_cast<float>(bits % 2048) / 1024.0f - 1.0f;
}

struct ReferenceRow {
	float max_value;
	float exp_sum;
	float lse;
	float target_logit;
	float weighted_sum;
};

// CPU mirror of the device epilogue: FP32 dot product, FP16 staging round,
// temperature scaling, then a plain (non-online) softmax reduction.
std::vector<ReferenceRow> reference(
		const std::vector<Element>& x,
		const std::vector<Element>& weight,
		const std::vector<std::int64_t>& target,
		int tokens,
		int hidden,
		int local_vocab,
		std::int64_t vocab_start,
		std::int64_t ignore_index,
		float inverse_temperature) {
	std::vector<ReferenceRow> rows(tokens);
	std::vector<float> scaled(local_vocab);
	for (int m = 0; m < tokens; ++m) {
		float row_max = -1.0e38f;
		for (int v = 0; v < local_vocab; ++v) {
			float dot = 0.0f;
			for (int k = 0; k < hidden; ++k) {
				dot += static_cast<float>(x[m * hidden + k]) *
					static_cast<float>(weight[v * hidden + k]);
			}
			float staged = static_cast<float>(static_cast<ElementLogit>(dot));
			scaled[v] = staged * inverse_temperature;
			row_max = fmaxf(row_max, scaled[v]);
		}

		double sum = 0.0;
		double weighted = 0.0;
		for (int v = 0; v < local_vocab; ++v) {
			double e = std::exp(
				static_cast<double>(scaled[v]) - static_cast<double>(row_max));
			sum += e;
			weighted += e * static_cast<double>(scaled[v]);
		}

		float target_logit = 0.0f;
		std::int64_t target_id = target[m];
		if (target_id != ignore_index) {
			std::int64_t local = target_id - vocab_start;
			if (local >= 0 && local < local_vocab) {
				target_logit = scaled[static_cast<int>(local)];
			}
		}

		rows[m].max_value = row_max;
		rows[m].exp_sum = static_cast<float>(sum);
		rows[m].lse = row_max + static_cast<float>(std::log(sum));
		rows[m].target_logit = target_logit;
		rows[m].weighted_sum = static_cast<float>(weighted);
	}
	return rows;
}

void expect_near(
		const char* label,
		int row,
		float actual,
		float expected,
		float absolute,
		float relative) {
	float tolerance = absolute + relative * fabsf(expected);
	if (!(fabsf(actual - expected) <= tolerance)) {
		std::printf(
			"  FAIL %-14s row %4d: got %14.6f want %14.6f (tol %.6f)\n",
			label, row, actual, expected, tolerance);
		++g_failures;
	}
}

// Test-only direct kernel launch. Production has no GEMM-only host wrapper:
// fused_linear_scaled_cross_entropy_forward owns this setup.
template <bool ReturnEntropy>
void launch_forward_gemm_selftest(
		const fslce::ForwardGemmParamsSm90<90>& params,
		cudaStream_t stream) {
	using Traits = fslce::ForwardGemmTraitsSm90<90>;
	using Config = typename Traits::Config;
	using Launch = fslce::ForwardGemmLaunchSm90<90>;
	using Smem = fslce::ForwardGemmSmemSm90<90, ReturnEntropy>;
	using Cluster = liger::sm90::ClusterLaunchSm90<90>;

	auto tma_load_x = liger::sm90::TmaSm90<90>::template make_load<
		typename Traits::SmemLayoutX1,
		Traits::kTileM,
		Traits::kTileK>(
			static_cast<const Element*>(params.x),
			params.tokens,
			params.hidden);
	auto tma_load_w =
		liger::sm90::TmaSm90<90>::template make_load_multicast<
			typename Traits::SmemLayoutW1,
			Traits::kAccumulatorTileN,
			Traits::kTileK,
			Traits::kClusterM>(
				static_cast<const Element*>(params.weight),
				params.local_vocab,
				params.hidden);
	auto* kernel_fn = &fslce::forward_gemm_kernel_sm90<
		ReturnEntropy, 90, decltype(tma_load_x), decltype(tma_load_w)>;

	constexpr int kSmemBytes = static_cast<int>(sizeof(Smem));
	CUDA_OK(Cluster::prepare(kernel_fn, kSmemBytes));
	int max_active_cluster_pairs = Cluster::max_active_clusters(
		kernel_fn, Config::kNumThreads, kSmemBytes, Config::kClusterM);
	fslce::ForwardGemmSplitSm90<90> split = Launch::resolve_split(
		params.tuning,
		params.tokens,
		params.local_vocab,
		max_active_cluster_pairs);

	std::size_t partial_rows =
		static_cast<std::size_t>(split.num_m_tiles) *
		static_cast<std::size_t>(split.split_n) *
		static_cast<std::size_t>(Config::kTileM);
	float* scratch = static_cast<float*>(params.workspace);
	fslce::ForwardGemmPartialsSm90<90> partials;
	partials.partial_max = scratch + 0 * partial_rows;
	partials.partial_sum = scratch + 1 * partial_rows;
	partials.partial_target = scratch + 2 * partial_rows;
	partials.partial_weighted =
		ReturnEntropy ? scratch + 3 * partial_rows : nullptr;

	CUDA_OK(Cluster::launch(
		kernel_fn,
		dim3(
			static_cast<unsigned>(Config::kClusterM),
			1u,
			static_cast<unsigned>(split.num_cluster_pairs)),
		Config::kNumThreads,
		kSmemBytes,
		Config::kClusterM,
		stream,
		tma_load_x,
		tma_load_w,
		params,
		partials,
		split));
	fslce::forward_split_reduce_kernel_sm90<ReturnEntropy, 90>
		<<<split.num_m_tiles, Config::kTileM, 0, stream>>>(
			params, partials, split);
}

template <bool ReturnEntropy>
void run_case(
		const std::string& name,
		int tokens,
		int hidden,
		int local_vocab,
		std::int64_t vocab_start,
		float temperature,
		std::uint64_t seed,
		fslce::ForwardGemmTuningSm90<90> tuning = {}) {
	std::uint64_t state = seed;
	std::vector<Element> host_x(static_cast<std::size_t>(tokens) * hidden);
	std::vector<Element> host_w(static_cast<std::size_t>(local_vocab) * hidden);
	std::vector<std::int64_t> host_target(tokens);

	const float scale = 1.0f / std::sqrt(static_cast<float>(hidden));
	for (auto& value : host_x) {
		value = static_cast<Element>(next_random(state) * scale * 4.0f);
	}
	for (auto& value : host_w) {
		value = static_cast<Element>(next_random(state) * scale * 4.0f);
	}
	const std::int64_t global_vocab = vocab_start + local_vocab + 37;
	for (int m = 0; m < tokens; ++m) {
		if (m % 11 == 3) {
			host_target[m] = -100;  // ignore_index
		} else if (m % 5 == 0) {
			// Force a target that lands outside this vocabulary shard.
			host_target[m] = (vocab_start + local_vocab) % global_vocab;
		} else {
			std::uint64_t bits = state = state * 2862933555777941757ULL + 3037000493ULL;
			host_target[m] = vocab_start +
				static_cast<std::int64_t>(bits % static_cast<std::uint64_t>(local_vocab));
		}
	}

	Element* device_x = nullptr;
	Element* device_w = nullptr;
	std::int64_t* device_target = nullptr;
	CUDA_OK(cudaMalloc(&device_x, host_x.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_w, host_w.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_target, host_target.size() * sizeof(std::int64_t)));
	CUDA_OK(cudaMemcpy(device_x, host_x.data(),
		host_x.size() * sizeof(Element), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_w, host_w.data(),
		host_w.size() * sizeof(Element), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_target, host_target.data(),
		host_target.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice));

	float* device_stats = nullptr;
	CUDA_OK(cudaMalloc(&device_stats, 4 * tokens * sizeof(float)));
	CUDA_OK(cudaMemset(device_stats, 0xff, 4 * tokens * sizeof(float)));

	using Launch = fslce::ForwardGemmLaunchSm90<90>;
	std::size_t workspace_bytes = Launch::workspace_bytes(
		tokens, local_vocab, ReturnEntropy, tuning.max_split_n);
	void* device_workspace = nullptr;
	CUDA_OK(cudaMalloc(&device_workspace, workspace_bytes));

	fslce::ForwardGemmParamsSm90<90> params;
	params.x = device_x;
	params.weight = device_w;
	params.target = device_target;
	params.output.local_max = device_stats + 0 * tokens;
	params.output.local_sum = device_stats + 1 * tokens;
	params.output.local_target = device_stats + 2 * tokens;
	params.output.local_weighted_sum = device_stats + 3 * tokens;
	params.workspace = device_workspace;
	params.workspace_bytes = workspace_bytes;
	params.tokens = tokens;
	params.hidden = hidden;
	params.local_vocab = local_vocab;
	params.vocab_start = vocab_start;
	params.ignore_index = -100;
	params.inverse_temperature = 1.0f / temperature;
	params.tuning = tuning;

	launch_forward_gemm_selftest<ReturnEntropy>(params, nullptr);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	std::vector<float> host_stats(4 * tokens);
	CUDA_OK(cudaMemcpy(host_stats.data(), device_stats,
		host_stats.size() * sizeof(float), cudaMemcpyDeviceToHost));

	std::vector<ReferenceRow> expected = reference(
		host_x, host_w, host_target, tokens, hidden, local_vocab,
		vocab_start, params.ignore_index, params.inverse_temperature);

	int before = g_failures;
	for (int m = 0; m < tokens; ++m) {
		expect_near("local_max", m, host_stats[0 * tokens + m],
			expected[m].max_value, 2.0e-3f, 1.0e-3f);
		expect_near("local_sum", m, host_stats[1 * tokens + m],
			expected[m].exp_sum, 1.0e-3f, 5.0e-3f);
		expect_near("local_target", m, host_stats[2 * tokens + m],
			expected[m].target_logit, 2.0e-3f, 1.0e-3f);
		if constexpr (ReturnEntropy) {
			expect_near("local_weighted", m, host_stats[3 * tokens + m],
				expected[m].weighted_sum, 5.0e-3f, 5.0e-3f);
		}
		if (g_failures > before + 20) break;
	}

	std::printf("%-46s %s  (M=%d H=%d Vloc=%d start=%lld T=%.2f entropy=%d "
		"tuning split=%d base=%d extra=%d)\n",
		name.c_str(),
		g_failures == before ? "PASS" : "FAIL",
		tokens, hidden, local_vocab,
		static_cast<long long>(vocab_start),
		temperature,
		static_cast<int>(ReturnEntropy),
		tuning.split_n, tuning.base_split_n, tuning.extra_m_pairs);

	CUDA_OK(cudaFree(device_x));
	CUDA_OK(cudaFree(device_w));
	CUDA_OK(cudaFree(device_target));
	CUDA_OK(cudaFree(device_stats));
	CUDA_OK(cudaFree(device_workspace));
}

void report_occupancy() {
	using Traits = fslce::ForwardGemmTraitsSm90<90>;
	std::printf(
		"geometry: threads=%d cluster_m=%d stages=%d tileM=%d tileK=%d "
		"logicalN=%d accumN=%d sliceN=%d chunkN=%d tma_bytes=%d\n",
		Traits::kNumThreads, Traits::kClusterM, Traits::kStages,
		Traits::kTileM, Traits::kTileK, Traits::kLogicalTileN,
		Traits::kAccumulatorTileN, Traits::kEpilogueSliceN,
		Traits::kSoftmaxChunkN, Traits::kTmaTransBytes);
	std::printf("smem: no-entropy=%zu B  entropy=%zu B  (Hopper limit %d B)\n",
		sizeof(fslce::ForwardGemmSmemSm90<90, false>),
		sizeof(fslce::ForwardGemmSmemSm90<90, true>),
		fslce::kHopperMaxSmemBytes);
}

}  // namespace

int main() {
	int device = 0;
	cudaDeviceProp properties = {};
	CUDA_OK(cudaGetDeviceProperties(&properties, device));
	if (properties.major != 9) {
		std::printf("skipping: requires a Hopper (sm90) GPU, found sm_%d%d\n",
			properties.major, properties.minor);
		return 0;
	}
	report_occupancy();

	try {
		// Aligned: M and V are exact multiples of the tile geometry.
		run_case<false>("aligned / no entropy", 256, 128, 640, 0, 1.0f, 1);
		run_case<true>("aligned / entropy", 256, 128, 640, 0, 1.0f, 2);
		// Ragged M (odd M-tile count exercises the redundant cluster peer),
		// ragged H (TMA zero fill in K) and ragged V (epilogue masking).
		// H stays a multiple of 8 so the BF16 row stride keeps TMA's 16 B
		// global-alignment requirement.
		run_case<false>("ragged M/H/V / no entropy", 200, 104, 517, 0, 1.0f, 3);
		run_case<true>("ragged M/H/V / entropy", 200, 104, 517, 0, 1.0f, 4);
		// K tail: 72 = one full K64 tile plus 8 valid columns.
		run_case<true>("ragged K tail", 128, 72, 333, 0, 1.0f, 8);
		// Sharded vocabulary with a non-zero start plus temperature scaling.
		run_case<true>("shard offset + temperature", 384, 192, 901, 4096, 0.7f, 5);
		run_case<false>("single M tile", 96, 64, 321, 12288, 1.3f, 6);
		// Local vocabulary smaller than one logical N320 tile.
		run_case<true>("tiny local vocabulary", 128, 64, 40, 100000, 1.0f, 7);

		// Vocabulary-split scheduler. M=1024 -> 8 M tiles -> 4 M pairs;
		// V=2560 -> 8 logical N320 tiles.
		fslce::ForwardGemmTuningSm90<90> uniform;
		uniform.split_n = 3;
		uniform.base_split_n = 3;
		uniform.extra_m_pairs = 0;
		run_case<true>("split: uniform base_split_n=3",
			1024, 128, 2560, 0, 1.0f, 9, uniform);

		// Uneven split: the first 3 of 4 M pairs carry an extra vocabulary
		// split, so cluster_work crosses both scheduler branches.
		fslce::ForwardGemmTuningSm90<90> uneven;
		uneven.split_n = 3;
		uneven.base_split_n = 2;
		uneven.extra_m_pairs = 3;
		run_case<true>("split: uneven base=2 extra=3",
			1024, 128, 2560, 0, 1.0f, 10, uneven);
		run_case<false>("split: uneven / no entropy",
			1024, 128, 2560, 7168, 0.85f, 11, uneven);

		// Max split with a ragged M tail on top of an uneven schedule.
		fslce::ForwardGemmTuningSm90<90> ragged_split;
		ragged_split.split_n = 2;
		ragged_split.base_split_n = 1;
		ragged_split.extra_m_pairs = 1;
		run_case<true>("split: uneven + ragged M/V",
			648, 136, 1613, 512, 1.0f, 12, ragged_split);
	} catch (const std::exception& error) {
		std::printf("exception: %s\n", error.what());
		return 1;
	}

	std::printf("%s: %d comparison failures\n",
		g_failures == 0 ? "OK" : "FAILED", g_failures);
	return g_failures == 0 ? 0 : 1;
}

#endif  // LIGER_FSLCE_FORWARD_SM90_SELFTEST
