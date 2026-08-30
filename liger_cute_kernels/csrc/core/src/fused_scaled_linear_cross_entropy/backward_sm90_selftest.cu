// ═══════════════════════════════════════════════════════════════════════════
// Internal self test for the tensor-parallel SM90 backward path.
//
// Compiled out unless LIGER_FSLCE_BACKWARD_SM90_SELFTEST is defined, so the
// core library's recursive .cu glob picks this file up as an empty translation
// unit. Build it with the production CMake objects so the non-RDC cluster/dW
// kernels and the isolated RDC fused fallback retain their production flags.
//   NVSHMEM_REMOTE_TRANSPORT=none NVSHMEM_DISABLE_NCCL=1 \
//     NVSHMEM_SYMMETRIC_SIZE=3G \
//     nvshmrun -n 4 ./backward_selftest
//
// Diagnostics (environment variables, all optional):
//   FSLCE_POISON=s|z|sz  wipe the symmetric dX staging (s) and/or the dZ wave
//                        workspace (z) between the two attempts of a case, so
//                        a case that only passes on the second attempt cannot
//                        hide behind buffers it never wrote.
//   FSLCE_CHANNELS=<n>   override the ignored legacy channel argument.
//   FSLCE_DELAY_WARP=0|1 inject delay into one CTA-local comm warp.
//   FSLCE_DELAY_ITERS=n  number of 64-cycle sleeps before each owned chunk.
//   FSLCE_ATTEMPTS=n     repeat every case n times to expose ordering races.
//
// It runs at any PE count, including one:
//   * dX is the FP32 TP *sum* over shards followed by one conversion to
//     X.dtype (BF16 in this specialization), so every PE must end up with the same
//     globally correct dX. The CPU reference rebuilds the whole vocabulary
//     from the shared seed and sums every shard's contribution, so a dropped
//     reduction, a wrong team, a mis-sliced tail group or a stale ring slot
//     all show up.
//   * dW is rank local, so it is checked against this shard's own tiles only.
// TilesPerReduce is swept, including one-tile chunks that alternate whole
// messages between the two comm warps. Legacy channel values are varied to
// verify that they no longer affect CTA ownership.
// ═══════════════════════════════════════════════════════════════════════════

#if defined(LIGER_FSLCE_BACKWARD_SM90_SELFTEST)

#include <arpa/inet.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backward_gemm_sm90.cuh"
#include "buffer_pool.cuh"
#include "workspace.cuh"

namespace {

namespace fslce = liger::fused_scaled_linear_cross_entropy;

using Element = __nv_bfloat16;

int g_pe = 0;
int g_n_pes = 1;
int g_failures = 0;
constexpr int kDefaultAttempts = 2;
constexpr int kMaxTokens = 1280;
constexpr int kMaxHidden = 8448;
constexpr int kMaxLocalVocab = 1024;
constexpr int kMaxTilesPerReduce = 4;
constexpr int kLegacyMaxCommChannels = 4;

void send_all(int fd, const void* data, std::size_t bytes) {
	const char* cursor = static_cast<const char*>(data);
	while (bytes != 0) {
		ssize_t sent = send(fd, cursor, bytes, 0);
		if (sent <= 0) std::exit(1);
		cursor += sent;
		bytes -= static_cast<std::size_t>(sent);
	}
}

void recv_all(int fd, void* data, std::size_t bytes) {
	char* cursor = static_cast<char*>(data);
	while (bytes != 0) {
		ssize_t received = recv(fd, cursor, bytes, 0);
		if (received <= 0) std::exit(1);
		cursor += received;
		bytes -= static_cast<std::size_t>(received);
	}
}

void exchange_unique_id(
		int rank,
		int ranks,
		const char* master,
		int port,
		nvshmemx_uniqueid_t* id) {
	if (rank == 0) {
		int server = socket(AF_INET, SOCK_STREAM, 0);
		int reuse = 1;
		setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
		sockaddr_in address = {};
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl(INADDR_ANY);
		address.sin_port = htons(static_cast<std::uint16_t>(port));
		if (bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
			listen(server, ranks) != 0) {
			std::exit(1);
		}
		for (int remote = 1; remote < ranks; ++remote) {
			int peer = accept(server, nullptr, nullptr);
			if (peer < 0) std::exit(1);
			send_all(peer, id, sizeof(*id));
			close(peer);
		}
		close(server);
		return;
	}

	sockaddr_in address = {};
	address.sin_family = AF_INET;
	address.sin_port = htons(static_cast<std::uint16_t>(port));
	if (inet_pton(AF_INET, master, &address.sin_addr) != 1) std::exit(1);
	for (int attempt = 0; attempt < 200; ++attempt) {
		int client = socket(AF_INET, SOCK_STREAM, 0);
		if (connect(
				client,
				reinterpret_cast<sockaddr*>(&address),
				sizeof(address)) == 0) {
			recv_all(client, id, sizeof(*id));
			close(client);
			return;
		}
		close(client);
		usleep(100000);
	}
	std::exit(1);
}

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

struct Reference {
	std::vector<float> dx;       // [tokens, hidden], TP-summed
	std::vector<float> dw_local; // [local_vocab, hidden], this shard
	std::vector<float> lse;      // [tokens]
	std::vector<float> entropy;  // [tokens]
};

// Full-vocabulary CPU reference. Reproduces the kernel's FP16 logit staging
// round so the two agree on the softmax input bit for bit.
Reference reference(
		const std::vector<float>& x,
		const std::vector<float>& weight_global,
		const std::vector<std::int64_t>& target,
		const std::vector<float>& grad_output,
		const std::vector<float>& entropy_grad,
		int tokens,
		int hidden,
		int global_vocab,
		int local_vocab,
		int vocab_start,
		std::int64_t ignore_index,
		float inverse_temperature,
		bool return_entropy) {
	Reference out;
	out.dx.assign(static_cast<std::size_t>(tokens) * hidden, 0.0f);
	out.dw_local.assign(
		static_cast<std::size_t>(local_vocab) * hidden, 0.0f);
	out.lse.assign(tokens, 0.0f);
	out.entropy.assign(tokens, 0.0f);

	std::vector<float> scaled(global_vocab);
	std::vector<float> dz(global_vocab);
	for (int m = 0; m < tokens; ++m) {
		float row_max = -1.0e38f;
		for (int v = 0; v < global_vocab; ++v) {
			float dot = 0.0f;
			for (int k = 0; k < hidden; ++k) {
				dot += x[static_cast<std::size_t>(m) * hidden + k] *
					weight_global[static_cast<std::size_t>(v) * hidden + k];
			}
			// float -> half -> float, matching the reference's FP16 logit
			// staging buffer.
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
		out.lse[m] = lse;
		out.entropy[m] = lse - static_cast<float>(weighted / sum);

		bool ignored = target[m] == ignore_index;
		float scale = ignored ? 0.0f : grad_output[m];
		float escale = (ignored || !return_entropy) ? 0.0f : entropy_grad[m];
		for (int v = 0; v < global_vocab; ++v) {
			float value = 0.0f;
			if (scale != 0.0f || escale != 0.0f) {
				float probability = std::exp(scaled[v] - lse);
				value = probability *
					(scale + (lse - out.entropy[m] - scaled[v]) * escale);
			}
			if (!ignored && v == static_cast<int>(target[m])) value -= scale;
			value *= inverse_temperature;
			// dZ is materialised in BF16 by the kernel.
			dz[v] = __bfloat162float(__float2bfloat16(value));
		}

		for (int h = 0; h < hidden; ++h) {
			float reduced = 0.0f;
			for (int pe = 0; pe < global_vocab / local_vocab; ++pe) {
				float partial = 0.0f;
				int shard_start = pe * local_vocab;
				for (int v = 0; v < local_vocab; ++v) {
					int global_v = shard_start + v;
					partial += dz[global_v] *
						weight_global[
							static_cast<std::size_t>(global_v) * hidden + h];
				}
				reduced += partial;
			}
			out.dx[static_cast<std::size_t>(m) * hidden + h] = reduced;
		}
		for (int v = 0; v < local_vocab; ++v) {
			float value = dz[vocab_start + v];
			if (value == 0.0f) continue;
			for (int h = 0; h < hidden; ++h) {
				out.dw_local[static_cast<std::size_t>(v) * hidden + h] +=
					value * x[static_cast<std::size_t>(m) * hidden + h];
			}
		}
	}
	return out;
}

struct Comparison {
	int bad;
	float max_absolute;
	float max_relative;
};

Comparison compare(
		const char* label,
		const std::vector<float>& actual,
		const std::vector<float>& expected,
		float absolute,
		float relative,
		int reported) {
	Comparison result = {};
	for (std::size_t i = 0; i < expected.size(); ++i) {
		float error = fabsf(actual[i] - expected[i]);
		float relative_error = error / fmaxf(fabsf(expected[i]), 1.0e-6f);
		result.max_absolute = fmaxf(result.max_absolute, error);
		result.max_relative = fmaxf(result.max_relative, relative_error);
		float tolerance = absolute + relative * fabsf(expected[i]);
		if (!(error <= tolerance)) {
			if (result.bad < reported) {
				std::printf(
					"[pe %d]   FAIL %-4s [%zu]: got %12.6f want %12.6f "
					"(tol %.6f)\n",
					g_pe, label, i, actual[i], expected[i], tolerance);
			}
			++result.bad;
		}
	}
	return result;
}

struct CaseShape {
	int tokens;
	int hidden;
	int local_vocab;
	float temperature;
	std::uint64_t seed;
};

void check_chunk_ownership() {
	using TestCommConfig = fslce::DxCommConfig<
		fslce::BackwardGemmConfigSm90<90>, fslce::kDxRingStages, 2>;
	constexpr int kGroupsPerWave = 136;
	constexpr int kNumCtas = 132;
	constexpr int kNumWaves = 3;
	int seen[kNumCtas] = {};
	for (int group = 0; group < kNumWaves * kGroupsPerWave; ++group) {
		int unit = group % kGroupsPerWave;
		int cta = unit % kNumCtas;
		int index = fslce::dx_cta_group_index(
			group, kGroupsPerWave, kNumCtas);
		fslce::DxCtaGroupSlot slot =
			fslce::dx_cta_group_slot<TestCommConfig>(index);
		bool valid = index == seen[cta] &&
			slot.comm_warp == 0 &&
			slot.index_in_warp == index &&
			slot.stage == slot.index_in_warp % fslce::kDxRingStages &&
			slot.pass == slot.index_in_warp / fslce::kDxRingStages;
		if (!valid) {
			std::printf(
				"[pe %d] invalid dX chunk owner: group=%d cta=%d "
				"index=%d expected=%d warp=%d warp_index=%d stage=%d pass=%d\n",
				g_pe, group, cta, index, seen[cta], slot.comm_warp,
				slot.index_in_warp, slot.stage, slot.pass);
			++g_failures;
		}
		++seen[cta];
	}
}

template <bool ReturnEntropy>
void run_case(
		const std::string& name,
		const CaseShape& shape,
		int tiles_per_reduce,
		int num_comm_channels) {  // NOLINT: mutated by FSLCE_CHANNELS
	const char* filter = std::getenv("FSLCE_FILTER");
	if (filter != nullptr && name.find(filter) == std::string::npos) return;

	// ABI-compatibility diagnostic: this value no longer routes groups.
	const char* channel_override = std::getenv("FSLCE_CHANNELS");
	if (channel_override != nullptr) {
		num_comm_channels = std::atoi(channel_override);
	}
	const int tokens = shape.tokens;
	const int hidden = shape.hidden;
	const int local_vocab = shape.local_vocab;
	const int global_vocab = local_vocab * g_n_pes;
	const int vocab_start = g_pe * local_vocab;
	const float scale = 3.0f / std::sqrt(static_cast<float>(hidden));

	std::vector<float> host_x(static_cast<std::size_t>(tokens) * hidden);
	for (std::size_t i = 0; i < host_x.size(); ++i) {
		host_x[i] = __bfloat162float(
			__float2bfloat16(element_at(shape.seed, i, scale)));
	}
	std::vector<float> host_w(
		static_cast<std::size_t>(global_vocab) * hidden);
	for (std::size_t i = 0; i < host_w.size(); ++i) {
		host_w[i] = __bfloat162float(__float2bfloat16(
			element_at(shape.seed ^ 0xABCDEF01ULL, i, scale)));
	}

	std::vector<std::int64_t> host_target(tokens);
	std::vector<float> host_grad_output(tokens);
	std::vector<float> host_entropy_grad(tokens);
	for (int m = 0; m < tokens; ++m) {
		if (m % 13 == 5) {
			host_target[m] = -100;  // ignore_index
		} else {
			std::uint64_t state = shape.seed ^ (0x5DEECE66DULL * (m + 1));
			state = state * 6364136223846793005ULL + 1442695040888963407ULL;
			host_target[m] = static_cast<std::int64_t>(
				(state >> 17) % static_cast<std::uint64_t>(global_vocab));
		}
		host_grad_output[m] = 0.5f + 0.5f * element_at(shape.seed ^ 7u, m, 1.0f);
		host_entropy_grad[m] = 0.25f * element_at(shape.seed ^ 11u, m, 1.0f);
	}

	Reference expected = reference(
		host_x, host_w, host_target, host_grad_output, host_entropy_grad,
		tokens, hidden, global_vocab, local_vocab, vocab_start, -100,
		1.0f / shape.temperature, ReturnEntropy);

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
	Element* device_dx = nullptr;
	Element* device_dw = nullptr;
	std::int64_t* device_target = nullptr;
	float* device_scalars = nullptr;  // grad_output | lse | entropy | e_grad
	CUDA_OK(cudaMalloc(&device_x, device_x_host.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_w, device_w_host.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_dx, device_x_host.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_dw, device_w_host.size() * sizeof(Element)));
	CUDA_OK(cudaMalloc(&device_target, tokens * sizeof(std::int64_t)));
	CUDA_OK(cudaMalloc(&device_scalars, 4 * tokens * sizeof(float)));
	CUDA_OK(cudaMemcpy(device_x, device_x_host.data(),
		device_x_host.size() * sizeof(Element), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_w, device_w_host.data(),
		device_w_host.size() * sizeof(Element), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_target, host_target.data(),
		tokens * sizeof(std::int64_t), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_scalars + 0 * tokens, host_grad_output.data(),
		tokens * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_scalars + 1 * tokens, expected.lse.data(),
		tokens * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_scalars + 2 * tokens, expected.entropy.data(),
		tokens * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_OK(cudaMemcpy(device_scalars + 3 * tokens, host_entropy_grad.data(),
		tokens * sizeof(float), cudaMemcpyHostToDevice));
	// Poison the outputs so a skipped store is a failure, not a pass.
	CUDA_OK(cudaMemset(device_dx, 0x7f,
		device_x_host.size() * sizeof(Element)));
	CUDA_OK(cudaMemset(device_dw, 0x7f,
		device_w_host.size() * sizeof(Element)));

	fslce::BackwardScratch scratch =
		fslce::reserve_backward_scratch(local_vocab);

	// Diagnostic: FSLCE_POISON=s wipes the symmetric dX staging between
	// attempts, =z wipes the dZ wave workspace, =sz both. Any case that only
	// passes without poisoning is reading a buffer it never wrote.
	const char* poison = std::getenv("FSLCE_POISON");
	bool poison_staging = poison != nullptr && std::strchr(poison, 's');
	bool poison_dz = poison != nullptr && std::strchr(poison, 'z');
	fslce::DxReduceWorkspace<float> staging =
		fslce::reserve_dx_reduce_workspace(
			tiles_per_reduce,
			fslce::kDxRingStages,
			fslce::backward_dx_resident_cta_capacity());
	std::size_t staging_bytes =
		fslce::backward_dx_staging_bytes(kMaxTilesPerReduce);

	fslce::BackwardTpParamsSm90<90> params;
	params.gemm.x = device_x;
	params.gemm.weight = device_w;
	params.gemm.target = device_target;
	params.gemm.grad_output = device_scalars + 0 * tokens;
	params.gemm.lse = device_scalars + 1 * tokens;
	params.gemm.entropy = device_scalars + 2 * tokens;
	params.gemm.entropy_grad = device_scalars + 3 * tokens;
	params.gemm.grad_input = device_dx;
	params.gemm.grad_weight = device_dw;
	params.gemm.dz_workspace = scratch.dz_workspace;
	params.gemm.dz_workspace_bytes = scratch.dz_workspace_bytes;
	params.gemm.grid_barrier = scratch.grid_barrier;
	params.gemm.tokens = tokens;
	params.gemm.hidden = hidden;
	params.gemm.local_vocab = local_vocab;
	params.gemm.vocab_start = vocab_start;
	params.gemm.ignore_index = -100;
	params.gemm.inverse_temperature = 1.0f / shape.temperature;
	const char* delay_warp = std::getenv("FSLCE_DELAY_WARP");
	const char* delay_iters = std::getenv("FSLCE_DELAY_ITERS");
	const char* attempts_env = std::getenv("FSLCE_ATTEMPTS");
	int attempts = attempts_env == nullptr
		? kDefaultAttempts
		: std::atoi(attempts_env);
	if (attempts < 1) attempts = 1;
	if (delay_warp != nullptr) {
		params.gemm.dx_comm_delay_warp = std::atoi(delay_warp);
	}
	if (delay_iters != nullptr) {
		params.gemm.dx_comm_delay_iterations = std::atoi(delay_iters);
	}
	params.tiles_per_reduce = tiles_per_reduce;
	params.num_comm_channels = num_comm_channels;

	// Each case is launched twice: a first-launch-only failure is a warm-up
	// bug, a failure that moves between repetitions is a race.
	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_OK(cudaEventCreate(&start));
	CUDA_OK(cudaEventCreate(&stop));
	for (int attempt = 0; attempt < attempts; ++attempt) {
		if (poison_staging) {
			CUDA_OK(cudaMemset(staging.partial, 0, staging_bytes));
			CUDA_OK(cudaMemset(staging.reduced, 0, staging_bytes));
		}
		if (poison_dz) {
			CUDA_OK(cudaMemset(
				scratch.dz_workspace, 0, scratch.dz_workspace_bytes));
		}
		CUDA_OK(cudaMemset(device_dx, 0x7f,
			device_x_host.size() * sizeof(Element)));
		CUDA_OK(cudaMemset(device_dw, 0x7f,
			device_w_host.size() * sizeof(Element)));
		nvshmem_barrier_all();
		CUDA_OK(cudaEventRecord(start));
		fslce::fused_linear_scaled_cross_entropy_backward<
			ReturnEntropy, 90>(params, nullptr);
		CUDA_OK(cudaEventRecord(stop));
		CUDA_OK(cudaGetLastError());
		CUDA_OK(cudaEventSynchronize(stop));
		float elapsed_ms = 0.0f;
		CUDA_OK(cudaEventElapsedTime(&elapsed_ms, start, stop));
		nvshmem_barrier_all();

		std::vector<Element> raw_dx(device_x_host.size());
		std::vector<Element> raw_dw(device_w_host.size());
		CUDA_OK(cudaMemcpy(raw_dx.data(), device_dx,
			raw_dx.size() * sizeof(Element), cudaMemcpyDeviceToHost));
		CUDA_OK(cudaMemcpy(raw_dw.data(), device_dw,
			raw_dw.size() * sizeof(Element), cudaMemcpyDeviceToHost));
		std::vector<float> got_dx(raw_dx.size());
		std::vector<float> got_dw(raw_dw.size());
		for (std::size_t i = 0; i < raw_dx.size(); ++i) {
			got_dx[i] = __bfloat162float(raw_dx[i]);
		}
		for (std::size_t i = 0; i < raw_dw.size(); ++i) {
			got_dw[i] = __bfloat162float(raw_dw[i]);
		}

		Comparison dx = compare(
			"dX", got_dx, expected.dx, 6.0e-3f, 4.0e-2f, 4);
		Comparison dw = compare(
			"dW", got_dw, expected.dw_local, 8.0e-3f, 4.0e-2f, 4);
		g_failures += dx.bad + dw.bad;

		if (g_pe == 0) {
			std::printf(
				"%-42s #%d %s  (PEs=%d M=%d H=%d Vloc=%d T=%.2f entropy=%d "
				"TPR=%d legacy_chan=%d dx_bad=%d dw_bad=%d ms=%.3f "
				"dx_max_abs=%.6g dx_max_rel=%.6g "
				"dw_max_abs=%.6g dw_max_rel=%.6g)\n",
				name.c_str(), attempt,
				(dx.bad + dw.bad) == 0 ? "PASS" : "FAIL",
				g_n_pes, tokens, hidden, local_vocab, shape.temperature,
				static_cast<int>(ReturnEntropy), tiles_per_reduce,
				num_comm_channels, dx.bad, dw.bad, elapsed_ms,
				dx.max_absolute, dx.max_relative,
				dw.max_absolute, dw.max_relative);
		}
	}
	CUDA_OK(cudaEventDestroy(start));
	CUDA_OK(cudaEventDestroy(stop));

	CUDA_OK(cudaFree(device_x));
	CUDA_OK(cudaFree(device_w));
	CUDA_OK(cudaFree(device_dx));
	CUDA_OK(cudaFree(device_dw));
	CUDA_OK(cudaFree(device_target));
	CUDA_OK(cudaFree(device_scalars));
}

}  // namespace

int main(int argc, char** argv) {
	if (argc == 6) {
		int rank = std::atoi(argv[1]);
		int ranks = std::atoi(argv[2]);
		int device = std::atoi(argv[3]);
		CUDA_OK(cudaSetDevice(device));
		CUDA_OK(cudaFree(nullptr));
		nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
		if (rank == 0 && nvshmemx_get_uniqueid(&id) != 0) return 1;
		exchange_unique_id(
			rank, ranks, argv[4], std::atoi(argv[5]), &id);
		nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
		nvshmemx_set_attr_uniqueid_args(
			rank, ranks, &id, &attributes);
		if (nvshmemx_init_attr(
				NVSHMEMX_INIT_WITH_UNIQUEID, &attributes) != 0) {
			return 1;
		}
	} else {
		nvshmem_init();
	}
	g_pe = nvshmem_my_pe();
	g_n_pes = nvshmem_n_pes();
	CUDA_OK(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));

	cudaDeviceProp properties = {};
	CUDA_OK(cudaGetDeviceProperties(
		&properties, nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
	if (properties.major != 9) {
		if (g_pe == 0) {
			std::printf("skipping: requires Hopper (sm90), found sm_%d%d\n",
				properties.major, properties.minor);
		}
		nvshmem_finalize();
		return 0;
	}

	if (g_pe == 0) {
		std::printf("tensor-parallel backward over %d PEs\n", g_n_pes);
	}
	check_chunk_ownership();

	// Capacities and multicast mappings are immutable once allocated, so
	// configure for the largest shape and the widest knobs used below.
	// Collective.
	fslce::configure_backward_tp_symmetric(
		kMaxTokens, kMaxHidden, kMaxLocalVocab,
		kMaxTilesPerReduce, kLegacyMaxCommChannels,
		static_cast<std::int64_t>(NVSHMEM_TEAM_WORLD));
	if (g_pe == 0) {
		std::printf(
			"symmetric staging %.2f MiB, device scratch %.2f MiB, "
			"resident_ctas=%d nvls_team_size=%d\n",
			fslce::backward_tp_pool_symmetric_bytes(
				kMaxTokens, kMaxHidden,
				kMaxTilesPerReduce, kLegacyMaxCommChannels) /
				(1024.0 * 1024.0),
			fslce::backward_tp_pool_device_bytes(kMaxLocalVocab) /
				(1024.0 * 1024.0),
			fslce::backward_dx_resident_cta_capacity(),
			fslce::backward_dx_team_size());
	}

	// One N256 hidden tile: no grouping, exercises the simplest ring path.
	const CaseShape small{256, 256, 512, 1.0f, 101};
	// Three N256 hidden tiles with a ragged last one (648 = 2 * 256 + 136), so
	// TilesPerReduce 2 produces a {2, 1} tail split and 4 a {3} tail.
	const CaseShape ragged{200, 648, 333, 0.8f, 103};
	// Two waves: exercises the dW TMA reduce-add path and a ring that wraps.
	const CaseShape two_waves{1280, 512, 512, 1.0f, 105};
	// Five K64 vocabulary tiles: H2048 must fall back to split-K1 rather than
	// dropping the final odd tile. Non-unit temperature checks the chain rule.
	const CaseShape odd_k_tiles{128, 2048, 320, 0.8f, 106};
	// 136 dX groups with TPR=1 on a 132-SM H100: launches the full resident
	// grid and makes four CTAs wrap to a second CTA-owned group.
	const CaseShape full_grid{1, 4352, 64, 1.0f, 107};
	// 136 two-tile chunks: the same full-grid wrap, now proving adjacent
	// grouped messages alternate between the CTA's two comm warps.
	const CaseShape full_grid_grouped{1, 8448, 64, 1.0f, 109};

	try {
		run_case<false>("dX/dW aligned, no entropy", small, 2, 2);
		run_case<true>("dX/dW aligned, entropy", small, 2, 2);
		run_case<true>("dX/dW ragged M/H/V, TilesPerReduce=1", ragged, 1, 2);
		run_case<true>("dX/dW ragged M/H/V, TilesPerReduce=2", ragged, 2, 2);
		run_case<true>("dX/dW ragged M/H/V, TilesPerReduce=4", ragged, 4, 2);
		run_case<true>("dX/dW ragged, legacy channel arg=1", ragged, 2, 1);
		run_case<true>("dX/dW ragged, legacy channel arg=4", ragged, 2, 4);
		run_case<false>("dX/dW two waves, TilesPerReduce=2", two_waves, 2, 4);
		run_case<true>(
			"dX/dW H2048 odd K tiles and temperature",
			odd_k_tiles,
			1,
			2);
		run_case<false>("dX/dW full resident CTA grid", full_grid, 1, 2);
		run_case<false>(
			"dX/dW full grid, alternating two-tile chunks",
			full_grid_grouped, 2, 2);
	} catch (const std::exception& error) {
		std::printf("[pe %d] exception: %s\n", g_pe, error.what());
		fslce::reset_fslce_tp_configuration();
		liger::global_buffer_pool().clear();
		nvshmem_finalize();
		return 1;
	}

	// Make every rank's verdict visible before PE 0 prints the summary.
	// Pooled, not raw nvshmem_malloc: this module never runs its own symmetric
	// allocation lifecycle.
	auto& pool = liger::global_buffer_pool();
	int* local_slot = static_cast<int*>(
		pool.get_symmetric("fslce_bwd_selftest_verdict_local", sizeof(int)));
	int* total_slot = static_cast<int*>(
		pool.get_symmetric("fslce_bwd_selftest_verdict_total", sizeof(int)));
	CUDA_OK(cudaMemcpy(
		local_slot, &g_failures, sizeof(int), cudaMemcpyHostToDevice));
	nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, total_slot, local_slot, 1);
	int host_total = 0;
	CUDA_OK(cudaMemcpy(
		&host_total, total_slot, sizeof(int), cudaMemcpyDeviceToHost));

	if (g_pe == 0) {
		std::printf("%s: %d comparison failures across %d PEs\n",
			host_total == 0 ? "OK" : "FAILED", host_total, g_n_pes);
	}
	// Release the pooled allocations while NVSHMEM is still up: the pool is a
	// function-local static whose destructor would otherwise run after
	// nvshmem_finalize().
	fslce::reset_fslce_tp_configuration();
	liger::global_buffer_pool().clear();
	nvshmem_finalize();
	return host_total == 0 ? 0 : 1;
}

#endif  // LIGER_FSLCE_BACKWARD_SM90_SELFTEST
