#pragma once

namespace liger {

struct TunedConfigFwdBwd {
	int TK, TKE, D, I;  // shape key
	// Fwd template fields. Runtime NS is selected in-kernel.
	int Fwd_TileN1, Fwd_TileK1, Fwd_Stages1, Fwd_EpiChunkN1;
	int Fwd_TileN2, Fwd_TileK2, Fwd_Stages2, Fwd_EpiChunkN2;
	int Fwd_ZBufferSlots, Fwd_CommNumStages;
	// Bwd template fields. Phase-1 runtime NS is selected in-kernel; NS2
	// remains static for MLP3/MLP4.
	int Bwd_NSplit2;
	int Bwd_TileN1, Bwd_TileK1, Bwd_Stages1;
	int Bwd_TileM3, Bwd_TileN3, Bwd_TileK3, Bwd_Stages3;
	int Bwd_EpiChunkN1, Bwd_EpiChunkN25, Bwd_EpiChunkN34;
	int Bwd_CommNumStages;
	int Fwd_TileM, Bwd_TileM;  // per-direction GEMM tile ∈ {64, 128}; comm tile = 128
	float fwd_ms, bwd_ms, combined_ms;  // per-shape best timings
};

struct TunedConfigFwdBwdTable {
	int Compute;  // device compute capability (Hopper=90, Blackwell=100)
	const TunedConfigFwdBwd* configs;
	int count;
};

} // namespace liger
