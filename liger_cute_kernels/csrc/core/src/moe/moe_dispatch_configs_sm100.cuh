#pragma once

#include "moe_fwd_bwd_tune_configs.hpp"

// SM100 / Blackwell dispatch config set. Keep this separate from SM90 so
// Blackwell-specific tuning can diverge without touching Hopper dispatch rows.
// The current UMMA MLP1 consumer supports the TileM=128 rows; TileM=64 rows are
// left out of SM100 static dispatch until their TMEM epilogue layout is fixed.

#define LIGER_MOE_FWD_EXTRA_CONFIGS_SM100(XG)

// BWD row schema (13 fields, matches LIGER_MOE_BWD_DISPATCH_ENTRY_* and the
// moe_bwd_fwd_bf16_tuned kernel template):
//   X(NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM)
//
// SM100 always selects the paired-CTA MLP3/MLP4 path. TM3/TN3 are kept at
// the canonical joined 256x256 shape, while the existing S3, EN34, and TK3
// fields tune both kernels together.
#define LIGER_MOE_BWD_TUNE_CONFIGS_SM100(X) \
	X(2, 128, 64, 4, 256, 256, 64, 4, 32, 64, 64, 2, 128) \
	X(2, 128, 64, 4, 256, 256, 64, 5, 32, 64, 64, 2, 128) \
	X(8, 128, 64, 4, 256, 256, 64, 5, 32, 64, 64, 2, 128) \
	X(2, 128, 64, 4, 256, 256, 64, 6, 32, 64, 64, 2, 128) \
	X(2, 128, 64, 4, 256, 256, 64, 5, 32, 64, 128, 2, 128)

#define LIGER_MOE_BWD_EXTRA_CONFIGS_SM100(XG)

#define LIGER_MOE_FWD_DISPATCH_CONFIGS_SM100(X, XG) \
	LIGER_MOE_FWD_TUNE_CONFIGS_TM128(X) \
	LIGER_MOE_FWD_EXTRA_CONFIGS_SM100(XG)

#define LIGER_MOE_BWD_DISPATCH_CONFIGS_SM100(X, XG) \
	LIGER_MOE_BWD_TUNE_CONFIGS_SM100(X) \
	LIGER_MOE_BWD_EXTRA_CONFIGS_SM100(XG)
