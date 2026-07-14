#pragma once

#include "moe_fwd_bwd_tune_configs.hpp"

// SM100 / Blackwell dispatch config set. Keep this separate from SM90 so
// Blackwell-specific tuning can diverge without touching Hopper dispatch rows.
// The current UMMA MLP1 consumer supports the TileM=128 rows; TileM=64 rows are
// left out of SM100 static dispatch until their TMEM epilogue layout is fixed.

#define LIGER_MOE_FWD_EXTRA_CONFIGS_SM100(XG)

#define LIGER_MOE_BWD_TUNE_CONFIGS_SM100(X) \
	X(2, 2, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 128) \
	X(2, 8, 128, 64, 4, 128, 256, 64, 2, 32, 64, 64, 2, 128)

#define LIGER_MOE_BWD_EXTRA_CONFIGS_SM100(XG)

#define LIGER_MOE_FWD_DISPATCH_CONFIGS_SM100(X, XG) \
	LIGER_MOE_FWD_TUNE_CONFIGS_TM128(X) \
	LIGER_MOE_FWD_EXTRA_CONFIGS_SM100(XG)

#define LIGER_MOE_BWD_DISPATCH_CONFIGS_SM100(X, XG) \
	LIGER_MOE_BWD_TUNE_CONFIGS_SM100(X) \
	LIGER_MOE_BWD_EXTRA_CONFIGS_SM100(XG)
