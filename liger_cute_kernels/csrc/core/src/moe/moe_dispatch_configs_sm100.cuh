#pragma once

#include "moe_fwd_bwd_tune_configs.hpp"

// SM100 / Blackwell dispatch config set. Keep this separate from SM90 so
// Blackwell-specific tuning can diverge without touching Hopper dispatch rows.
// The current UMMA MLP1 consumer supports the TileM=128 rows; TileM=64 rows are
// left out of SM100 static dispatch until their TMEM epilogue layout is fixed.

#define LIGER_MOE_FWD_EXTRA_CONFIGS_SM100(XG) \
	XG(8, 128, 64, 4, 64, 256, 64, 4, 64, 4, 4, 128, 64) \
	XG(6, 128, 64, 4, 64, 256, 64, 4, 64, 4, 3, 128, 64)

#define LIGER_MOE_BWD_EXTRA_CONFIGS_SM100(XG) \
	XG(8, 8, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 3, 128, 64) \
	XG(6, 6, 128, 64, 4, 256, 128, 64, 2, 32, 64, 64, 3, 128, 64)

#define LIGER_MOE_FWD_DISPATCH_CONFIGS_SM100(X, XG) \
	LIGER_MOE_FWD_TUNE_CONFIGS_TM128(X) \
	LIGER_MOE_FWD_EXTRA_CONFIGS_SM100(XG)

#define LIGER_MOE_BWD_DISPATCH_CONFIGS_SM100(X, XG) \
	LIGER_MOE_BWD_TUNE_CONFIGS_TM128(X) \
	LIGER_MOE_BWD_EXTRA_CONFIGS_SM100(XG)
