#pragma once

#include "moe_fwd_bwd_tune_configs.hpp"

// SM90 / Hopper dispatch config set. The action macros receive the same row
// shapes as the base tuning menus:
//   FWD: X(TN1, TK1, S1, EC1, TN2, TK2, S2, EC2, ZB, CS, TM)
//        XG(..., GTM) for explicit comm/gemm decoupling.
//   BWD: X(NS2, TN1, TK1, S1, TM3, TN3, TK3, S3, EN1, EN25, EN34, CS, TM)  (13 fields)
//        XG(..., GTM) for explicit comm/gemm decoupling.
// The BWD row starts at `NS2` and has 13 fields, matching the
// 13-param LIGER_MOE_BWD_DISPATCH_ENTRY_* macros and the moe_bwd_fwd_bf16_tuned<>
// template.

#define LIGER_MOE_FWD_EXTRA_CONFIGS_SM90(X, XG) \
	LIGER_MOE_FWD_TUNE_CONFIGS_SM90_ONLY(X)

#define LIGER_MOE_BWD_EXTRA_CONFIGS_SM90(XG)

#define LIGER_MOE_FWD_DISPATCH_CONFIGS_SM90(X, XG) \
	LIGER_MOE_TUNE_CONFIGS(X) \
	LIGER_MOE_FWD_EXTRA_CONFIGS_SM90(X, XG)

#define LIGER_MOE_BWD_DISPATCH_CONFIGS_SM90(X, XG) \
	LIGER_MOE_BWD_TUNE_CONFIGS(X) \
	LIGER_MOE_BWD_EXTRA_CONFIGS_SM90(XG)
