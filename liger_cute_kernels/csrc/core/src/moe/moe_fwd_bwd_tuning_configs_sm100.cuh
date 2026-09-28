#pragma once

#include "moe_fwd_bwd_tuning_config_types.cuh"
#include "moe_fwd_bwd_tuning_configs_single_sm100.cuh"
#include "moe_fwd_bwd_tuning_configs_multi_sm100.cuh"

namespace liger {

static constexpr TunedConfigFwdBwdTable kTunedConfigTablesSingle[] = {
	{100, kTunedConfigsSingleSm100, kNumTunedConfigsSingleSm100},
};

static constexpr int kNumTunedConfigTablesSingle =
	sizeof(kTunedConfigTablesSingle) / sizeof(kTunedConfigTablesSingle[0]);

static constexpr int kNumTunedConfigsSingle =
	kNumTunedConfigsSingleSm100;

static constexpr TunedConfigFwdBwdTable kTunedConfigTablesMulti[] = {
	{100, kTunedConfigsMultiSm100, kNumTunedConfigsMultiSm100},
};

static constexpr int kNumTunedConfigTablesMulti =
	sizeof(kTunedConfigTablesMulti) / sizeof(kTunedConfigTablesMulti[0]);

static constexpr int kNumTunedConfigsMulti =
	kNumTunedConfigsMultiSm100;

} // namespace liger
