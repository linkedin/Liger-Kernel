#pragma once

#include "moe_fwd_bwd_tuning_config_types.cuh"
#include "moe_fwd_bwd_tuning_configs_single_sm90.cuh"
#include "moe_fwd_bwd_tuning_configs_multi_sm90.cuh"

namespace liger {

static constexpr TunedConfigFwdBwdTable kTunedConfigTablesSingle[] = {
	{90, kTunedConfigsSingleSm90, kNumTunedConfigsSingleSm90},
};

static constexpr int kNumTunedConfigTablesSingle =
	sizeof(kTunedConfigTablesSingle) / sizeof(kTunedConfigTablesSingle[0]);

static constexpr int kNumTunedConfigsSingle =
	kNumTunedConfigsSingleSm90;

static constexpr TunedConfigFwdBwdTable kTunedConfigTablesMulti[] = {
	{90, kTunedConfigsMultiSm90, kNumTunedConfigsMultiSm90},
};

static constexpr int kNumTunedConfigTablesMulti =
	sizeof(kTunedConfigTablesMulti) / sizeof(kTunedConfigTablesMulti[0]);

static constexpr int kNumTunedConfigsMulti =
	kNumTunedConfigsMultiSm90;

} // namespace liger
