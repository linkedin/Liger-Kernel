#pragma once

#include <cutlass/arch/reg_reconfig.h>

namespace liger {

template <int ProducerRegisters, int ConsumerRegisters>
struct MoeWarpSpecializedRegistersSm90 {
	static_assert(
		ProducerRegisters >= 24 && ProducerRegisters <= 256 &&
			ConsumerRegisters >= 24 && ConsumerRegisters <= 256);
	static_assert(
		ProducerRegisters % 8 == 0 && ConsumerRegisters % 8 == 0);
	static_assert(
		128 * ProducerRegisters + 256 * ConsumerRegisters <= 64512);

	CUTE_DEVICE static void producer() {
		cutlass::arch::warpgroup_reg_dealloc<ProducerRegisters>();
	}

	CUTE_DEVICE static void consumer() {
		cutlass::arch::warpgroup_reg_alloc<ConsumerRegisters>();
	}
};

} // namespace liger
