#pragma once

#include <cute/atom/copy_traits_sm90_tma.hpp>

namespace liger {

using namespace cute;

template <int Compute>
struct TmaCopyAtomForCompute {
	using Load = SM90_TMA_LOAD;
	using Store = SM90_TMA_STORE;
	using ReduceAdd = SM90_TMA_REDUCE_ADD;
};

template <>
struct TmaCopyAtomForCompute<100> {
	// CUTLASS SM100 1-SM UMMA kernels still use the SM90 global TMA atom
	// names. The SM100_TMA_2SM_* atoms are for 2-SM cluster TMA and require a
	// different TiledCopy/thread layout.
	using Load = SM90_TMA_LOAD;
	using Store = SM90_TMA_STORE;
	using ReduceAdd = SM90_TMA_REDUCE_ADD;
};

template <int Compute>
using TmaLoadAtomForCompute = typename TmaCopyAtomForCompute<Compute>::Load;

template <int Compute>
using TmaStoreAtomForCompute = typename TmaCopyAtomForCompute<Compute>::Store;

template <int Compute>
using TmaReduceAddAtomForCompute = typename TmaCopyAtomForCompute<Compute>::ReduceAdd;

}  // namespace liger
