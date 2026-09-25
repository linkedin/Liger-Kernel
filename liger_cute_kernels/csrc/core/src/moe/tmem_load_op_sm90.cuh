#pragma once

#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>

namespace liger {

using namespace cute;

template <int EpiChunkN> struct TmemLoadOpSelector;
template <> struct TmemLoadOpSelector<8>   { using Op = SM100_TMEM_LOAD_32dp32b8x;   };
template <> struct TmemLoadOpSelector<16>  { using Op = SM100_TMEM_LOAD_32dp32b16x;  };
template <> struct TmemLoadOpSelector<32>  { using Op = SM100_TMEM_LOAD_32dp32b32x;  };
template <> struct TmemLoadOpSelector<64>  { using Op = SM100_TMEM_LOAD_32dp32b64x;  };
template <> struct TmemLoadOpSelector<128> { using Op = SM100_TMEM_LOAD_32dp32b128x; };

template <int EpiChunkN>
using TmemLoadOp = typename TmemLoadOpSelector<EpiChunkN>::Op;

}  // namespace liger
