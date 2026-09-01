#pragma once

#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

#if LIGER_CUTE_DISPATCH_COMPUTE == 90
#include "mlp2_t_fused_sm90.cuh"
#else
#include "mlp2_t_fused_sm100.cuh"
#endif
