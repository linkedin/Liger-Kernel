#pragma once

#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

#if LIGER_CUTE_DISPATCH_COMPUTE == 90
#include "mlp5_sm90.cuh"
#else
#include "mlp5_sm100.cuh"
#endif
