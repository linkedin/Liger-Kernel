#pragma once

#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

#if LIGER_CUTE_DISPATCH_COMPUTE == 90
#include "tma_copy_atom_sm90.cuh"
#else
#include "tma_copy_atom_sm100.cuh"
#endif
