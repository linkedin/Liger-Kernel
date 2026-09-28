#ifndef LIGER_CUTE_DISPATCH_COMPUTE
#define LIGER_CUTE_DISPATCH_COMPUTE 0
#endif

#if LIGER_CUTE_DISPATCH_COMPUTE == 90
#include "moe_sm90.inc"
#else
#include "moe_sm100.inc"
#endif
