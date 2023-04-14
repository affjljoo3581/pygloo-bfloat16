#pragma once

#include <gloo/math.h>


namespace gloo {

struct alignas(2) bfloat16 { uint16_t x; };

template<> void sum<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);
extern template void sum<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);

template<> void product<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);
extern template void product<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);

template<> void min<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);
extern template void min<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);

template<> void max<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);
extern template void max<bfloat16>(void* c_, const void* a_, const void* b_, size_t n);

} // namespace gloo