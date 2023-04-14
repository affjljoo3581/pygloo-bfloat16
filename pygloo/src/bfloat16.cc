#include <bfloat16.h>


namespace gloo {

inline bfloat16 float_to_bfloat16(float x) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return *reinterpret_cast<bfloat16*>(reinterpret_cast<uint16_t*>(&x));
#else
  return *reinterpret_cast<bfloat16*>(&(reinterpret_cast<uint16_t*>(&x)[1]));
#endif
}

inline float bfloat16_to_float(bfloat16 x) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  uint16_t y[2] = { x.x, 0 };
#else
  uint16_t y[2] = { 0, x.x };
#endif
  return *reinterpret_cast<float*>(y);
}

/*
template<> void sum<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  for (auto i = 0; i < n; i ++) 
    c[i] = float_to_bfloat16(bfloat16_to_float(a[i]) + bfloat16_to_float(b[i]));
}
*/

template<> void sum<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);
  
  float a_fp32, b_fp32, c_fp32;
  uint16_t *a_fp32_ptr = reinterpret_cast<uint16_t*>(&a_fp32);
  uint16_t *b_fp32_ptr = reinterpret_cast<uint16_t*>(&b_fp32);
  uint16_t *c_fp32_ptr = reinterpret_cast<uint16_t*>(&c_fp32);

  for (auto i = 0; i < n; i ++) {
    a_fp32 = b_fp32 = 0;
    a_fp32_ptr[1] = a[i].x;
    b_fp32_ptr[1] = b[i].x;
    
    c_fp32 = a_fp32 + b_fp32;
    c[i].x = c_fp32_ptr[1];
  }
}

template<> void product<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  for (auto i = 0; i < n; i ++)
    c[i] = float_to_bfloat16(bfloat16_to_float(a[i]) * bfloat16_to_float(b[i]));
}

template<> void min<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  for (auto i = 0; i < n; i ++) {
    float ai = bfloat16_to_float(a[i]);
    float bi = bfloat16_to_float(b[i]);
    c[i] = float_to_bfloat16(ai < bi ? ai : bi);
  }
}

template<> void max<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  for (auto i = 0; i < n; i ++) {
    float ai = bfloat16_to_float(a[i]);
    float bi = bfloat16_to_float(b[i]);
    c[i] = float_to_bfloat16(ai > bi ? ai : bi);
  }
}

} // namespace gloo