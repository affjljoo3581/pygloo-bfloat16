#include <bfloat16.h>

#ifdef __AVX__
#include <immintrin.h>
#endif

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define BF16_OFFSET 0
#else
#define BF16_OFFSET 1
#endif

namespace gloo {

template<> void sum<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  float a_fp32 = 0, b_fp32 = 0, c_fp32;
  uint16_t *a_fp32_ptr = reinterpret_cast<uint16_t*>(&a_fp32);
  uint16_t *b_fp32_ptr = reinterpret_cast<uint16_t*>(&b_fp32);
  uint16_t *c_fp32_ptr = reinterpret_cast<uint16_t*>(&c_fp32);

  size_t i = 0;

#ifdef __AVX__
  __m256 va32f = _mm256_setzero_ps();
  __m256 vb32f = _mm256_setzero_ps();
  __m128i vc16i;

  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m128i va16i = _mm_loadu_si128((__m128i*)(&a[i]));
    __m128i vb16i = _mm_loadu_si128((__m128i*)(&b[i]));

    *((uint16_t*) &va32f + 0 + BF16_OFFSET) = *((uint16_t*) &va16i + 0);
    *((uint16_t*) &va32f + 2 + BF16_OFFSET) = *((uint16_t*) &va16i + 1);
    *((uint16_t*) &va32f + 4 + BF16_OFFSET) = *((uint16_t*) &va16i + 2);
    *((uint16_t*) &va32f + 6 + BF16_OFFSET) = *((uint16_t*) &va16i + 3);
    *((uint16_t*) &va32f + 8 + BF16_OFFSET) = *((uint16_t*) &va16i + 4);
    *((uint16_t*) &va32f + 10 + BF16_OFFSET) = *((uint16_t*) &va16i + 5);
    *((uint16_t*) &va32f + 12 + BF16_OFFSET) = *((uint16_t*) &va16i + 6);
    *((uint16_t*) &va32f + 14 + BF16_OFFSET) = *((uint16_t*) &va16i + 7);

    *((uint16_t*) &vb32f + 0 + BF16_OFFSET) = *((uint16_t*) &vb16i + 0);
    *((uint16_t*) &vb32f + 2 + BF16_OFFSET) = *((uint16_t*) &vb16i + 1);
    *((uint16_t*) &vb32f + 4 + BF16_OFFSET) = *((uint16_t*) &vb16i + 2);
    *((uint16_t*) &vb32f + 6 + BF16_OFFSET) = *((uint16_t*) &vb16i + 3);
    *((uint16_t*) &vb32f + 8 + BF16_OFFSET) = *((uint16_t*) &vb16i + 4);
    *((uint16_t*) &vb32f + 10 + BF16_OFFSET) = *((uint16_t*) &vb16i + 5);
    *((uint16_t*) &vb32f + 12 + BF16_OFFSET) = *((uint16_t*) &vb16i + 6);
    *((uint16_t*) &vb32f + 14 + BF16_OFFSET) = *((uint16_t*) &vb16i + 7);

    __m256 vc32f = _mm256_add_ps(va32f, vb32f);
    *((uint16_t*) &vc16i + 0) = *((uint16_t*) &vc32f + 0 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 1) = *((uint16_t*) &vc32f + 2 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 2) = *((uint16_t*) &vc32f + 4 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 3) = *((uint16_t*) &vc32f + 6 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 4) = *((uint16_t*) &vc32f + 8 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 5) = *((uint16_t*) &vc32f + 10 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 6) = *((uint16_t*) &vc32f + 12 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 7) = *((uint16_t*) &vc32f + 14 + BF16_OFFSET);

    _mm_storeu_si128((__m128i*)(&c[i]), vc16i);
  }
#endif

  for (; i < n; i ++) {
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

  float a_fp32 = 0, b_fp32 = 0, c_fp32;
  uint16_t *a_fp32_ptr = reinterpret_cast<uint16_t*>(&a_fp32);
  uint16_t *b_fp32_ptr = reinterpret_cast<uint16_t*>(&b_fp32);
  uint16_t *c_fp32_ptr = reinterpret_cast<uint16_t*>(&c_fp32);

  size_t i = 0;

#ifdef __AVX__
  __m256 va32f = _mm256_setzero_ps();
  __m256 vb32f = _mm256_setzero_ps();
  __m128i vc16i;

  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m128i va16i = _mm_loadu_si128((__m128i*)(&a[i]));
    __m128i vb16i = _mm_loadu_si128((__m128i*)(&b[i]));

    *((uint16_t*) &va32f + 0 + BF16_OFFSET) = *((uint16_t*) &va16i + 0);
    *((uint16_t*) &va32f + 2 + BF16_OFFSET) = *((uint16_t*) &va16i + 1);
    *((uint16_t*) &va32f + 4 + BF16_OFFSET) = *((uint16_t*) &va16i + 2);
    *((uint16_t*) &va32f + 6 + BF16_OFFSET) = *((uint16_t*) &va16i + 3);
    *((uint16_t*) &va32f + 8 + BF16_OFFSET) = *((uint16_t*) &va16i + 4);
    *((uint16_t*) &va32f + 10 + BF16_OFFSET) = *((uint16_t*) &va16i + 5);
    *((uint16_t*) &va32f + 12 + BF16_OFFSET) = *((uint16_t*) &va16i + 6);
    *((uint16_t*) &va32f + 14 + BF16_OFFSET) = *((uint16_t*) &va16i + 7);

    *((uint16_t*) &vb32f + 0 + BF16_OFFSET) = *((uint16_t*) &vb16i + 0);
    *((uint16_t*) &vb32f + 2 + BF16_OFFSET) = *((uint16_t*) &vb16i + 1);
    *((uint16_t*) &vb32f + 4 + BF16_OFFSET) = *((uint16_t*) &vb16i + 2);
    *((uint16_t*) &vb32f + 6 + BF16_OFFSET) = *((uint16_t*) &vb16i + 3);
    *((uint16_t*) &vb32f + 8 + BF16_OFFSET) = *((uint16_t*) &vb16i + 4);
    *((uint16_t*) &vb32f + 10 + BF16_OFFSET) = *((uint16_t*) &vb16i + 5);
    *((uint16_t*) &vb32f + 12 + BF16_OFFSET) = *((uint16_t*) &vb16i + 6);
    *((uint16_t*) &vb32f + 14 + BF16_OFFSET) = *((uint16_t*) &vb16i + 7);

    __m256 vc32f = _mm256_mul_ps(va32f, vb32f);
    *((uint16_t*) &vc16i + 0) = *((uint16_t*) &vc32f + 0 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 1) = *((uint16_t*) &vc32f + 2 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 2) = *((uint16_t*) &vc32f + 4 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 3) = *((uint16_t*) &vc32f + 6 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 4) = *((uint16_t*) &vc32f + 8 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 5) = *((uint16_t*) &vc32f + 10 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 6) = *((uint16_t*) &vc32f + 12 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 7) = *((uint16_t*) &vc32f + 14 + BF16_OFFSET);

    _mm_storeu_si128((__m128i*)(&c[i]), vc16i);
  }
#endif

  for (; i < n; i ++) {
    a_fp32_ptr[1] = a[i].x;
    b_fp32_ptr[1] = b[i].x;
    
    c_fp32 = a_fp32 * b_fp32;
    c[i].x = c_fp32_ptr[1];
  }
}

template<> void min<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
  bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  float a_fp32 = 0, b_fp32 = 0, c_fp32;
  uint16_t *a_fp32_ptr = reinterpret_cast<uint16_t*>(&a_fp32);
  uint16_t *b_fp32_ptr = reinterpret_cast<uint16_t*>(&b_fp32);
  uint16_t *c_fp32_ptr = reinterpret_cast<uint16_t*>(&c_fp32);

  size_t i = 0;

#ifdef __AVX__
  __m256 va32f = _mm256_setzero_ps();
  __m256 vb32f = _mm256_setzero_ps();
  __m128i vc16i;

  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m128i va16i = _mm_loadu_si128((__m128i*)(&a[i]));
    __m128i vb16i = _mm_loadu_si128((__m128i*)(&b[i]));

    *((uint16_t*) &va32f + 0 + BF16_OFFSET) = *((uint16_t*) &va16i + 0);
    *((uint16_t*) &va32f + 2 + BF16_OFFSET) = *((uint16_t*) &va16i + 1);
    *((uint16_t*) &va32f + 4 + BF16_OFFSET) = *((uint16_t*) &va16i + 2);
    *((uint16_t*) &va32f + 6 + BF16_OFFSET) = *((uint16_t*) &va16i + 3);
    *((uint16_t*) &va32f + 8 + BF16_OFFSET) = *((uint16_t*) &va16i + 4);
    *((uint16_t*) &va32f + 10 + BF16_OFFSET) = *((uint16_t*) &va16i + 5);
    *((uint16_t*) &va32f + 12 + BF16_OFFSET) = *((uint16_t*) &va16i + 6);
    *((uint16_t*) &va32f + 14 + BF16_OFFSET) = *((uint16_t*) &va16i + 7);

    *((uint16_t*) &vb32f + 0 + BF16_OFFSET) = *((uint16_t*) &vb16i + 0);
    *((uint16_t*) &vb32f + 2 + BF16_OFFSET) = *((uint16_t*) &vb16i + 1);
    *((uint16_t*) &vb32f + 4 + BF16_OFFSET) = *((uint16_t*) &vb16i + 2);
    *((uint16_t*) &vb32f + 6 + BF16_OFFSET) = *((uint16_t*) &vb16i + 3);
    *((uint16_t*) &vb32f + 8 + BF16_OFFSET) = *((uint16_t*) &vb16i + 4);
    *((uint16_t*) &vb32f + 10 + BF16_OFFSET) = *((uint16_t*) &vb16i + 5);
    *((uint16_t*) &vb32f + 12 + BF16_OFFSET) = *((uint16_t*) &vb16i + 6);
    *((uint16_t*) &vb32f + 14 + BF16_OFFSET) = *((uint16_t*) &vb16i + 7);

    __m256 vc32f = _mm256_min_ps(va32f, vb32f);
    *((uint16_t*) &vc16i + 0) = *((uint16_t*) &vc32f + 0 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 1) = *((uint16_t*) &vc32f + 2 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 2) = *((uint16_t*) &vc32f + 4 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 3) = *((uint16_t*) &vc32f + 6 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 4) = *((uint16_t*) &vc32f + 8 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 5) = *((uint16_t*) &vc32f + 10 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 6) = *((uint16_t*) &vc32f + 12 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 7) = *((uint16_t*) &vc32f + 14 + BF16_OFFSET);

    _mm_storeu_si128((__m128i*)(&c[i]), vc16i);
  }
#endif

  for (; i < n; i ++) {
    a_fp32_ptr[1] = a[i].x;
    b_fp32_ptr[1] = b[i].x;
    
    c_fp32 = a_fp32 < b_fp32 ? a_fp32 : b_fp32;
    c[i].x = c_fp32_ptr[1];
  }
}

template<> void max<bfloat16>(void* c_, const void* a_, const void* b_, size_t n) {
    bfloat16* c = static_cast<bfloat16*>(c_);
  const bfloat16* a = static_cast<const bfloat16*>(a_);
  const bfloat16* b = static_cast<const bfloat16*>(b_);

  float a_fp32 = 0, b_fp32 = 0, c_fp32;
  uint16_t *a_fp32_ptr = reinterpret_cast<uint16_t*>(&a_fp32);
  uint16_t *b_fp32_ptr = reinterpret_cast<uint16_t*>(&b_fp32);
  uint16_t *c_fp32_ptr = reinterpret_cast<uint16_t*>(&c_fp32);

  size_t i = 0;

#ifdef __AVX__
  __m256 va32f = _mm256_setzero_ps();
  __m256 vb32f = _mm256_setzero_ps();
  __m128i vc16i;

  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m128i va16i = _mm_loadu_si128((__m128i*)(&a[i]));
    __m128i vb16i = _mm_loadu_si128((__m128i*)(&b[i]));

    *((uint16_t*) &va32f + 0 + BF16_OFFSET) = *((uint16_t*) &va16i + 0);
    *((uint16_t*) &va32f + 2 + BF16_OFFSET) = *((uint16_t*) &va16i + 1);
    *((uint16_t*) &va32f + 4 + BF16_OFFSET) = *((uint16_t*) &va16i + 2);
    *((uint16_t*) &va32f + 6 + BF16_OFFSET) = *((uint16_t*) &va16i + 3);
    *((uint16_t*) &va32f + 8 + BF16_OFFSET) = *((uint16_t*) &va16i + 4);
    *((uint16_t*) &va32f + 10 + BF16_OFFSET) = *((uint16_t*) &va16i + 5);
    *((uint16_t*) &va32f + 12 + BF16_OFFSET) = *((uint16_t*) &va16i + 6);
    *((uint16_t*) &va32f + 14 + BF16_OFFSET) = *((uint16_t*) &va16i + 7);

    *((uint16_t*) &vb32f + 0 + BF16_OFFSET) = *((uint16_t*) &vb16i + 0);
    *((uint16_t*) &vb32f + 2 + BF16_OFFSET) = *((uint16_t*) &vb16i + 1);
    *((uint16_t*) &vb32f + 4 + BF16_OFFSET) = *((uint16_t*) &vb16i + 2);
    *((uint16_t*) &vb32f + 6 + BF16_OFFSET) = *((uint16_t*) &vb16i + 3);
    *((uint16_t*) &vb32f + 8 + BF16_OFFSET) = *((uint16_t*) &vb16i + 4);
    *((uint16_t*) &vb32f + 10 + BF16_OFFSET) = *((uint16_t*) &vb16i + 5);
    *((uint16_t*) &vb32f + 12 + BF16_OFFSET) = *((uint16_t*) &vb16i + 6);
    *((uint16_t*) &vb32f + 14 + BF16_OFFSET) = *((uint16_t*) &vb16i + 7);

    __m256 vc32f = _mm256_max_ps(va32f, vb32f);
    *((uint16_t*) &vc16i + 0) = *((uint16_t*) &vc32f + 0 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 1) = *((uint16_t*) &vc32f + 2 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 2) = *((uint16_t*) &vc32f + 4 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 3) = *((uint16_t*) &vc32f + 6 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 4) = *((uint16_t*) &vc32f + 8 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 5) = *((uint16_t*) &vc32f + 10 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 6) = *((uint16_t*) &vc32f + 12 + BF16_OFFSET);
    *((uint16_t*) &vc16i + 7) = *((uint16_t*) &vc32f + 14 + BF16_OFFSET);

    _mm_storeu_si128((__m128i*)(&c[i]), vc16i);
  }
#endif

  for (; i < n; i ++) {
    a_fp32_ptr[1] = a[i].x;
    b_fp32_ptr[1] = b[i].x;
    
    c_fp32 = a_fp32 > b_fp32 ? a_fp32 : b_fp32;
    c[i].x = c_fp32_ptr[1];
  }
}

} // namespace gloo