#pragma once
#include <cstdint>

inline uint16_t fp22_to_fp16(uint32_t fp22) {
  uint32_t s = (fp22 >> 21) & 0x1;
  int32_t e = (fp22 >> 13) & 0xFF;
  uint32_t m = fp22 & 0x1FFF;

  if (e == 0xFF) {
    return (s << 15) | (0x1F << 10) | (m ? 0x200 : 0);
  }
  if (e == 0 && m == 0) {
    return (s << 15);
  }

  int32_t e16 = e - 112;
  if (e16 <= 0) {
    if (e == 0 || (1 - e16) > 24)
      return (s << 15);

    uint32_t ext = (1u << 13) | m;
    uint32_t shift = static_cast<uint32_t>(1 - e16);
    uint32_t shifted = ext >> shift;
    uint32_t frac = (shifted >> 3) & 0x3FF;
    bool g = (shifted >> 2) & 1;
    bool r = (shifted >> 1) & 1;
    bool st = ((shifted & 0x1) != 0) || ((ext & ((1u << shift) - 1)) != 0);
    if (g && (r || st || (frac & 1))) {
      ++frac;
      if (frac >= 0x400) {
        return (s << 15) | (1 << 10);
      }
    }
    return (s << 15) | frac;
  }

  if (e16 >= 31) {
    return (s << 15) | (0x1F << 10);
  }

  uint32_t frac = (m >> 3) & 0x3FF;
  bool g = (m >> 2) & 1;
  bool r = (m >> 1) & 1;
  bool st = (m & 0x1) != 0;
  if (g && (r || st || (frac & 1))) {
    ++frac;
    if (frac >= 0x400) {
      frac = 0;
      ++e16;
      if (e16 >= 31)
        return (s << 15) | (0x1F << 10);
    }
  }

  return (s << 15) | ((e16 & 0x1F) << 10) | frac;
}
