#pragma once
#include <cstdint>
// fp22 (e8m13) -> fp8 (e5m2)

inline uint16_t fp22_to_fp8(uint32_t fp22) {
    bool s = (fp22 >> 21) & 1;
    int e = (fp22 >> 13) & 0xFF;
    int m = fp22 & 0x1FFF;

    // 1. 处理 NaN 和 Inf (对应 fp22 指数位全 1)
    if (e == 0xFF) { 
        if (m) return (s << 7) | (0x1F << 2) | 2; // NaN (置位尾数最高位)
        return (s << 7) | (0x1F << 2);            // Inf
    }
    // 2. 处理 +0 / -0
    if (e == 0 && m == 0) return (s << 7);

    // 3. 计算在 FP8 (E5M2) 下的指数 (去除偏移量差值 112)
    int e5 = e - 112;

    // 4. 处理下溢为 FP8 次正规数 (Subnormal) 或 0 的情况
    // (注：由于两者指数偏差不同，fp22_to_fp8 无法再直接使用你原代码里的 clz 来处理，而是需要进行隐含位对齐)
    if (e5 <= 0) {
        // 如果原本就是 fp22 的次正规数 (e==0)，真实指数极小(-126)，或者需要右移的位数大于14，必定向下完全溢出为0
        if (e == 0 || (1 - e5) > 14) return (s << 7);

        int shift = 1 - e5; // 需要将正规数的隐含1右移的位数
        uint32_t ext_m = (1 << 13) | m; // 补上正规数隐含的 '1'
        uint32_t shifted_m = ext_m >> shift;
        uint32_t lost_mask = (1 << shift) - 1; // 记录被右移完全挤掉的位，用于Sticky判定

        // 提取 FP8 次正规数尾数 (2位)，并计算 Guard(g), Round(r), Sticky(st)
        int fp8m = (shifted_m >> 11) & 3;
        bool g = (shifted_m >> 10) & 1;
        bool r = (shifted_m >> 9) & 1;
        bool st = ((shifted_m & 0x1FF) != 0) || ((ext_m & lost_mask) != 0);

        // Round-to-nearest-even (偶数舍入)
        if (g && (r || st || (fp8m & 1))) {
            fp8m++;
            if (fp8m >= 4) return (s << 7) | (1 << 2); // 进位后恰好成为 FP8 的最小正规数
        }
        return (s << 7) | fp8m; // e5 此时为 0
    }

    // 5. 处理上溢为 Inf 的情况
    if (e5 >= 31) {
        return (s << 7) | (0x1F << 2);
    }

    // 6. 正常 FP8 正规数的截断和舍入 (13位 -> 2位)
    int fp8m = (m >> 11) & 3;
    bool g = (m >> 10) & 1;
    bool r = (m >> 9) & 1;
    bool st = (m & 0x1FF) != 0;

    // Round-to-nearest-even (与你的原代码逻辑完全一致)
    if (g && (r || st || (fp8m & 1))) {
        fp8m++;
        if (fp8m >= 4) {
            fp8m = 0; 
            e5++;
            if (e5 >= 31) return (s << 7) | (0x1F << 2); // 进位可能导致上溢到 Inf
        }
    }

    return (s << 7) | (e5 << 2) | fp8m;
}