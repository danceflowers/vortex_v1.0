#pragma once
#include <cstdint>
#include <cstring>
#include "fp_types.h"

enum SparseMode : uint8_t {
    SPARSE_DENSE = 0,
    SPARSE_2_TO_4 = 1,
    SPARSE_1_TO_4 = 2,
};

inline bool fp9_is_zero(uint16_t fp9) {
    return (((fp9 >> 3) & 0x1F) == 0) && ((fp9 & 0x7) == 0);
}

inline int sparse_group_nnz(SparseMode mode) {
    switch (mode) {
        case SPARSE_2_TO_4: return 2;
        case SPARSE_1_TO_4: return 1;
        default: return 4;
    }
}

inline uint8_t encode_sparse_meta_2_to_4(uint8_t idx0, uint8_t idx1) {
    return (uint8_t)((idx0 & 0x3) | ((idx1 & 0x3) << 2));
}

inline uint8_t encode_sparse_meta_1_to_4(uint8_t idx0) {
    return (uint8_t)(idx0 & 0x3);
}

inline void decode_sparse_meta_group(uint8_t meta, SparseMode mode, uint8_t& idx0, uint8_t& idx1) {
    idx0 = meta & 0x3;
    idx1 = (mode == SPARSE_2_TO_4) ? ((meta >> 2) & 0x3) : 0;
}

inline void prepare_sparse_a_payload_and_meta(const uint16_t dense_a_row[8],
                                              SparseMode mode,
                                              PrecisionType prec,
                                              uint32_t packed_payload[8],
                                              uint8_t packed_meta[2]) {
    std::memset(packed_payload, 0, sizeof(uint32_t) * 8);
    packed_meta[0] = 0;
    packed_meta[1] = 0;

    if (mode == SPARSE_DENSE) {
        for (int k = 0; k < 8; ++k) {
            packed_payload[k] = dense_a_row[k];
        }
        return;
    }

    const int nnz_limit = sparse_group_nnz(mode);
    int payload_cursor = 0;

    for (int g = 0; g < 2; ++g) {
        uint8_t chosen_idx[2] = {0, 0};
        int chosen_count = 0;

        for (int i = 0; i < 4; ++i) {
            const int k = g * 4 + i;
            uint16_t fp9 = convert_to_fp9(dense_a_row[k], prec);
            if (!fp9_is_zero(fp9)) {
                if (chosen_count < nnz_limit) {
                    chosen_idx[chosen_count] = (uint8_t)i;
                    packed_payload[payload_cursor++] = dense_a_row[k];
                }
                ++chosen_count;
            }
        }

        if (mode == SPARSE_2_TO_4) {
            packed_meta[g] = encode_sparse_meta_2_to_4(chosen_idx[0], chosen_idx[1]);
            if (chosen_count < 2) {
                for (int pad = chosen_count; pad < 2; ++pad) {
                    packed_payload[payload_cursor++] = 0;
                }
            }
        } else {
            packed_meta[g] = encode_sparse_meta_1_to_4(chosen_idx[0]);
            if (chosen_count < 1) {
                packed_payload[payload_cursor++] = 0;
            }
        }
    }
}

inline void input_parser_bypass_expand_sparse_operands(const uint32_t packed_a_payload[8],
                                                       const uint32_t dense_b[8],
                                                       const uint8_t packed_meta[2],
                                                       SparseMode mode,
                                                       PrecisionType prec,
                                                       uint32_t routed_a[8],
                                                       uint32_t routed_b[8]) {
    for (int i = 0; i < 8; ++i) {
        routed_a[i] = 0;
        routed_b[i] = 0;
    }

    if (mode == SPARSE_DENSE) {
        for (int i = 0; i < 8; ++i) {
            routed_a[i] = packed_a_payload[i];
            routed_b[i] = dense_b[i];
        }
        return;
    }

    const int lanes_per_group = sparse_group_nnz(mode);
    int payload_cursor = 0;
    int lane_cursor = 0;

    for (int g = 0; g < 2; ++g) {
        uint8_t idx0 = 0, idx1 = 0;
        decode_sparse_meta_group(packed_meta[g], mode, idx0, idx1);
        const uint8_t idx[2] = {idx0, idx1};
        for (int lane = 0; lane < lanes_per_group; ++lane) {
            routed_a[lane_cursor] = packed_a_payload[payload_cursor++];
            routed_b[lane_cursor] = dense_b[g * 4 + idx[lane]];
            ++lane_cursor;
        }
    }
}

inline void apply_structured_sparsity_to_matrix_a(uint16_t a_raw[8][8],
                                                  SparseMode mode,
                                                  PrecisionType prec,
                                                  uint32_t seed_hint = 0) {
    if (mode == SPARSE_DENSE) return;

    for (int i = 0; i < 8; ++i) {
        for (int g = 0; g < 2; ++g) {
            int keep_budget = sparse_group_nnz(mode);
            int selected[2] = {0, 0};
            int sel_count = 0;

            // Simple deterministic policy: keep the first keep_budget positions,
            // optionally rotate by seed_hint to vary the pattern across runs.
            int start = (int)((seed_hint + i + g) & 0x3);
            for (int t = 0; t < 4 && sel_count < keep_budget; ++t) {
                selected[sel_count++] = (start + t) & 0x3;
            }

            bool keep_mask[4] = {false, false, false, false};
            for (int t = 0; t < sel_count; ++t) {
                keep_mask[selected[t]] = true;
            }

            for (int t = 0; t < 4; ++t) {
                if (!keep_mask[t]) {
                    a_raw[i][g * 4 + t] = 0;
                }
            }

            // Ensure the retained entries are non-zero whenever possible.
            for (int t = 0; t < sel_count; ++t) {
                int k = g * 4 + selected[t];
                uint16_t fp9 = convert_to_fp9(a_raw[i][k], prec);
                if (fp9_is_zero(fp9)) {
                    switch (prec) {
                        case PREC_FP4_E2M1: a_raw[i][k] = 0x1; break;
                        case PREC_FP8_E4M3:
                        case PREC_FP8_E5M2: a_raw[i][k] = 0x1; break;
                        case PREC_FP16:     a_raw[i][k] = 0x0001; break;
                        default:            a_raw[i][k] = 0x1; break;
                    }
                }
            }
        }
    }
}

