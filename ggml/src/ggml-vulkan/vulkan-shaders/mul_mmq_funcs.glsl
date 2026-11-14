#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include "types.glsl"

int32_t mmq_dot(const int32_t a, const int32_t b) {
    // With MMQ_SUDOT the A quants must be unsigned
    if (MMQ_SUDOT) {
        return dotPacked4x8EXT(uint32_t(a), b);
    }
    return dotPacked4x8EXT(a, b);
}

// With MMQ_SUDOT, flip the sign bit of each byte. This adds a bias of 128 to signed quants.
int32_t mmq_a_store(const int32_t v) {
    return MMQ_SUDOT ? v ^ int32_t(0x80808080) : v;
}

// Each iqs value maps to a 32-bit integer

#if defined(DATA_A_Q2_0)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint block_idx = ib / 2;
        const uint byte_idx = (ib & 1u) * 8u + iqs;
        const uint bits = uint(data_a[block_idx].qs[byte_idx]);
        buf_a[buf_ib].qs[iqs] = pack32(i8vec4(
            int8_t(bits & 3u),
            int8_t((bits >> 2u) & 3u),
            int8_t((bits >> 4u) & 3u),
            int8_t(bits >> 6u)));

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(data_a[block_idx].d);
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; ++iqs) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; ++iqs) {
        q_sum += mmq_dot(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - float(cache_b.ds.y)));
}
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
// 2-byte loads for Q4_0 blocks (18 bytes)
// 4-byte loads for Q4_1 blocks (20 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
    #ifdef DATA_A_Q4_0
        buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                               data_a_packed16[ib].qs[iqs * 2 + 1]));

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
        }
    #else // DATA_A_Q4_1
        buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib].dm);
        }
    #endif
    } else {
#ifdef DATA_A_Q4_0
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(0.0f);
        }
#else // DATA_A_Q4_1
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(0.0f);
        }
#endif
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const i32vec2 qs_a = i32vec2( vui       & 0x0F0F0F0F,
                                     (vui >> 4) & 0x0F0F0F0F);

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += mmq_dot(qs_a.x, qs_b0);
        q_sum += mmq_dot(qs_a.y, qs_b1);
    }

#ifdef DATA_A_Q4_0
    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - 8.0 * float(cache_b.ds.y)));
#else // DATA_A_Q4_1
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm.x) * float(cache_b.ds.x) + float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
#endif
}
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
// 2-byte loads for Q5_0 blocks (22 bytes)
// 4-byte loads for Q5_1 blocks (24 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
    #ifdef DATA_A_Q5_0
        buf_a[buf_ib].qs[iqs] = pack32(u16vec2(data_a_packed16[ib].qs[iqs * 2],
                                               data_a_packed16[ib].qs[iqs * 2 + 1]));

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
            buf_a[buf_ib].qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
        }
    #else // DATA_A_Q5_1
        buf_a[buf_ib].qs[iqs] = data_a_packed32[ib].qs[iqs];

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib].dm);
            buf_a[buf_ib].qh = data_a_packed32[ib].qh;
        }
    #endif
    } else {
#ifdef DATA_A_Q5_0
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(0.0f);
            buf_a[buf_ib].qh = 0;
        }
#else // DATA_A_Q5_1
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(0.0f);
            buf_a[buf_ib].qh = 0;
        }
#endif
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].qh = buf_a[buf_ib].qh;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const uint32_t vui = cache_a[ib_a].qs[iqs];
        const int32_t qh = int32_t(cache_a[ib_a].qh >> (4 * iqs));
        const int32_t qs_a0 = int32_t(vui & 0x0F0F0F0F)
                         | ((qh & 0xF) * 0x02040810) & 0x10101010; // (0,1,2,3) -> (4,12,20,28)
        const int32_t qs_a1 = int32_t((vui >> 4) & 0x0F0F0F0F)
                         | (((qh >> 16) & 0xF) * 0x02040810) & 0x10101010; // (16,17,18,19) -> (4,12,20,28)

        const int32_t qs_b0 = cache_b.qs[iqs];
        const int32_t qs_b1 = cache_b.qs[iqs + 4];

        q_sum += mmq_dot(qs_a0, qs_b0);
        q_sum += mmq_dot(qs_a1, qs_b1);
    }

#ifdef DATA_A_Q5_0
    return ACC_TYPE(float(cache_a[ib_a].dm) * (float(q_sum) * float(cache_b.ds.x) - 16.0 * float(cache_b.ds.y)));
#else // DATA_A_Q5_1
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm.x) * float(cache_b.ds.x) + float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
#endif
}
#endif

#if defined(DATA_A_Q8_0)
// 2-byte loads for Q8_0 blocks (34 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        buf_a[buf_ib].qs[iqs] = mmq_a_store(pack32(i16vec2(data_a_packed16[ib].qs[iqs * 2],
                                                           data_a_packed16[ib].qs[iqs * 2 + 1])));

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
        const int32_t qs_b = cache_b.qs[iqs];

        q_sum += mmq_dot(qs_a, qs_b);
    }
    if (MMQ_SUDOT) {
        q_sum -= 128 * (cache_b_qsum[0] + cache_b_qsum[1]);
    }

    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm) * float(cache_b.ds.x));
}
#endif

#if defined(DATA_A_MXFP4)
// 1-byte loads for mxfp4 blocks (17 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint32_t qs = pack32(u8vec4(data_a[ib].qs[iqs * 4    ],
                                          data_a[ib].qs[iqs * 4 + 1],
                                          data_a[ib].qs[iqs * 4 + 2],
                                          data_a[ib].qs[iqs * 4 + 3]));

        const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
        const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);

        buf_a[buf_ib].qs[iqs    ] = mmq_a_store(pack32(i8vec4(kvalues_mxfp4[i_a0.x], kvalues_mxfp4[i_a0.y], kvalues_mxfp4[i_a0.z], kvalues_mxfp4[i_a0.w])));
        buf_a[buf_ib].qs[iqs + 4] = mmq_a_store(pack32(i8vec4(kvalues_mxfp4[i_a1.x], kvalues_mxfp4[i_a1.y], kvalues_mxfp4[i_a1.z], kvalues_mxfp4[i_a1.w])));

        if (iqs == 0) {
            buf_a[buf_ib].d = FLOAT_TYPE(e8m0_to_fp32(data_a[ib].e) * 0.5);
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;
        buf_a[buf_ib].qs[iqs + 4] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += mmq_dot(qs_a, cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 128 * (cache_b_qsum[0] + cache_b_qsum[1]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ4_NL)
// 2-byte loads for IQ4_NL blocks (18 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint32_t vui = uint32_t(data_a_packed16[ib].qs[2 * iqs]) | (uint32_t(data_a_packed16[ib].qs[2 * iqs + 1]) << 16);
        const i32vec2 qs = iq4nl_to_i8x8(vui);

        buf_a[buf_ib].qs[iqs    ] = qs.x;
        buf_a[buf_ib].qs[iqs + 4] = qs.y;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;
        buf_a[buf_ib].qs[iqs + 4] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].dm) * float(cache_b.ds.x) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ4_XS)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint ib32 = ib % 8;
        const uint32_t vui = data_a_packed32[ib_k].qs[4 * ib32 + iqs];
        const i32vec2 qs = iq4nl_to_i8x8(vui);

        buf_a[buf_ib].qs[iqs    ] = mmq_a_store(qs.x);
        buf_a[buf_ib].qs[iqs + 4] = mmq_a_store(qs.y);

        if (iqs == 0) {
            const uint sl = (data_a_packed32[ib_k].scales_l >> (4 * ib32)) & 0xF;
            const uint sh = (data_a_packed32[ib_k].scales_h >> (2 * ib32)) & 3;
            buf_a[buf_ib].d = FLOAT_TYPE(float(data_a[ib_k].d) * float(int(sl | (sh << 4)) - 32));
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;
        buf_a[buf_ib].qs[iqs + 4] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += mmq_dot(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 128 * (cache_b_qsum[0] + cache_b_qsum[1]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

// For k-quants, ib and iqs still assume 32-wide blocks, but k-quants are 256-wide
// iqs still refers to a 32-bit integer, meaning 0..7 for 32-wide quants
#if defined(DATA_A_Q2_K)
// 4-byte loads for Q2_K blocks (84 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

        const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
        const uint qs_shift = ((iqs_k % 32) / 8) * 2;

        // Repack 4x4 quants into one int
        const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x03030303;
        const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x03030303;
        const uint32_t vals2 = (data_a_packed32[ib_k].qs[qs_idx + 2] >> qs_shift) & 0x03030303;
        const uint32_t vals3 = (data_a_packed32[ib_k].qs[qs_idx + 3] >> qs_shift) & 0x03030303;

        buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 2) | (vals2 << 4) | (vals3 << 6);

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(data_a_packed32[ib_k].dm);
            buf_a[buf_ib].scales = unpack8(uint32_t(data_a_packed16[ib_k].scales[iqs_k / 8])).xy; // vec4 used due to #12147
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(0.0f);
            buf_a[buf_ib].scales = u8vec2(0);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    cache_a[reg_ib].scales = buf_a[buf_ib].scales;

    [[unroll]] for (uint iqs = 0; iqs < 2; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t sum_d = 0;
    int32_t sum_m = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        const uint8_t scale = cache_a[ib_a].scales[iqs / 4];
        const int32_t scale_m = int32_t(scale >> 4) * 0x01010101; // Duplicate 8-bit value across 32-bits.
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 4] >> ((iqs % 4) * 2)) & 0x03030303);

        sum_d += mmq_dot(qs_a, cache_b.qs[iqs]) * (scale & 0xF);
        sum_m += mmq_dot(scale_m, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].dm.x) * float(sum_d) - float(cache_a[ib_a].dm.y) * float(sum_m)));
}
#endif

#if defined(DATA_A_Q3_K)
// 2-byte loads for Q3_K blocks (110 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint hm_idx = iqs * QUANT_R_MMQ;
        const uint iqs_k = (ib % 8) * 8 + hm_idx;

        const uint qs_idx = (iqs_k / 32) * 8 + (iqs_k % 8);
        const uint qs_shift = ((iqs_k % 32) / 8) * 2;
        const uint hm_shift = iqs_k / 8;

        // Repack 2x4 quants into one int
        // Add the 3rd bit instead of subtracting it to allow packing the quants
        // vec4 for unpack8 used due to #12147
        const i8vec2 vals00 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2        ] >> qs_shift) & uint16_t(0x0303)))).xy |
                              unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2    ] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
        const i8vec2 vals01 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 1    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                              unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 1] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
        const i8vec2 vals10 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 2    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                              unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 2] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
        const i8vec2 vals11 = unpack8(int32_t(int16_t((data_a_packed16[ib_k].qs[qs_idx * 2 + 3    ] >> qs_shift) & uint16_t(0x0303)))).xy |
                              unpack8(int32_t(int16_t(((data_a_packed16[ib_k].hmask[hm_idx * 2 + 3] >> hm_shift) & uint16_t(0x0101))) << 2)).xy;
        buf_a[buf_ib].qs[iqs] = pack32(u8vec4(vals00.x, vals00.y, vals01.x, vals01.y)) |
                               (pack32(u8vec4(vals10.x, vals10.y, vals11.x, vals11.y)) << 4);

        if (iqs == 0) {
            const uint is = iqs_k / 4;
            const i8vec2 scales = i8vec2(unpack8(uint32_t(((data_a_packed16[ib_k].scales[(is % 8      ) / 2] >> (4 * (is / 8))) & 0x0F0F) |
                                                         (((data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2] >> (2 * (is / 4))) & 0x0303) << 4))).xy); // vec4 used due to #12147

            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales - 32));
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        // Subtract 4 from the quants to correct the 3rd bit offset. MMQ_SUDOT keeps them unsigned and subtracts 4 times the B sum.
        const int32_t qs_u = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
        const int32_t qs_a = MMQ_SUDOT ? qs_u : pack32(unpack8(qs_u) - int8_t(4));

        q_sum += mmq_dot(qs_a, cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 4 * cache_b_qsum[0];
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_u = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
        const int32_t qs_a = MMQ_SUDOT ? qs_u : pack32(unpack8(qs_u) - int8_t(4));

        q_sum += mmq_dot(qs_a, cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 4 * cache_b_qsum[1];
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

#if defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
// 4-byte loads for Q4_K blocks (144 bytes) and Q5_K blocks (176 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint iqs_k = (ib % 8) * 8 + iqs * QUANT_R_MMQ;

        const uint qs_idx = (iqs_k / 16) * 8 + (iqs_k % 8);
        const uint qs_shift = ((iqs_k % 16) / 8) * 4;

        // Repack 2x4 quants into one int
    #if defined(DATA_A_Q4_K)
        const uint32_t vals0 = (data_a_packed32[ib_k].qs[qs_idx    ] >> qs_shift) & 0x0F0F0F0F;
        const uint32_t vals1 = (data_a_packed32[ib_k].qs[qs_idx + 1] >> qs_shift) & 0x0F0F0F0F;

        buf_a[buf_ib].qs[iqs] = vals0 | (vals1 << 4);
    #else // defined(DATA_A_Q5_K)
        const uint qh_idx = iqs * QUANT_R_MMQ;
        const uint qh_shift = iqs_k / 8;

        buf_a[buf_ib].qs[iqs] = int32_t(((data_a_packed32[ib_k].qs[qs_idx] >> qs_shift) & 0x0F0F0F0F) |
                                       (((data_a_packed32[ib_k].qh[qh_idx] >> qh_shift) & 0x01010101) << 4));
    #endif

        if (iqs == 0) {
            // Scale index
            const uint is = iqs_k / 8;
            u8vec2 scale_dm;
            if (is < 4) {
                scale_dm = u8vec2(data_a[ib_k].scales[is] & 0x3F, data_a[ib_k].scales[is + 4] & 0x3F);
            } else {
                scale_dm = u8vec2((data_a[ib_k].scales[is+4] & 0xF) | ((data_a[ib_k].scales[is-4] & 0xC0) >> 2),
                                  (data_a[ib_k].scales[is+4] >>  4) | ((data_a[ib_k].scales[is  ] & 0xC0) >> 2));
            }

            buf_a[buf_ib].dm = FLOAT_TYPEV2(vec2(data_a_packed32[ib_k].dm) * vec2(scale_dm));
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].dm = FLOAT_TYPEV2(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;

    [[unroll]] for (uint iqs = 0; iqs < 8 / QUANT_R_MMQ; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
#if defined(DATA_A_Q4_K)
        const int32_t qs_a = int32_t((cache_a[ib_a].qs[iqs / 2] >> ((iqs % 2) * 4)) & 0x0F0F0F0F);
#else // defined(DATA_A_Q5_K)
        const int32_t qs_a = cache_a[ib_a].qs[iqs];
#endif

        q_sum += mmq_dot(qs_a, cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * float(cache_a[ib_a].dm.x) * float(q_sum) - float(cache_a[ib_a].dm.y) * float(cache_b.ds.y));
}
#endif

#if defined(DATA_A_Q6_K)
// 2-byte loads for Q6_K blocks (210 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint iqs_k = (ib % 8) * 8 + iqs;

        const uint ql_idx = (iqs_k / 32) * 16 + iqs_k % 16;
        const uint ql_shift = ((iqs_k % 32) / 16) * 4;

        const uint qh_idx = (iqs_k / 32) * 8 + iqs;
        const uint qh_shift = ((iqs_k % 32) / 8) * 2;

        const i8vec2 vals00 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2    ] >> ql_shift) & uint16_t(0x0F0F))).xy |
                              unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2    ] >> qh_shift) & uint16_t(0x0303)) << 4)).xy);
        const i8vec2 vals01 = (unpack8(int32_t((data_a_packed16[ib_k].ql[ql_idx * 2 + 1] >> ql_shift) & uint16_t(0x0F0F))).xy |
                              unpack8(int32_t(((data_a_packed16[ib_k].qh[qh_idx * 2 + 1] >> qh_shift) & uint16_t(0x0303)) << 4)).xy);
        // MMQ_SUDOT keeps the quants unsigned and subtracts 32 times the B sum
        buf_a[buf_ib].qs[iqs] = MMQ_SUDOT ? pack32(i8vec4(vals00, vals01)) : pack32(i8vec4(vals00 - int8_t(32), vals01 - int8_t(32)));

        if (iqs == 0) {
            const uint is = iqs_k / 4;
            const i8vec2 scales = unpack8(int32_t(data_a_packed16[ib_k].scales[is / 2])).xy;

            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(float(data_a_packed16[ib_k].d) * vec2(scales));
        }
    } else {
        buf_a[buf_ib].qs[iqs] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    float result = 0.0;
    int32_t q_sum = 0;

    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += mmq_dot(qs_a, cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 32 * cache_b_qsum[0];
    }
    result += float(cache_a[ib_a].d_scales[0]) * float(q_sum);
    q_sum = 0;

    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];

        q_sum += mmq_dot(qs_a, cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 32 * cache_b_qsum[1];
    }
    result += float(cache_a[ib_a].d_scales[1]) * float(q_sum);

    return ACC_TYPE(float(cache_b.ds.x) * result);
}
#endif

#if defined(DATA_A_IQ3_S)
// 2-byte loads for IQ3_S blocks (110 bytes)
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint ib32 = ib % 8;

        // grid indices for qs[2 * iqs] and qs[2 * iqs + 1]
        const uint qs    = uint(data_a_packed16[ib_k].qs[ib32 * 4 + iqs]);
        // their two high index bits
        const uint qh    = uint(data_a_packed16[ib_k].qh[ib32 / 2]) >> ((ib32 & 1) * 8 + 2 * iqs);
        // one sign bit per value, 8 values
        const uint signs = uint(data_a_packed16[ib_k].signs[ib32 * 2 + iqs / 2]) >> ((iqs & 1) * 8);

        // grid holds 4 values of 1..15, one per byte
        const ivec4 vals0 = ivec4(unpack8(iq3s_grid[( qs       & 0xFF) | ((qh & 1) << 8)]));
        const ivec4 vals1 = ivec4(unpack8(iq3s_grid[((qs >> 8) & 0xFF) | ((qh & 2) << 7)]));

        // negate with (v ^ -s) - -s to avoid branches
        const ivec4 m0 = -(ivec4(signs,      signs >> 1, signs >> 2, signs >> 3) & 1);
        const ivec4 m1 = -(ivec4(signs >> 4, signs >> 5, signs >> 6, signs >> 7) & 1);

        buf_a[buf_ib].qs[2 * iqs    ] = mmq_a_store(pack32(i8vec4((vals0 ^ m0) - m0)));
        buf_a[buf_ib].qs[2 * iqs + 1] = mmq_a_store(pack32(i8vec4((vals1 ^ m1) - m1)));

        if (iqs == 0) {
            const uint scale = (uint(data_a_packed16[ib_k].scales[ib32 / 4]) >> ((ib32 & 3) * 4)) & 0xF;

            buf_a[buf_ib].d = FLOAT_TYPE(float(data_a_packed16[ib_k].d) * float(1 + 2 * scale));
        }
    } else {
        buf_a[buf_ib].qs[2 * iqs] = 0;
        buf_a[buf_ib].qs[2 * iqs + 1] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += mmq_dot(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }
    if (MMQ_SUDOT) {
        q_sum -= 128 * (cache_b_qsum[0] + cache_b_qsum[1]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ2_S)
// 2-byte loads for IQ2_S blocks (82 bytes). Each iqs handles one group of 8 values.
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint ib32 = ib % 8;

        // low 8 bits of the grid index (qs[0..31]) and the sign byte (qs[32..63])
        const uint byte_shift = (iqs & 1) * 8;
        const uint qs    = (uint(data_a_packed16[ib_k].qs[     ib32 * 2 + iqs / 2]) >> byte_shift) & 0xFF;
        const uint signs = (uint(data_a_packed16[ib_k].qs[16 + ib32 * 2 + iqs / 2]) >> byte_shift) & 0xFF;
        // two high index bits per group
        const uint qh    = (uint(data_a_packed16[ib_k].qh[ib32 / 2]) >> ((ib32 & 1) * 8 + 2 * iqs)) & 3;

        // grid holds 8 unsigned values (8, 25 or 43), one per byte
        const uvec2 grid = iq2s_grid[qs | (qh << 8)];
        const ivec4 vals0 = ivec4(unpack8(grid.x));
        const ivec4 vals1 = ivec4(unpack8(grid.y));

        // negate with (v ^ -s) - -s to avoid branches
        const ivec4 m0 = -(ivec4(signs,      signs >> 1, signs >> 2, signs >> 3) & 1);
        const ivec4 m1 = -(ivec4(signs >> 4, signs >> 5, signs >> 6, signs >> 7) & 1);

        buf_a[buf_ib].qs[2 * iqs    ] = pack32(i8vec4((vals0 ^ m0) - m0));
        buf_a[buf_ib].qs[2 * iqs + 1] = pack32(i8vec4((vals1 ^ m1) - m1));

        if (iqs == 0) {
            // 4-bit scales, low nibble for the first 16 values, high nibble for the second 16
            const uint sc = (uint(data_a_packed16[ib_k].scales[ib32 / 2]) >> ((ib32 & 1) * 8)) & 0xFF;
            const float d = float(data_a_packed16[ib_k].d) * 0.125f;

            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(d * float(1 + 2 * (sc & 0xF)), d * float(1 + 2 * (sc >> 4)));
        }
    } else {
        buf_a[buf_ib].qs[2 * iqs] = 0;
        buf_a[buf_ib].qs[2 * iqs + 1] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum0 = 0;
    int32_t q_sum1 = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        q_sum0 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }
    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        q_sum1 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].d_scales[0]) * float(q_sum0) +
                                           float(cache_a[ib_a].d_scales[1]) * float(q_sum1)));
}
#endif

#if defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ3_XXS)
// Expand 7 sign bits + parity bit to 8 int8 lanes of +1/-1 masks and apply them to 8 grid values
i32vec2 iq_apply_signs8(const uvec2 grid, const uint sign7) {
    const uint signs = sign7 | (bitCount(sign7) << 7);
    const ivec4 vals0 = ivec4(unpack8(grid.x));
    const ivec4 vals1 = ivec4(unpack8(grid.y));
    const ivec4 m0 = -(ivec4(signs,      signs >> 1, signs >> 2, signs >> 3) & 1);
    const ivec4 m1 = -(ivec4(signs >> 4, signs >> 5, signs >> 6, signs >> 7) & 1);
    return i32vec2(pack32(i8vec4((vals0 ^ m0) - m0)), pack32(i8vec4((vals1 ^ m1) - m1)));
}
#endif

#if defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ3_XXS)
// Each iqs handles one group of 8 values, one scale per 32 values
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint ib32 = ib % 8;

#if defined(DATA_A_IQ2_XXS)
        // 8 bytes per 32 values: 4 grid indices, then 4x7 sign bits + 4-bit scale
        const uint aux = pack32(u16vec2(data_a_packed16[ib_k].qs[4 * ib32 + 2], data_a_packed16[ib_k].qs[4 * ib32 + 3]));
        const uint idx = (uint(data_a_packed16[ib_k].qs[4 * ib32 + iqs / 2]) >> ((iqs & 1) * 8)) & 0xFF;
        const uvec2 grid = iq2xxs_grid[idx];
        const float db = 0.125f * float(1 + 2 * (aux >> 28));
#else
        // 64 bytes of grid indices (2 per group), then one uint32 of 4x7 sign bits + 4-bit scale per 32 values
        const uint aux = pack32(u16vec2(data_a_packed16[ib_k].qs[32 + 2 * ib32], data_a_packed16[ib_k].qs[32 + 2 * ib32 + 1]));
        const uint idx = uint(data_a_packed16[ib_k].qs[4 * ib32 + iqs]);
        const uvec2 grid = uvec2(iq3xxs_grid[idx & 0xFF], iq3xxs_grid[idx >> 8]);
        const float db = 0.25f * float(1 + 2 * (aux >> 28));
#endif

        const i32vec2 q = iq_apply_signs8(grid, bitfieldExtract(aux, 7 * int(iqs), 7));
        buf_a[buf_ib].qs[2 * iqs    ] = q.x;
        buf_a[buf_ib].qs[2 * iqs + 1] = q.y;

        if (iqs == 0) {
            buf_a[buf_ib].d = FLOAT_TYPE(float(data_a_packed16[ib_k].d) * db);
        }
    } else {
        buf_a[buf_ib].qs[2 * iqs] = 0;
        buf_a[buf_ib].qs[2 * iqs + 1] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d = FLOAT_TYPE(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d = buf_a[buf_ib].d;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
}
#endif

#if defined(DATA_A_IQ2_XS)
// Each iqs handles one group of 8 values: 9-bit grid index + 7 sign bits, one scale per 16 values
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint ib32 = ib % 8;

        const uint qs = uint(data_a_packed16[ib_k].qs[4 * ib32 + iqs]);
        const i32vec2 q = iq_apply_signs8(iq2xs_grid[qs & 511], qs >> 9);
        buf_a[buf_ib].qs[2 * iqs    ] = q.x;
        buf_a[buf_ib].qs[2 * iqs + 1] = q.y;

        if (iqs == 0) {
            const uint sc = (uint(data_a_packed16[ib_k].scales[ib32 / 2]) >> ((ib32 & 1) * 8)) & 0xFF;
            const float d = float(data_a_packed16[ib_k].d) * 0.125f;

            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(d * float(1 + 2 * (sc & 0xF)), d * float(1 + 2 * (sc >> 4)));
        }
    } else {
        buf_a[buf_ib].qs[2 * iqs] = 0;
        buf_a[buf_ib].qs[2 * iqs + 1] = 0;

        if (iqs == 0) {
            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(0.0f);
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum0 = 0;
    int32_t q_sum1 = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        q_sum0 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }
    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        q_sum1 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].d_scales[0]) * float(q_sum0) +
                                           float(cache_a[ib_a].d_scales[1]) * float(q_sum1)));
}
#endif

#if defined(DATA_A_IQ1_S) || defined(DATA_A_IQ1_M)
// The grid holds g + 1 (g in {-1, 0, 1}) in 4-bit fields and delta is +-1/8, so 8 * (g + delta)
// = 8 * nibble - 8 + sign is an exact int8. Returns values 0..3 in .x and 4..7 in .y.
i32vec2 iq1_grid_to_i8x8(const uint grid, const int sign) {
    const ivec4 lo = ivec4(unpack8((grid     ) & 0x0F0F0F0Fu));
    const ivec4 hi = ivec4(unpack8((grid >> 4) & 0x0F0F0F0Fu));
    return i32vec2(pack32(i8vec4(8 * lo - 8 + sign)), pack32(i8vec4(8 * hi - 8 + sign)));
}

// Each iqs handles one group of 8 values
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_k = ib / 8;
        const uint ib32 = ib % 8;
        const uint qs = (uint(data_a_packed16[ib_k].qs[2 * ib32 + iqs / 2]) >> ((iqs & 1) * 8)) & 0xFF;

#if defined(DATA_A_IQ1_S)
        // per 32 values: 4x3 high index bits, 3-bit scale, delta sign
        const uint qh = uint(data_a[ib_k].qh[ib32]);
        const uint idx = qs | (bitfieldExtract(qh, 3 * int(iqs), 3) << 8);
        const int sign = (qh & 0x8000) != 0 ? -1 : 1;
#else
        // per 8 values: 3 high index bits and a delta sign bit
        const uint qh = (uint(data_a_packed16[ib_k].qh[ib32]) >> (4 * iqs)) & 0xF;
        const uint idx = qs | ((qh & 7) << 8);
        const int sign = (qh & 8) != 0 ? -1 : 1;
#endif

        const i32vec2 q = iq1_grid_to_i8x8(iq1s_grid_gpu[idx], sign);
        buf_a[buf_ib].qs[2 * iqs    ] = q.x;
        buf_a[buf_ib].qs[2 * iqs + 1] = q.y;

        if (iqs == 0) {
#if defined(DATA_A_IQ1_S)
            const float dl = float(data_a[ib_k].d) * float(2 * bitfieldExtract(qh, 12, 3) + 1);
            buf_a[buf_ib].d = FLOAT_TYPE(dl * 0.125f);
#else
            // fp16 d is spread over the top 4 bits of the 4 scale words
            const u16vec4 s = u16vec4(data_a[ib_k].scales[0], data_a[ib_k].scales[1], data_a[ib_k].scales[2], data_a[ib_k].scales[3]) >> 12;
            const float d = unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x * 0.125f;
            const uint sc = uint(data_a[ib_k].scales[ib32 / 2]);
            const uint ib16 = (2 * ib32) & 3;
            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(d * float(2 * bitfieldExtract(sc, 3 * int(ib16    ), 3) + 1),
                                                  d * float(2 * bitfieldExtract(sc, 3 * int(ib16 + 1), 3) + 1));
#endif
        }
    } else {
        buf_a[buf_ib].qs[2 * iqs] = 0;
        buf_a[buf_ib].qs[2 * iqs + 1] = 0;

        if (iqs == 0) {
#if defined(DATA_A_IQ1_S)
            buf_a[buf_ib].d = FLOAT_TYPE(0.0f);
#else
            buf_a[buf_ib].d_scales = FLOAT_TYPEV2(0.0f);
#endif
        }
    }
}

void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
#if defined(DATA_A_IQ1_S)
    cache_a[reg_ib].d = buf_a[buf_ib].d;
#else
    cache_a[reg_ib].d_scales = buf_a[buf_ib].d_scales;
#endif

    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}

ACC_TYPE mmq_dot_product(const uint ib_a) {
#if defined(DATA_A_IQ1_S)
    int32_t q_sum = 0;
    [[unroll]] for (uint iqs = 0; iqs < 8; iqs++) {
        q_sum += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_a[ib_a].d) * float(cache_b.ds.x) * float(q_sum));
#else
    int32_t q_sum0 = 0;
    int32_t q_sum1 = 0;
    [[unroll]] for (uint iqs = 0; iqs < 4; iqs++) {
        q_sum0 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }
    [[unroll]] for (uint iqs = 4; iqs < 8; iqs++) {
        q_sum1 += dotPacked4x8EXT(cache_a[ib_a].qs[iqs], cache_b.qs[iqs]);
    }

    return ACC_TYPE(float(cache_b.ds.x) * (float(cache_a[ib_a].d_scales[0]) * float(q_sum0) +
                                           float(cache_a[ib_a].d_scales[1]) * float(q_sum1)));
#endif
}
#endif

void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs, const bool is_in_bounds) {
    if (is_in_bounds) {
        const uint ib_outer = ib / 4;
        const uint ib_inner = ib % 4;

        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(data_b[ib_outer].ds[ib_inner]);
        }

        const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
        buf_b[buf_ib].qs[iqs * 4    ] = values.x;
        buf_b[buf_ib].qs[iqs * 4 + 1] = values.y;
        buf_b[buf_ib].qs[iqs * 4 + 2] = values.z;
        buf_b[buf_ib].qs[iqs * 4 + 3] = values.w;
#ifdef MMQ_B_SUMS
        if (MMQ_SUDOT) {
            buf_b_qsum[buf_ib * 2 + iqs] = dotPacked4x8EXT(values.x, 0x01010101u) + dotPacked4x8EXT(values.y, 0x01010101u) +
                                           dotPacked4x8EXT(values.z, 0x01010101u) + dotPacked4x8EXT(values.w, 0x01010101u);
        }
#endif
    } else {
        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(0.0f);
        }

        buf_b[buf_ib].qs[iqs * 4    ] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 1] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 2] = 0;
        buf_b[buf_ib].qs[iqs * 4 + 3] = 0;
#ifdef MMQ_B_SUMS
        if (MMQ_SUDOT) {
            buf_b_qsum[buf_ib * 2 + iqs] = 0;
        }
#endif
    }
}

void block_b_to_registers(const uint ib) {
    cache_b.ds = buf_b[ib].ds;
    [[unroll]] for (uint iqs = 0; iqs < BK / 4; iqs++) {
        cache_b.qs[iqs] = buf_b[ib].qs[iqs];
    }
#ifdef MMQ_B_SUMS
    if (MMQ_SUDOT) {
        cache_b_qsum[0] = buf_b_qsum[ib * 2];
        cache_b_qsum[1] = buf_b_qsum[ib * 2 + 1];
    }
#endif
}
