// K/V loaders for the Intel Xe decode kernels (flash_attn_decode_phase_{1,2}.comp).
//
// Default (F16): binding 1 is read as f16vec4 and XE_LOAD_KV4(i) is a plain load,
// where i is an f16vec4 index (= element offset / 4).
//
// DATA_KV_Q8_0: binding 1 is read as raw q8_0 blocks (34 bytes = 17 x u16:
// one f16 scale followed by 32 int8 quants) and dequantized to f16vec4 on load.
// Strides in the push constants are then in elements (blocks * 32), so the
// same element-offset arithmetic as the F16 path can be reused unchanged.
// Each 4-element group lies inside a single block because all callers use
// offsets that are multiples of 4 and rows are whole blocks (HS % 32 == 0).
//
// Decode is bandwidth bound: reading q8_0 directly moves ~1.06 B/element
// instead of 2 B/element for F16, without an extra dequant pass.

#if defined(DATA_KV_Q8_0)

// Requires GL_EXT_shader_explicit_arithmetic_types_int16 (enabled by the includer).

layout (binding = 1) readonly buffer KVQ8 { uint16_t data_kvq[]; };

const uint XE_Q8_0_BLOCK_U16 = 17u;

f16vec4 xe_dequant_q8_0_4(const uint e) {
    const uint ib   = e >> 5;
    const uint iqs  = e & 31u;
    const uint base = ib * XE_Q8_0_BLOCK_U16;
    const float16_t d = uint16BitsToHalf(data_kvq[base]);
    const int w = int(uint(data_kvq[base + 1u + (iqs >> 1)]) |
                     (uint(data_kvq[base + 2u + (iqs >> 1)]) << 16));
    const ivec4 q = ivec4(bitfieldExtract(w,  0, 8), bitfieldExtract(w,  8, 8),
                          bitfieldExtract(w, 16, 8), bitfieldExtract(w, 24, 8));
    return f16vec4(q) * d;
}

#define XE_LOAD_KV4(i) xe_dequant_q8_0_4((i) * 4u)

#else

layout (binding = 1) readonly buffer KV4 { f16vec4 data_kv4[]; };

#define XE_LOAD_KV4(i) data_kv4[i]

#endif
