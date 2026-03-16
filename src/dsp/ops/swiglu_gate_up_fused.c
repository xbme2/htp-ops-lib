// Baseline fused kernel for:
//   gate_matmul(iq4_nl, fp32->fp32)
//   up_matmul(iq4_nl, fp32->fp32)
//   SiLU(gate) * up
//
// This is the correctness-first, non-pipelined version.
// It reuses the same HMX/HVX dataflow style as mat_mul.c, but keeps the
// intermediate gate/up outputs inside VTCM instead of storing them to DDR.
// Weight load -> dequant -> gate matmul -> weight load -> dequant -> up matmul
// -> SWIGLU fuse are executed sequentially on purpose.

#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/quants.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"

#define FUSED_WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define FUSED_ACTIVATION_AREA_SIZE (1 * 1024 * 1024)
#define FUSED_OUTPUT_AREA_SIZE     (1 * 1024 * 1024)
#define FUSED_QWEIGHT_AREA_SIZE    (1 * 1024 * 1024)
#define SILU_LUT_MAX_BITS          12
#define SILU_LUT_MAX_ENTRIES       ((1 << SILU_LUT_MAX_BITS) + 1)

static inline size_t get_super_block_size_local(enum ggml_type weight_type) {
  switch (weight_type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_IQ4_NL:
      return sizeof(my_block_q4_0);
    case GGML_TYPE_Q8_0:
      return sizeof(my_block_q8_0);
    default:
      return 0;
  }
}

static inline int dma_issue_load_from_ddr_local(dma_desc_1d_t *desc, void *vtcm_dst, const void *src, size_t size) {
  dma_wait_for_idle();

  desc->next       = 0;
  desc->length     = size;
  desc->type       = DMA_DESC_TYPE_1D;
  desc->src_bypass = 1;
  desc->dst_bypass = 0;
  desc->ordered    = 1;
  desc->dstate     = DMA_DESC_DSTATE_PENDING;
  desc->src        = (uint32_t) src;
  desc->dst        = (uint32_t) vtcm_dst;

  return dma_submit_one(desc);
}

static void find_chunk_size_local(size_t x_max, size_t y_max, size_t xy_max, size_t x_unit, size_t y_unit, size_t *x_out,
                                  size_t *y_out) {
  int64_t best_xy = 0;
  size_t best_x = 0;
  size_t best_y = 0;

  for (size_t x = x_max; x > 0; x -= x_unit) {
    const size_t y = smin(align_down(xy_max / x, y_unit), y_max);
    const int64_t xy = x * y;
    if (best_xy < xy) {
      best_xy = xy;
      best_x = x;
      best_y = y;
    }
  }

  *x_out = best_x;
  *y_out = best_y;
}

static void transfer_activation_chunk_fp32_to_fp16_local(__fp16 *restrict vtcm_dst, const float *restrict src, int n_rows,
                                                         int k_block, int k_stride) {
  assert(k_block % HMX_FP16_TILE_N_COLS == 0 && k_stride % HMX_FP16_TILE_N_COLS == 0);
  assert(VLEN == 32 * sizeof(float));

  for (int r = 0; r < n_rows; r += 2) {
    const int prefetch_row_idx = r + 2;
    if (prefetch_row_idx < n_rows) {
      const float *prefetch_addr = src + prefetch_row_idx * k_stride;
      l2fetch(prefetch_addr, k_stride * sizeof(float), k_block * sizeof(float), 2, 0);
    }

    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;
    const bool next_row_valid = (r + 1) < n_rows;

    const HVX_Vector *pv_in0 = (const HVX_Vector *) (src + (r + 0) * k_stride);
    const HVX_Vector *pv_in1 = (const HVX_Vector *) (src + (r + 1) * k_stride);

    for (int c = 0; c < k_block; c += 32) {
      const HVX_Vector v0 = *pv_in0++;
      const HVX_Vector v1 = next_row_valid ? *pv_in1++ : Q6_V_vzero();
      const HVX_Vector v_out = hvx_my_wsf_to_vhf(v1, v0);

      const int c0 = c / HMX_FP16_TILE_N_COLS;
      const int tile_idx = r0 * (k_block / HMX_FP16_TILE_N_COLS) + c0;
      HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HMX_FP16_TILE_N_ELMS);
      tile[r1 / 2] = v_out;
    }
  }
}

static void core_dot_chunk_fp16_local(__fp16 *output, const __fp16 *activation, const __fp16 *weight, const __fp16 *scales,
                                      int n_row_tiles, int n_col_tiles, int n_dot_tiles) {
  hmx_unit_acquire();

  asm volatile("mxclracc.hf");
  hmx_set_output_scales(scales);

  for (int r = 0; r < n_row_tiles; ++r) {
    for (int c = 0; c < n_col_tiles; ++c) {
      const __fp16 *row_tiles = activation + r * n_dot_tiles * HMX_FP16_TILE_N_ELMS;
      const __fp16 *col_tiles = weight + c * n_dot_tiles * HMX_FP16_TILE_N_ELMS;

      for (int k0 = 0; k0 < n_dot_tiles; k0 += 32) {
        const int offset = k0 * HMX_FP16_TILE_N_ELMS;
        const size_t n_tiles = smin(n_dot_tiles - k0, 32);
        hmx_load_tiles_fp16(row_tiles + offset, col_tiles + offset, n_tiles);
      }

      __fp16 *out_tile = output + (r * n_col_tiles + c) * HMX_FP16_TILE_N_ELMS;
      hmx_consume_accumulator_fp16(out_tile);
    }
  }

  hmx_unit_release();
}

static inline HVX_Vector hvx_silu_vec_f32_local(HVX_Vector v_x_sf) {
  const HVX_Vector v_zero   = Q6_V_vzero();
  const HVX_Vector v_one_sf = Q6_V_vsplat_R(0x3F800000);
  const HVX_Vector v_log2e  = Q6_V_vsplat_R(0xBFB8AA3C);  // -1 / ln(2)

  const HVX_Vector v_scaled_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_x_sf, v_log2e);
  const HVX_Vector v_scaled_sf   = Q6_Vsf_equals_Vqf32(v_scaled_qf32);
  const HVX_Vector v_exp_sf      = hvx_my_exp2_vsf(v_scaled_sf);

  const HVX_Vector v_denom_qf32 = Q6_Vqf32_vadd_VsfVsf(v_exp_sf, v_one_sf);
  const HVX_Vector v_denom_sf   = Q6_Vsf_equals_Vqf32(v_denom_qf32);
  const HVX_Vector v_inv_qf32   = hvx_my_inv_vqf32_vsf(v_denom_sf);

  const HVX_Vector v_x_qf32   = Q6_Vqf32_vadd_VsfVsf(v_x_sf, v_zero);
  const HVX_Vector v_out_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(v_x_qf32, v_inv_qf32);
  return Q6_Vsf_equals_Vqf32(v_out_qf32);
}

static inline float silu_exact_scalar_local(float x) {
  return x / (1.0f + expf(-x));
}

static void build_silu_lut_local(float *lut, int lut_size, float clamp) {
  const float step = (2.0f * clamp) / (float) lut_size;
  for (int i = 0; i <= lut_size; ++i) {
    const float x = -clamp + step * (float) i;
    lut[i] = silu_exact_scalar_local(x);
  }
}

static inline float silu_lut_scalar_local(float x, const float *lut, int lut_size, float clamp) {
  if (x <= -clamp) {
    return lut[0];
  }
  if (x >= clamp) {
    return lut[lut_size];
  }

  const float step = (2.0f * clamp) / (float) lut_size;
  const float u = (x + clamp) / step;
  int i = (int) u;
  if (i < 0) {
    i = 0;
  } else if (i >= lut_size) {
    i = lut_size - 1;
  }

  const float t = u - (float) i;
  const float y0 = lut[i];
  const float y1 = lut[i + 1];
  return y0 + (y1 - y0) * t;
}

static void fuse_gate_up_chunk_fp16_to_fp32_local(float *restrict dst,
                                                  const __fp16 *restrict gate_vtcm,
                                                  const __fp16 *restrict up_vtcm,
                                                  int n_rows,
                                                  int n_cols,
                                                  int dst_stride,
                                                  const float *restrict silu_lut,
                                                  int silu_lut_size,
                                                  float silu_lut_clamp,
                                                  bool use_silu_lut) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  _Alignas(VLEN) float gate_row0[32];
  _Alignas(VLEN) float gate_row1[32];
  _Alignas(VLEN) float up_row0[32];
  _Alignas(VLEN) float up_row1[32];
  _Alignas(VLEN) float out_row0[32];
  _Alignas(VLEN) float out_row1[32];

  for (int r = 0; r < n_rows; r += 2) {
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      const int c0 = c / HMX_FP16_TILE_N_COLS;

      const __fp16 *gate_tile = gate_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *up_tile   = up_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      const HVX_Vector v_gate_hf = ((const HVX_Vector *) gate_tile)[r1 / 2];
      const HVX_Vector v_up_hf   = ((const HVX_Vector *) up_tile)[r1 / 2];

      const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);
      const HVX_VectorPair vp_up   = hvx_my_vhf_to_wsf(v_up_hf);

      HVX_Vector v_out0_sf;
      HVX_Vector v_out1_sf;

      if (!use_silu_lut) {
        const HVX_Vector v_silu0_sf = hvx_silu_vec_f32_local(Q6_V_lo_W(vp_gate));
        const HVX_Vector v_silu1_sf = hvx_silu_vec_f32_local(Q6_V_hi_W(vp_gate));

        const HVX_Vector v_mul0_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_silu0_sf, Q6_V_lo_W(vp_up));
        const HVX_Vector v_mul1_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_silu1_sf, Q6_V_hi_W(vp_up));

        v_out0_sf = Q6_Vsf_equals_Vqf32(v_mul0_qf32);
        v_out1_sf = Q6_Vsf_equals_Vqf32(v_mul1_qf32);
      } else {
        vmem(gate_row0) = Q6_V_lo_W(vp_gate);
        vmem(gate_row1) = Q6_V_hi_W(vp_gate);
        vmem(up_row0)   = Q6_V_lo_W(vp_up);
        vmem(up_row1)   = Q6_V_hi_W(vp_up);

        for (int i = 0; i < 32; ++i) {
          out_row0[i] = silu_lut_scalar_local(gate_row0[i], silu_lut, silu_lut_size, silu_lut_clamp) * up_row0[i];
          out_row1[i] = silu_lut_scalar_local(gate_row1[i], silu_lut, silu_lut_size, silu_lut_clamp) * up_row1[i];
        }

        v_out0_sf = vmem(out_row0);
        v_out1_sf = vmem(out_row1);
      }

      HVX_Vector *pv_out0 = (HVX_Vector *) (dst + r * dst_stride + c);
      *pv_out0 = v_out0_sf;

      if (r + 1 < n_rows) {
        HVX_Vector *pv_out1 = (HVX_Vector *) (dst + (r + 1) * dst_stride + c);
        *pv_out1 = v_out1_sf;
      }
    }
  }
}

void dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne, int k,
                                                       enum ggml_type type, void *vtcm_scratch);

int hmx_hvx_swiglu_gate_up_fused_qk_0_d16a32(float *restrict dst,
                                             const float *restrict activation,
                                             const uint8_t *restrict gate_weight,
                                             const uint8_t *restrict up_weight,
                                             int m,
                                             int k,
                                             int n,
                                             enum ggml_type weight_type,
                                             int silu_lut_bits,
                                             float silu_lut_clamp,
                                             bool use_silu_lut) {
  if (!dst || !activation || !gate_weight || !up_weight || !m || !k || !n) {
    return -1;
  }
  if (weight_type != GGML_TYPE_IQ4_NL) {
    return -1;
  }
  if (k % QK_K != 0 || n % 32 != 0) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) || !is_aligned(gate_weight, VLEN) || !is_aligned(up_weight, VLEN)) {
    return -1;
  }
  if (use_silu_lut &&
      (silu_lut_bits <= 0 || silu_lut_bits > SILU_LUT_MAX_BITS || !(silu_lut_clamp > 0.0f))) {
    return -1;
  }

  _Alignas(VLEN) float silu_lut[SILU_LUT_MAX_ENTRIES];
  int silu_lut_size = 0;
  if (use_silu_lut) {
    silu_lut_size = 1 << silu_lut_bits;
    build_silu_lut_local(silu_lut, silu_lut_size, silu_lut_clamp);
  }

  const size_t super_block_size = get_super_block_size_local(weight_type);
  if (super_block_size == 0) {
    return -1;
  }

  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, FUSED_WEIGHT_AREA_SIZE);
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, FUSED_ACTIVATION_AREA_SIZE);
  __fp16  *vtcm_gate_out   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, FUSED_OUTPUT_AREA_SIZE);
  __fp16  *vtcm_up_out     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, FUSED_OUTPUT_AREA_SIZE);
  void    *vtcm_qweight    = vtcm_seq_alloc(&vtcm_ptr, FUSED_QWEIGHT_AREA_SIZE);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

  const size_t vec_dot_size       = k * sizeof(__fp16);
  const size_t m_chunk_max_n_rows = align_down(FUSED_ACTIVATION_AREA_SIZE / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  const size_t n_chunk_max_n_cols = align_down(FUSED_WEIGHT_AREA_SIZE / vec_dot_size, HMX_FP16_TILE_N_COLS);

  size_t m_chunk_n_rows = 0;
  size_t n_chunk_n_cols = 0;
  find_chunk_size_local(m_chunk_max_n_rows, n_chunk_max_n_cols, FUSED_OUTPUT_AREA_SIZE / sizeof(__fp16),
                        HMX_FP16_TILE_N_ROWS, HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  if (m_chunk_n_rows == 0 || n_chunk_n_cols == 0) {
    return -1;
  }

  dma_desc_1d_t dma_desc __attribute__((aligned(64)));

  for (size_t mr = 0; mr < (size_t) m; mr += m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) m - mr, m_chunk_n_rows);
    const float *activation_chunk = activation + mr * k;

    transfer_activation_chunk_fp32_to_fp16_local(vtcm_activation, activation_chunk, (int) n_rows, k, k);

    for (size_t nc = 0; nc < (size_t) n; nc += n_chunk_n_cols) {
      const size_t n_cols = smin((size_t) n - nc, n_chunk_n_cols);
      const size_t chunk_ne = n_cols * k;
      assert(chunk_ne % QK_K == 0);
      const size_t qweight_chunk_size = chunk_ne / QK_K * super_block_size;
      const size_t weight_offset = (nc * k / QK_K) * super_block_size;

      dma_issue_load_from_ddr_local(&dma_desc, vtcm_qweight, gate_weight + weight_offset, qweight_chunk_size);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, NULL, (int) chunk_ne, k, weight_type, vtcm_qweight);
      core_dot_chunk_fp16_local(vtcm_gate_out, vtcm_activation, vtcm_weight, vtcm_scales,
                                (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS),
                                (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS),
                                k / HMX_FP16_TILE_N_COLS);

      dma_issue_load_from_ddr_local(&dma_desc, vtcm_qweight, up_weight + weight_offset, qweight_chunk_size);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight, NULL, (int) chunk_ne, k, weight_type, vtcm_qweight);
      core_dot_chunk_fp16_local(vtcm_up_out, vtcm_activation, vtcm_weight, vtcm_scales,
                                (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS),
                                (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS),
                                k / HMX_FP16_TILE_N_COLS);

      fuse_gate_up_chunk_fp16_to_fp32_local(dst + mr * n + nc,
                                            vtcm_gate_out,
                                            vtcm_up_out,
                                            (int) n_rows,
                                            (int) n_cols,
                                            n,
                                            silu_lut,
                                            silu_lut_size,
                                            silu_lut_clamp,
                                            use_silu_lut);
    }
  }

  return 0;
}
