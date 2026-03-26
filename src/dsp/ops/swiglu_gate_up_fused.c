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
#include <stdlib.h>

#include "dsp/dma_utils.h"
#include "dsp/hmx_mgr.h"
#include "dsp/hmx_utils.h"
#include "dsp/hvx_convert.h"
#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"
#include "dsp/quants.h"
#include "dsp/utils.h"
#include "dsp/vtcm_mgr.h"
#include "dsp/worker_pool.h"

#define FUSED_WEIGHT_AREA_SIZE     (1 * 1024 * 1024)
#define FUSED_ACTIVATION_AREA_SIZE (1 * 1024 * 1024)
#define FUSED_OUTPUT_AREA_SIZE     (1 * 1024 * 1024)
#define FUSED_QWEIGHT_AREA_SIZE    (1 * 1024 * 1024)
#define SILU_LUT_MAX_BITS          12
#define SILU_LUT_MAX_ENTRIES       ((1 << SILU_LUT_MAX_BITS) + 1)

// Active stage selection:
//   0 -> correctness-first baseline
//   1 -> branch-interleaved qweight prefetch
//   2 -> smaller-VTCM dual-weight-buffer pipeline
//   3 -> smaller-VTCM stage2 + double-output-buffer fuse/store overlap
//   4 -> high-m pipeline-tuned chunking on top of stage3
//   5 -> two-pass true pipeline (gate pass with SiLU store, then up pass with in-place mul)
//   6 -> single-function joint pipeline (interleaved gate/up 4-stage schedule)
#ifndef SWIGLU_GATE_UP_ACTIVE_STAGE
#define SWIGLU_GATE_UP_ACTIVE_STAGE 1
#endif

#define FUSED_STAGE_PIPE_WEIGHT_AREA_SIZE  (768 * 1024)
#define FUSED_STAGE_PIPE_OUTPUT_AREA_SIZE  (768 * 1024)
#define FUSED_STAGE_PIPE_QWEIGHT_AREA_SIZE (768 * 1024)

static inline size_t get_weight_area_size_local(void) {
#if SWIGLU_GATE_UP_ACTIVE_STAGE >= 2
  return FUSED_STAGE_PIPE_WEIGHT_AREA_SIZE;
#else
  return FUSED_WEIGHT_AREA_SIZE;
#endif
}

static inline size_t get_activation_area_size_local(void) {
  return FUSED_ACTIVATION_AREA_SIZE;
}

static inline size_t get_output_area_size_local(void) {
#if SWIGLU_GATE_UP_ACTIVE_STAGE >= 2
  return FUSED_STAGE_PIPE_OUTPUT_AREA_SIZE;
#else
  return FUSED_OUTPUT_AREA_SIZE;
#endif
}

static inline size_t get_qweight_area_size_local(void) {
#if SWIGLU_GATE_UP_ACTIVE_STAGE >= 2
  return FUSED_STAGE_PIPE_QWEIGHT_AREA_SIZE;
#else
  return FUSED_QWEIGHT_AREA_SIZE;
#endif
}

static inline bool stage_uses_weight_aux_local(void) {
#if SWIGLU_GATE_UP_ACTIVE_STAGE >= 2
  return true;
#else
  return false;
#endif
}

static inline bool stage_uses_output_aux_local(void) {
#if SWIGLU_GATE_UP_ACTIVE_STAGE == 3 || SWIGLU_GATE_UP_ACTIVE_STAGE == 4 || SWIGLU_GATE_UP_ACTIVE_STAGE == 5
  return true;
#else
  return false;
#endif
}

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

static inline size_t align_up_local(size_t x, size_t a) {
  return ((x + a - 1) / a) * a;
}

static void choose_chunk_shape_high_m_pipeline_local(int m,
                                                     int k,
                                                     int n,
                                                     size_t weight_area_size,
                                                     size_t activation_area_size,
                                                     size_t output_area_size,
                                                     size_t qweight_area_size,
                                                     size_t super_block_size,
                                                     size_t *m_chunk_n_rows,
                                                     size_t *n_chunk_n_cols) {
  const size_t vec_dot_size = (size_t) k * sizeof(__fp16);
  const size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  const size_t n_chunk_max_n_cols_by_weight = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);
  const size_t qweight_bytes_per_col = ((size_t) k / QK_K) * super_block_size;
  const size_t n_chunk_max_n_cols_by_qweight = qweight_bytes_per_col == 0
                                                   ? 0
                                                   : align_down(qweight_area_size / qweight_bytes_per_col,
                                                                HMX_FP16_TILE_N_COLS);
  const size_t n_chunk_max_n_cols = smin(n_chunk_max_n_cols_by_weight, n_chunk_max_n_cols_by_qweight);

  size_t preferred_rows = align_up_local((size_t) m, HMX_FP16_TILE_N_ROWS);
  preferred_rows = smin(preferred_rows, m_chunk_max_n_rows);

  if (m >= 128) {
    preferred_rows = smax(preferred_rows, (size_t) 128);
  } else if (m >= 64) {
    preferred_rows = smax(preferred_rows, (size_t) 64);
  }
  preferred_rows = smin(preferred_rows, m_chunk_max_n_rows);

  const size_t n_cols_cap_by_output = align_down((output_area_size / sizeof(__fp16)) / preferred_rows,
                                                 HMX_FP16_TILE_N_COLS);
  size_t preferred_cols = smin(n_chunk_max_n_cols, n_cols_cap_by_output);
  preferred_cols = smin(preferred_cols, align_up_local((size_t) smax(n, HMX_FP16_TILE_N_COLS), HMX_FP16_TILE_N_COLS));
  if (preferred_cols == 0) {
    preferred_cols = HMX_FP16_TILE_N_COLS;
  }

  *m_chunk_n_rows = preferred_rows;
  *n_chunk_n_cols = preferred_cols;
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

static inline const uint8_t *query_silu_neg_table_local(void) {
  return (const uint8_t *) vtcm_manager_query_area("swiglu::silu_neg_hf");
}

static inline HVX_Vector vhf_abs_local(HVX_Vector v_hf) {
  return Q6_V_vand_VV(v_hf, Q6_Vh_vsplat_R(0x7fff));
}

static inline HVX_Vector vhf_force_negative_local(HVX_Vector v_hf) {
  return Q6_V_vor_VV(v_hf, Q6_Vh_vsplat_R(0x8000));
}

static inline HVX_Vector silu_vhf_from_neg_table_local(HVX_Vector v_gate_hf, const uint8_t *silu_neg_table) {
  const HVX_Vector v_zero_sf = Q6_V_vzero();

  const HVX_Vector v_neg_abs_gate_hf = vhf_force_negative_local(vhf_abs_local(v_gate_hf));
  const HVX_Vector v_gather_input = Q6_Vh_vasl_VhR(v_neg_abs_gate_hf, 1);

  _Alignas(VLEN) HVX_Vector v_silu_neg_hf;
  Q6_vgather_ARMVh(&v_silu_neg_hf, (size_t) silu_neg_table, 65535, v_gather_input);

  const HVX_VectorPair vp_gate     = hvx_my_vhf_to_wsf(v_gate_hf);
  const HVX_VectorPair vp_silu_neg = hvx_my_vhf_to_wsf(v_silu_neg_hf);

  const HVX_Vector v_gate0_sf     = Q6_V_lo_W(vp_gate);
  const HVX_Vector v_gate1_sf     = Q6_V_hi_W(vp_gate);
  const HVX_Vector v_silu_neg0_sf = Q6_V_lo_W(vp_silu_neg);
  const HVX_Vector v_silu_neg1_sf = Q6_V_hi_W(vp_silu_neg);

  const HVX_VectorPred q_gate0_neg = Q6_Q_vcmp_gt_VsfVsf(v_zero_sf, v_gate0_sf);
  const HVX_VectorPred q_gate1_neg = Q6_Q_vcmp_gt_VsfVsf(v_zero_sf, v_gate1_sf);

  const HVX_Vector v_silu_pos0_sf =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_gate0_sf, v_silu_neg0_sf));
  const HVX_Vector v_silu_pos1_sf =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_gate1_sf, v_silu_neg1_sf));

  const HVX_Vector v_silu0_sf = Q6_V_vmux_QVV(q_gate0_neg, v_silu_neg0_sf, v_silu_pos0_sf);
  const HVX_Vector v_silu1_sf = Q6_V_vmux_QVV(q_gate1_neg, v_silu_neg1_sf, v_silu_pos1_sf);
  return hvx_my_wsf_to_vhf(v_silu1_sf, v_silu0_sf);
}

static void fuse_gate_up_fp32_local(float *restrict dst,
                                    const float *restrict gate,
                                    const float *restrict up,
                                    size_t n_elems,
                                    const float *restrict silu_lut,
                                    int silu_lut_size,
                                    float silu_lut_clamp,
                                    bool use_silu_lut) {
  for (size_t i = 0; i < n_elems; ++i) {
    const float gate_val = gate[i];
    const float silu_val = use_silu_lut
                               ? silu_lut_scalar_local(gate_val, silu_lut, silu_lut_size, silu_lut_clamp)
                               : silu_exact_scalar_local(gate_val);
    dst[i] = silu_val * up[i];
  }
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
  const uint8_t *silu_neg_table = use_silu_lut ? NULL : query_silu_neg_table_local();

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

      HVX_Vector v_out0_sf;
      HVX_Vector v_out1_sf;

      if (!use_silu_lut && silu_neg_table) {
        const HVX_Vector v_silu_hf = silu_vhf_from_neg_table_local(v_gate_hf, silu_neg_table);
        const HVX_VectorPair vp_silu = hvx_my_vhf_to_wsf(v_silu_hf);
        const HVX_VectorPair vp_up   = hvx_my_vhf_to_wsf(v_up_hf);

        const HVX_Vector v_mul0_qf32 = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vp_silu), Q6_V_lo_W(vp_up));
        const HVX_Vector v_mul1_qf32 = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vp_silu), Q6_V_hi_W(vp_up));

        v_out0_sf = Q6_Vsf_equals_Vqf32(v_mul0_qf32);
        v_out1_sf = Q6_Vsf_equals_Vqf32(v_mul1_qf32);
      } else if (!use_silu_lut) {
        const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);
        const HVX_VectorPair vp_up   = hvx_my_vhf_to_wsf(v_up_hf);

        const HVX_Vector v_silu0_sf = hvx_silu_vec_f32_local(Q6_V_lo_W(vp_gate));
        const HVX_Vector v_silu1_sf = hvx_silu_vec_f32_local(Q6_V_hi_W(vp_gate));

        const HVX_Vector v_mul0_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_silu0_sf, Q6_V_lo_W(vp_up));
        const HVX_Vector v_mul1_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_silu1_sf, Q6_V_hi_W(vp_up));

        v_out0_sf = Q6_Vsf_equals_Vqf32(v_mul0_qf32);
        v_out1_sf = Q6_Vsf_equals_Vqf32(v_mul1_qf32);
      } else {
        const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);
        const HVX_VectorPair vp_up   = hvx_my_vhf_to_wsf(v_up_hf);
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

static void silu_gate_chunk_fp16_to_fp32_local(float *restrict dst,
                                               const __fp16 *restrict gate_vtcm,
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
  _Alignas(VLEN) float out_row0[32];
  _Alignas(VLEN) float out_row1[32];

  for (int r = 0; r < n_rows; r += 2) {
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      const int c0 = c / HMX_FP16_TILE_N_COLS;
      const __fp16 *gate_tile = gate_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;
      const HVX_Vector v_gate_hf = ((const HVX_Vector *) gate_tile)[r1 / 2];
      const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);

      HVX_Vector v_out0_sf;
      HVX_Vector v_out1_sf;

      if (!use_silu_lut) {
        v_out0_sf = hvx_silu_vec_f32_local(Q6_V_lo_W(vp_gate));
        v_out1_sf = hvx_silu_vec_f32_local(Q6_V_hi_W(vp_gate));
      } else {
        vmem(gate_row0) = Q6_V_lo_W(vp_gate);
        vmem(gate_row1) = Q6_V_hi_W(vp_gate);

        for (int i = 0; i < 32; ++i) {
          out_row0[i] = silu_lut_scalar_local(gate_row0[i], silu_lut, silu_lut_size, silu_lut_clamp);
          out_row1[i] = silu_lut_scalar_local(gate_row1[i], silu_lut, silu_lut_size, silu_lut_clamp);
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

static void silu_gate_chunk_fp16_inplace_local(__fp16 *restrict gate_vtcm,
                                               int n_rows,
                                               int n_cols,
                                               const float *restrict silu_lut,
                                               int silu_lut_size,
                                               float silu_lut_clamp,
                                               bool use_silu_lut) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;
  const uint8_t *silu_neg_table = use_silu_lut ? NULL : query_silu_neg_table_local();

  _Alignas(VLEN) float gate_row0[32];
  _Alignas(VLEN) float gate_row1[32];
  _Alignas(VLEN) float out_row0[32];
  _Alignas(VLEN) float out_row1[32];

  for (int r = 0; r < n_rows; r += 2) {
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      const int c0 = c / HMX_FP16_TILE_N_COLS;
      __fp16 *gate_tile = gate_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;
      HVX_Vector *pv_gate_tile = (HVX_Vector *) gate_tile;
      const HVX_Vector v_gate_hf = pv_gate_tile[r1 / 2];

      HVX_Vector v_silu_hf;
      if (!use_silu_lut) {
        if (silu_neg_table) {
          v_silu_hf = silu_vhf_from_neg_table_local(v_gate_hf, silu_neg_table);
        } else {
          const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);
          const HVX_Vector v_silu0_sf = hvx_silu_vec_f32_local(Q6_V_lo_W(vp_gate));
          const HVX_Vector v_silu1_sf = hvx_silu_vec_f32_local(Q6_V_hi_W(vp_gate));
          v_silu_hf = hvx_my_wsf_to_vhf(v_silu1_sf, v_silu0_sf);
        }
      } else {
        const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);
        vmem(gate_row0) = Q6_V_lo_W(vp_gate);
        vmem(gate_row1) = Q6_V_hi_W(vp_gate);

        for (int i = 0; i < 32; ++i) {
          out_row0[i] = silu_lut_scalar_local(gate_row0[i], silu_lut, silu_lut_size, silu_lut_clamp);
          out_row1[i] = silu_lut_scalar_local(gate_row1[i], silu_lut, silu_lut_size, silu_lut_clamp);
        }

        v_silu_hf = hvx_my_wsf_to_vhf(vmem(out_row1), vmem(out_row0));
      }

      pv_gate_tile[r1 / 2] = v_silu_hf;
    }
  }
}

static void mul_dst_fp32_by_up_chunk_local(float *restrict dst,
                                           const __fp16 *restrict up_vtcm,
                                           int n_rows,
                                           int n_cols,
                                           int dst_stride) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  for (int r = 0; r < n_rows; r += 2) {
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      const int c0 = c / HMX_FP16_TILE_N_COLS;
      const __fp16 *up_tile = up_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;
      const HVX_Vector v_up_hf = ((const HVX_Vector *) up_tile)[r1 / 2];
      const HVX_VectorPair vp_up = hvx_my_vhf_to_wsf(v_up_hf);

      HVX_Vector *pv_dst0 = (HVX_Vector *) (dst + r * dst_stride + c);
      const HVX_Vector v_dst0_sf = *pv_dst0;
      const HVX_Vector v_mul0_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_dst0_sf, Q6_V_lo_W(vp_up));
      *pv_dst0 = Q6_Vsf_equals_Vqf32(v_mul0_qf32);

      if (r + 1 < n_rows) {
        HVX_Vector *pv_dst1 = (HVX_Vector *) (dst + (r + 1) * dst_stride + c);
        const HVX_Vector v_dst1_sf = *pv_dst1;
        const HVX_Vector v_mul1_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_dst1_sf, Q6_V_hi_W(vp_up));
        *pv_dst1 = Q6_Vsf_equals_Vqf32(v_mul1_qf32);
      }
    }
  }
}

static void mul_gate_up_chunk_fp16_to_fp32_local(float *restrict dst,
                                                 const __fp16 *restrict gate_vtcm,
                                                 const __fp16 *restrict up_vtcm,
                                                 int n_rows,
                                                 int n_cols,
                                                 int dst_stride) {
  assert(n_cols % HMX_FP16_TILE_N_COLS == 0);

  const int n_col_tiles = n_cols / HMX_FP16_TILE_N_COLS;

  for (int r = 0; r < n_rows; r += 2) {
    const int r0 = r / HMX_FP16_TILE_N_ROWS;
    const int r1 = r % HMX_FP16_TILE_N_ROWS;

    for (int c = 0; c < n_cols; c += HMX_FP16_TILE_N_COLS) {
      const int c0 = c / HMX_FP16_TILE_N_COLS;
      const __fp16 *gate_tile = gate_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;
      const __fp16 *up_tile = up_vtcm + (r0 * n_col_tiles + c0) * HMX_FP16_TILE_N_ELMS;

      const HVX_Vector v_gate_hf = ((const HVX_Vector *) gate_tile)[r1 / 2];
      const HVX_Vector v_up_hf = ((const HVX_Vector *) up_tile)[r1 / 2];
      const HVX_VectorPair vp_gate = hvx_my_vhf_to_wsf(v_gate_hf);
      const HVX_VectorPair vp_up = hvx_my_vhf_to_wsf(v_up_hf);

      HVX_Vector *pv_dst0 = (HVX_Vector *) (dst + r * dst_stride + c);
      *pv_dst0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vp_gate), Q6_V_lo_W(vp_up)));

      if (r + 1 < n_rows) {
        HVX_Vector *pv_dst1 = (HVX_Vector *) (dst + (r + 1) * dst_stride + c);
        *pv_dst1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vp_gate), Q6_V_hi_W(vp_up)));
      }
    }
  }
}

static void split_silu_mul_gate_up_chunk_fp16_to_fp32_local(float *restrict dst,
                                                             __fp16 *restrict gate_vtcm,
                                                             const __fp16 *restrict up_vtcm,
                                                             int n_rows,
                                                             int n_cols,
                                                             int dst_stride,
                                                             const float *restrict silu_lut,
                                                             int silu_lut_size,
                                                             float silu_lut_clamp,
                                                             bool use_silu_lut) {
  silu_gate_chunk_fp16_inplace_local(gate_vtcm,
                                     n_rows,
                                     n_cols,
                                     silu_lut,
                                     silu_lut_size,
                                     silu_lut_clamp,
                                     use_silu_lut);
  mul_gate_up_chunk_fp16_to_fp32_local(dst, gate_vtcm, up_vtcm, n_rows, n_cols, dst_stride);
}

typedef struct {
  float *dst;
  const float *activation;
  const uint8_t *gate_weight;
  const uint8_t *up_weight;
  int m;
  int k;
  int n;
  enum ggml_type weight_type;
  size_t super_block_size;
  __fp16 *vtcm_weight;
  __fp16 *vtcm_weight_aux;
  __fp16 *vtcm_activation;
  __fp16 *vtcm_gate_out;
  __fp16 *vtcm_gate_out_aux;
  __fp16 *vtcm_up_out;
  __fp16 *vtcm_up_out_aux;
  void *vtcm_qweight;
  __fp16 *vtcm_scales;
  size_t m_chunk_n_rows;
  size_t n_chunk_n_cols;
  const float *silu_lut;
  int silu_lut_size;
  float silu_lut_clamp;
  bool use_silu_lut;
} swiglu_gate_up_qk_stage_ctx_local_t;

extern worker_pool_context_t hmx_worker_pool_ctx;

typedef struct {
  __fp16            *c;
  const __fp16      *a;
  const __fp16      *b;
  const __fp16      *s;
  int                n_row_tiles;
  int                n_col_tiles;
  int                n_dot_tiles;
  worker_synctoken_t sync_ctx;
} swiglu_gate_up_core_dot_task_state_local_t;

static void swiglu_gate_up_core_dot_hmx_worker_local(void *data, int _worker_index) {
  (void) _worker_index;
  swiglu_gate_up_core_dot_task_state_local_t *st = (swiglu_gate_up_core_dot_task_state_local_t *) data;

  core_dot_chunk_fp16_local(st->c, st->a, st->b, st->s, st->n_row_tiles, st->n_col_tiles, st->n_dot_tiles);
  worker_pool_synctoken_jobdone(&st->sync_ctx);
}

static inline void submit_core_dot_chunk_fp16_async_local(swiglu_gate_up_core_dot_task_state_local_t *state,
                                                          worker_pool_job_t *job,
                                                          __fp16 *output,
                                                          const __fp16 *activation,
                                                          const __fp16 *weight,
                                                          const __fp16 *scales,
                                                          int n_row_tiles,
                                                          int n_col_tiles,
                                                          int n_dot_tiles) {
  state->c = output;
  state->a = activation;
  state->b = weight;
  state->s = scales;
  state->n_row_tiles = n_row_tiles;
  state->n_col_tiles = n_col_tiles;
  state->n_dot_tiles = n_dot_tiles;

  job->dptr = state;
  job->fptr = &swiglu_gate_up_core_dot_hmx_worker_local;

  worker_pool_synctoken_init(&state->sync_ctx, 1);
  worker_pool_submit(hmx_worker_pool_ctx, *job);
}

static int swiglu_gate_up_qk_stage0_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  dma_desc_1d_t dma_desc __attribute__((aligned(64)));

  for (size_t mr = 0; mr < (size_t) ctx->m; mr += ctx->m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) ctx->m - mr, ctx->m_chunk_n_rows);
    const float *activation_chunk = ctx->activation + mr * ctx->k;

    transfer_activation_chunk_fp32_to_fp16_local(ctx->vtcm_activation, activation_chunk, (int) n_rows, ctx->k, ctx->k);

    for (size_t nc = 0; nc < (size_t) ctx->n; nc += ctx->n_chunk_n_cols) {
      const size_t n_cols = smin((size_t) ctx->n - nc, ctx->n_chunk_n_cols);
      const size_t chunk_ne = n_cols * ctx->k;
      const size_t qweight_chunk_size = chunk_ne / QK_K * ctx->super_block_size;
      const size_t weight_offset = (nc * ctx->k / QK_K) * ctx->super_block_size;

      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight + weight_offset, qweight_chunk_size);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(ctx->vtcm_weight, NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);
      core_dot_chunk_fp16_local(ctx->vtcm_gate_out, ctx->vtcm_activation, ctx->vtcm_weight, ctx->vtcm_scales,
                                (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS),
                                (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS),
                                ctx->k / HMX_FP16_TILE_N_COLS);

      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->up_weight + weight_offset, qweight_chunk_size);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(ctx->vtcm_weight, NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);
      core_dot_chunk_fp16_local(ctx->vtcm_up_out, ctx->vtcm_activation, ctx->vtcm_weight, ctx->vtcm_scales,
                                (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS),
                                (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS),
                                ctx->k / HMX_FP16_TILE_N_COLS);

      fuse_gate_up_chunk_fp16_to_fp32_local(ctx->dst + mr * ctx->n + nc,
                                            ctx->vtcm_gate_out,
                                            ctx->vtcm_up_out,
                                            (int) n_rows,
                                            (int) n_cols,
                                            ctx->n,
                                            ctx->silu_lut,
                                            ctx->silu_lut_size,
                                            ctx->silu_lut_clamp,
                                            ctx->use_silu_lut);
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_stage1_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  dma_desc_1d_t dma_desc __attribute__((aligned(64)));

  for (size_t mr = 0; mr < (size_t) ctx->m; mr += ctx->m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) ctx->m - mr, ctx->m_chunk_n_rows);
    const float *activation_chunk = ctx->activation + mr * ctx->k;

    transfer_activation_chunk_fp32_to_fp16_local(ctx->vtcm_activation, activation_chunk, (int) n_rows, ctx->k, ctx->k);

    bool gate_prefetched = false;
    if ((size_t) ctx->n > 0) {
      const size_t n_cols_first = smin((size_t) ctx->n, ctx->n_chunk_n_cols);
      const size_t chunk_ne_first = n_cols_first * ctx->k;
      const size_t qweight_chunk_size_first = chunk_ne_first / QK_K * ctx->super_block_size;
      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight, qweight_chunk_size_first);
      gate_prefetched = true;
    }

    for (size_t nc = 0; nc < (size_t) ctx->n; nc += ctx->n_chunk_n_cols) {
      const size_t n_cols = smin((size_t) ctx->n - nc, ctx->n_chunk_n_cols);
      const size_t chunk_ne = n_cols * ctx->k;
      const size_t qweight_chunk_size = chunk_ne / QK_K * ctx->super_block_size;
      const size_t weight_offset = (nc * ctx->k / QK_K) * ctx->super_block_size;
      const int n_row_tiles = (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS);

      assert(gate_prefetched);

      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(ctx->vtcm_weight, NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);

      // overlap gate HMX with current up DMA.
      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->up_weight + weight_offset, qweight_chunk_size);
      core_dot_chunk_fp16_local(ctx->vtcm_gate_out, ctx->vtcm_activation, ctx->vtcm_weight, ctx->vtcm_scales,
                                n_row_tiles, n_col_tiles, ctx->k / HMX_FP16_TILE_N_COLS);

      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(ctx->vtcm_weight, NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);

      // overlap up HMX with next gate DMA.
      const size_t nc_next = nc + ctx->n_chunk_n_cols;
      gate_prefetched = false;
      if (nc_next < (size_t) ctx->n) {
        const size_t n_cols_next = smin((size_t) ctx->n - nc_next, ctx->n_chunk_n_cols);
        const size_t chunk_ne_next = n_cols_next * ctx->k;
        const size_t qweight_chunk_size_next = chunk_ne_next / QK_K * ctx->super_block_size;
        const size_t weight_offset_next = (nc_next * ctx->k / QK_K) * ctx->super_block_size;

        dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight + weight_offset_next,
                                      qweight_chunk_size_next);
        gate_prefetched = true;
      }

      core_dot_chunk_fp16_local(ctx->vtcm_up_out, ctx->vtcm_activation, ctx->vtcm_weight, ctx->vtcm_scales,
                                n_row_tiles, n_col_tiles, ctx->k / HMX_FP16_TILE_N_COLS);

      fuse_gate_up_chunk_fp16_to_fp32_local(ctx->dst + mr * ctx->n + nc,
                                            ctx->vtcm_gate_out,
                                            ctx->vtcm_up_out,
                                            (int) n_rows,
                                            (int) n_cols,
                                            ctx->n,
                                            ctx->silu_lut,
                                            ctx->silu_lut_size,
                                            ctx->silu_lut_clamp,
                                            ctx->use_silu_lut);
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_stage2_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  dma_desc_1d_t dma_desc __attribute__((aligned(64)));
  static swiglu_gate_up_core_dot_task_state_local_t hmx_task_state;
  static worker_pool_job_t hmx_task_job;

  if (!ctx->vtcm_weight_aux) {
    return -1;
  }

  for (size_t mr = 0; mr < (size_t) ctx->m; mr += ctx->m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) ctx->m - mr, ctx->m_chunk_n_rows);
    const float *activation_chunk = ctx->activation + mr * ctx->k;

    transfer_activation_chunk_fp32_to_fp16_local(ctx->vtcm_activation, activation_chunk, (int) n_rows, ctx->k, ctx->k);

    bool gate_prefetched = false;
    if ((size_t) ctx->n > 0) {
      const size_t n_cols_first = smin((size_t) ctx->n, ctx->n_chunk_n_cols);
      const size_t chunk_ne_first = n_cols_first * ctx->k;
      const size_t qweight_chunk_size_first = chunk_ne_first / QK_K * ctx->super_block_size;
      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight, qweight_chunk_size_first);
      gate_prefetched = true;
    }

    for (size_t nc = 0; nc < (size_t) ctx->n; nc += ctx->n_chunk_n_cols) {
      const size_t n_cols = smin((size_t) ctx->n - nc, ctx->n_chunk_n_cols);
      const size_t chunk_ne = n_cols * ctx->k;
      const size_t qweight_chunk_size = chunk_ne / QK_K * ctx->super_block_size;
      const size_t weight_offset = (nc * ctx->k / QK_K) * ctx->super_block_size;
      const int n_row_tiles = (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
      const int n_dot_tiles = ctx->k / HMX_FP16_TILE_N_COLS;
      const size_t nc_next = nc + ctx->n_chunk_n_cols;

      assert(gate_prefetched);

      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(ctx->vtcm_weight, NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);

      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             ctx->vtcm_gate_out, ctx->vtcm_activation, ctx->vtcm_weight, ctx->vtcm_scales,
                                             n_row_tiles, n_col_tiles, n_dot_tiles);

      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->up_weight + weight_offset, qweight_chunk_size);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(ctx->vtcm_weight_aux, NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);

      gate_prefetched = false;
      if (nc_next < (size_t) ctx->n) {
        const size_t n_cols_next = smin((size_t) ctx->n - nc_next, ctx->n_chunk_n_cols);
        const size_t chunk_ne_next = n_cols_next * ctx->k;
        const size_t qweight_chunk_size_next = chunk_ne_next / QK_K * ctx->super_block_size;
        const size_t weight_offset_next = (nc_next * ctx->k / QK_K) * ctx->super_block_size;
        dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight + weight_offset_next,
                                      qweight_chunk_size_next);
        gate_prefetched = true;
      }

      worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             ctx->vtcm_up_out, ctx->vtcm_activation, ctx->vtcm_weight_aux, ctx->vtcm_scales,
                                             n_row_tiles, n_col_tiles, n_dot_tiles);

      worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

      fuse_gate_up_chunk_fp16_to_fp32_local(ctx->dst + mr * ctx->n + nc,
                                            ctx->vtcm_gate_out,
                                            ctx->vtcm_up_out,
                                            (int) n_rows,
                                            (int) n_cols,
                                            ctx->n,
                                            ctx->silu_lut,
                                            ctx->silu_lut_size,
                                            ctx->silu_lut_clamp,
                                            ctx->use_silu_lut);
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_stage3_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  dma_desc_1d_t dma_desc __attribute__((aligned(64)));
  static swiglu_gate_up_core_dot_task_state_local_t hmx_task_state;
  static worker_pool_job_t hmx_task_job;

  if (!ctx->vtcm_weight_aux || !ctx->vtcm_gate_out_aux || !ctx->vtcm_up_out_aux) {
    return -1;
  }

  for (size_t mr = 0; mr < (size_t) ctx->m; mr += ctx->m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) ctx->m - mr, ctx->m_chunk_n_rows);
    const float *activation_chunk = ctx->activation + mr * ctx->k;
    const __fp16 *weight_bufs[2] = { ctx->vtcm_weight, ctx->vtcm_weight_aux };
    __fp16 *gate_out_bufs[2] = { ctx->vtcm_gate_out, ctx->vtcm_gate_out_aux };
    __fp16 *up_out_bufs[2] = { ctx->vtcm_up_out, ctx->vtcm_up_out_aux };

    transfer_activation_chunk_fp32_to_fp16_local(ctx->vtcm_activation, activation_chunk, (int) n_rows, ctx->k, ctx->k);

    bool gate_prefetched = false;
    if ((size_t) ctx->n > 0) {
      const size_t n_cols_first = smin((size_t) ctx->n, ctx->n_chunk_n_cols);
      const size_t chunk_ne_first = n_cols_first * ctx->k;
      const size_t qweight_chunk_size_first = chunk_ne_first / QK_K * ctx->super_block_size;
      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight, qweight_chunk_size_first);
      gate_prefetched = true;
    }

    bool prev_chunk_ready = false;
    size_t prev_nc = 0;
    size_t prev_n_cols = 0;
    int prev_slot = 0;

    for (size_t nc = 0, chunk_idx = 0; nc < (size_t) ctx->n; nc += ctx->n_chunk_n_cols, ++chunk_idx) {
      const int slot = (int) (chunk_idx & 1);
      const size_t n_cols = smin((size_t) ctx->n - nc, ctx->n_chunk_n_cols);
      const size_t chunk_ne = n_cols * ctx->k;
      const size_t qweight_chunk_size = chunk_ne / QK_K * ctx->super_block_size;
      const size_t weight_offset = (nc * ctx->k / QK_K) * ctx->super_block_size;
      const int n_row_tiles = (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
      const int n_dot_tiles = ctx->k / HMX_FP16_TILE_N_COLS;
      const size_t nc_next = nc + ctx->n_chunk_n_cols;

      assert(gate_prefetched);

      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx((__fp16 *) weight_bufs[0], NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);

      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             gate_out_bufs[slot], ctx->vtcm_activation, weight_bufs[0], ctx->vtcm_scales,
                                             n_row_tiles, n_col_tiles, n_dot_tiles);

      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->up_weight + weight_offset, qweight_chunk_size);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx((__fp16 *) weight_bufs[1], NULL, (int) chunk_ne, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);

      gate_prefetched = false;
      if (nc_next < (size_t) ctx->n) {
        const size_t n_cols_next = smin((size_t) ctx->n - nc_next, ctx->n_chunk_n_cols);
        const size_t chunk_ne_next = n_cols_next * ctx->k;
        const size_t qweight_chunk_size_next = chunk_ne_next / QK_K * ctx->super_block_size;
        const size_t weight_offset_next = (nc_next * ctx->k / QK_K) * ctx->super_block_size;
        dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight + weight_offset_next,
                                      qweight_chunk_size_next);
        gate_prefetched = true;
      }

      worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             up_out_bufs[slot], ctx->vtcm_activation, weight_bufs[1], ctx->vtcm_scales,
                                             n_row_tiles, n_col_tiles, n_dot_tiles);

      if (prev_chunk_ready) {
        fuse_gate_up_chunk_fp16_to_fp32_local(ctx->dst + mr * ctx->n + prev_nc,
                                              gate_out_bufs[prev_slot],
                                              up_out_bufs[prev_slot],
                                              (int) n_rows,
                                              (int) prev_n_cols,
                                              ctx->n,
                                              ctx->silu_lut,
                                              ctx->silu_lut_size,
                                              ctx->silu_lut_clamp,
                                              ctx->use_silu_lut);
      }

      worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

      prev_chunk_ready = true;
      prev_nc = nc;
      prev_n_cols = n_cols;
      prev_slot = slot;
    }

    if (prev_chunk_ready) {
      fuse_gate_up_chunk_fp16_to_fp32_local(ctx->dst + mr * ctx->n + prev_nc,
                                            gate_out_bufs[prev_slot],
                                            up_out_bufs[prev_slot],
                                            (int) n_rows,
                                            (int) prev_n_cols,
                                            ctx->n,
                                            ctx->silu_lut,
                                            ctx->silu_lut_size,
                                            ctx->silu_lut_clamp,
                                            ctx->use_silu_lut);
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_stage4_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  swiglu_gate_up_qk_stage_ctx_local_t tuned = *ctx;

  choose_chunk_shape_high_m_pipeline_local(ctx->m,
                                           ctx->k,
                                           ctx->n,
                                           get_weight_area_size_local(),
                                           get_activation_area_size_local(),
                                           get_output_area_size_local(),
                                           get_qweight_area_size_local(),
                                           ctx->super_block_size,
                                           &tuned.m_chunk_n_rows,
                                           &tuned.n_chunk_n_cols);

  return swiglu_gate_up_qk_stage3_local(&tuned);
}

static int swiglu_gate_up_qk_pipeline_gate_pass_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  static dma_desc_1d_t dma_desc __attribute__((aligned(64)));
  static swiglu_gate_up_core_dot_task_state_local_t hmx_task_state;
  static worker_pool_job_t hmx_task_job;

  const int n_chunk_cnt = (int) ceil_div((size_t) ctx->n, ctx->n_chunk_n_cols);
  const int n_rows = ctx->m;
  __fp16 *vtcm_weight_bufs[2] = { ctx->vtcm_weight, ctx->vtcm_weight_aux };
  __fp16 *vtcm_output_bufs[2] = { ctx->vtcm_gate_out, ctx->vtcm_gate_out_aux };

  if (!ctx->vtcm_weight_aux || !ctx->vtcm_gate_out_aux || n_chunk_cnt <= 0) {
    return -1;
  }

  // Gate pass as a 4-stage pipeline, modeled after mat_mul.c:
  //   A: DMA load gate qweight chunk
  //   B: dequantize qweight chunk into fp16 weight tiles
  //   C: HMX matmul -> fp16 gate tiles
  //   D: SiLU(gate) -> fp32 dst
  //
  // Buffering scheme:
  //   A --> B: vtcm_qweight, 1 buffer
  //   B --> C: vtcm_weight / vtcm_weight_aux, 2 buffers
  //   C --> D: vtcm_gate_out / vtcm_gate_out_aux, 2 buffers

  // prologue: A0
  const size_t n_cols_a0 = smin((size_t) ctx->n, ctx->n_chunk_n_cols);
  const size_t chunk_ne_a0 = n_cols_a0 * ctx->k;
  const size_t chunk_size_a0 = chunk_ne_a0 / QK_K * ctx->super_block_size;
  dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->gate_weight, chunk_size_a0);

  // prologue: B0, A1, C0, B1
  dma_wait_for_idle();
  dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[0], NULL, (int) chunk_ne_a0, ctx->k,
                                                    ctx->weight_type, ctx->vtcm_qweight);

  if (1 < n_chunk_cnt) {
    const size_t n_cols_a1 = smin((size_t) ctx->n - ctx->n_chunk_n_cols, ctx->n_chunk_n_cols);
    const size_t chunk_ne_a1 = n_cols_a1 * ctx->k;
    const size_t chunk_size_a1 = chunk_ne_a1 / QK_K * ctx->super_block_size;
    const uint8_t *qweight_chunk_a1 = ctx->gate_weight + (ctx->n_chunk_n_cols * ctx->k / QK_K) * ctx->super_block_size;
    dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, qweight_chunk_a1, chunk_size_a1);
  }

  submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                         vtcm_output_bufs[0], ctx->vtcm_activation, vtcm_weight_bufs[0], ctx->vtcm_scales,
                                         (int) ceil_div((size_t) n_rows, HMX_FP16_TILE_N_ROWS),
                                         (int) ceil_div(n_cols_a0, HMX_FP16_TILE_N_COLS),
                                         ctx->k / HMX_FP16_TILE_N_COLS);

  if (1 < n_chunk_cnt) {
    const size_t n_cols_b1 = smin((size_t) ctx->n - ctx->n_chunk_n_cols, ctx->n_chunk_n_cols);
    const size_t chunk_ne_b1 = n_cols_b1 * ctx->k;
    dma_wait_for_idle();
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[1], NULL, (int) chunk_ne_b1, ctx->k,
                                                      ctx->weight_type, ctx->vtcm_qweight);
  }

  for (int i = 0; i < n_chunk_cnt; ++i) {
    const size_t nc = (size_t) i * ctx->n_chunk_n_cols;
    const size_t nc_p1 = nc + ctx->n_chunk_n_cols;
    const size_t nc_p2 = nc + 2 * ctx->n_chunk_n_cols;
    const size_t n_cols = smin((size_t) ctx->n - nc, ctx->n_chunk_n_cols);
    const size_t n_cols_p1 = smin((size_t) ctx->n - nc_p1, ctx->n_chunk_n_cols);
    const size_t n_cols_p2 = smin((size_t) ctx->n - nc_p2, ctx->n_chunk_n_cols);

    // issue A_{i+2}
    if (i + 2 < n_chunk_cnt) {
      const size_t chunk_ne_p2 = n_cols_p2 * ctx->k;
      const size_t chunk_size_p2 = chunk_ne_p2 / QK_K * ctx->super_block_size;
      const uint8_t *qweight_chunk_p2 = ctx->gate_weight + (nc_p2 * ctx->k / QK_K) * ctx->super_block_size;
      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, qweight_chunk_p2, chunk_size_p2);
    }

    // wait for C_i
    worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

    // B_{i+1} should already be ready here, so we can immediately issue C_{i+1}.
    if (i + 1 < n_chunk_cnt) {
      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             vtcm_output_bufs[(i + 1) % 2],
                                             ctx->vtcm_activation,
                                             vtcm_weight_bufs[(i + 1) % 2],
                                             ctx->vtcm_scales,
                                             (int) ceil_div((size_t) n_rows, HMX_FP16_TILE_N_ROWS),
                                             (int) ceil_div(n_cols_p1, HMX_FP16_TILE_N_COLS),
                                             ctx->k / HMX_FP16_TILE_N_COLS);
    }

    // compute D_i
    silu_gate_chunk_fp16_to_fp32_local(ctx->dst + nc,
                                       vtcm_output_bufs[i % 2],
                                       n_rows,
                                       (int) n_cols,
                                       ctx->n,
                                       ctx->silu_lut,
                                       ctx->silu_lut_size,
                                       ctx->silu_lut_clamp,
                                       ctx->use_silu_lut);

    // wait for A_{i+2}, then compute B_{i+2}
    if (i + 2 < n_chunk_cnt) {
      const size_t chunk_ne_p2 = n_cols_p2 * ctx->k;
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[(i + 2) % 2], NULL, (int) chunk_ne_p2, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_pipeline_up_pass_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  static dma_desc_1d_t dma_desc __attribute__((aligned(64)));
  static swiglu_gate_up_core_dot_task_state_local_t hmx_task_state;
  static worker_pool_job_t hmx_task_job;

  const int n_chunk_cnt = (int) ceil_div((size_t) ctx->n, ctx->n_chunk_n_cols);
  const int n_rows = ctx->m;
  __fp16 *vtcm_weight_bufs[2] = { ctx->vtcm_weight, ctx->vtcm_weight_aux };
  __fp16 *vtcm_output_bufs[2] = { ctx->vtcm_up_out, ctx->vtcm_up_out_aux };

  if (!ctx->vtcm_weight_aux || !ctx->vtcm_up_out_aux || n_chunk_cnt <= 0) {
    return -1;
  }

  // Up pass uses the same 4-stage structure as mat_mul.c:
  //   A: DMA load up qweight chunk
  //   B: dequantize qweight chunk into fp16 weight tiles
  //   C: HMX matmul -> fp16 up tiles
  //   D: dst *= up
  //
  // This is intentionally split from the gate pass so that stage D stays thin
  // and can be overlapped by the same A/B/C cadence as the standalone matmul.

  // prologue: A0
  const size_t n_cols_a0 = smin((size_t) ctx->n, ctx->n_chunk_n_cols);
  const size_t chunk_ne_a0 = n_cols_a0 * ctx->k;
  const size_t chunk_size_a0 = chunk_ne_a0 / QK_K * ctx->super_block_size;
  dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, ctx->up_weight, chunk_size_a0);

  // prologue: B0, A1, C0, B1
  dma_wait_for_idle();
  dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[0], NULL, (int) chunk_ne_a0, ctx->k,
                                                    ctx->weight_type, ctx->vtcm_qweight);

  if (1 < n_chunk_cnt) {
    const size_t n_cols_a1 = smin((size_t) ctx->n - ctx->n_chunk_n_cols, ctx->n_chunk_n_cols);
    const size_t chunk_ne_a1 = n_cols_a1 * ctx->k;
    const size_t chunk_size_a1 = chunk_ne_a1 / QK_K * ctx->super_block_size;
    const uint8_t *qweight_chunk_a1 = ctx->up_weight + (ctx->n_chunk_n_cols * ctx->k / QK_K) * ctx->super_block_size;
    dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, qweight_chunk_a1, chunk_size_a1);
  }

  submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                         vtcm_output_bufs[0], ctx->vtcm_activation, vtcm_weight_bufs[0], ctx->vtcm_scales,
                                         (int) ceil_div((size_t) n_rows, HMX_FP16_TILE_N_ROWS),
                                         (int) ceil_div(n_cols_a0, HMX_FP16_TILE_N_COLS),
                                         ctx->k / HMX_FP16_TILE_N_COLS);

  if (1 < n_chunk_cnt) {
    const size_t n_cols_b1 = smin((size_t) ctx->n - ctx->n_chunk_n_cols, ctx->n_chunk_n_cols);
    const size_t chunk_ne_b1 = n_cols_b1 * ctx->k;
    dma_wait_for_idle();
    dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[1], NULL, (int) chunk_ne_b1, ctx->k,
                                                      ctx->weight_type, ctx->vtcm_qweight);
  }

  for (int i = 0; i < n_chunk_cnt; ++i) {
    const size_t nc = (size_t) i * ctx->n_chunk_n_cols;
    const size_t nc_p1 = nc + ctx->n_chunk_n_cols;
    const size_t nc_p2 = nc + 2 * ctx->n_chunk_n_cols;
    const size_t n_cols = smin((size_t) ctx->n - nc, ctx->n_chunk_n_cols);
    const size_t n_cols_p1 = smin((size_t) ctx->n - nc_p1, ctx->n_chunk_n_cols);
    const size_t n_cols_p2 = smin((size_t) ctx->n - nc_p2, ctx->n_chunk_n_cols);

    // issue A_{i+2}
    if (i + 2 < n_chunk_cnt) {
      const size_t chunk_ne_p2 = n_cols_p2 * ctx->k;
      const size_t chunk_size_p2 = chunk_ne_p2 / QK_K * ctx->super_block_size;
      const uint8_t *qweight_chunk_p2 = ctx->up_weight + (nc_p2 * ctx->k / QK_K) * ctx->super_block_size;
      dma_issue_load_from_ddr_local(&dma_desc, ctx->vtcm_qweight, qweight_chunk_p2, chunk_size_p2);
    }

    // wait for C_i
    worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

    // B_{i+1} should already be ready here, so we can immediately issue C_{i+1}.
    if (i + 1 < n_chunk_cnt) {
      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             vtcm_output_bufs[(i + 1) % 2],
                                             ctx->vtcm_activation,
                                             vtcm_weight_bufs[(i + 1) % 2],
                                             ctx->vtcm_scales,
                                             (int) ceil_div((size_t) n_rows, HMX_FP16_TILE_N_ROWS),
                                             (int) ceil_div(n_cols_p1, HMX_FP16_TILE_N_COLS),
                                             ctx->k / HMX_FP16_TILE_N_COLS);
    }

    // compute D_i
    mul_dst_fp32_by_up_chunk_local(ctx->dst + nc,
                                   vtcm_output_bufs[i % 2],
                                   n_rows,
                                   (int) n_cols,
                                   ctx->n);

    // wait for A_{i+2}, then compute B_{i+2}
    if (i + 2 < n_chunk_cnt) {
      const size_t chunk_ne_p2 = n_cols_p2 * ctx->k;
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(vtcm_weight_bufs[(i + 2) % 2], NULL, (int) chunk_ne_p2, ctx->k,
                                                        ctx->weight_type, ctx->vtcm_qweight);
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_stage5_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  swiglu_gate_up_qk_stage_ctx_local_t tuned = *ctx;

  choose_chunk_shape_high_m_pipeline_local(ctx->m,
                                           ctx->k,
                                           ctx->n,
                                           get_weight_area_size_local(),
                                           get_activation_area_size_local(),
                                           get_output_area_size_local(),
                                           get_qweight_area_size_local(),
                                           ctx->super_block_size,
                                           &tuned.m_chunk_n_rows,
                                           &tuned.n_chunk_n_cols);

  // Stage 5 is a "true pipeline" variant for larger m:
  //   pass 1: gate matmul in a 4-stage pipeline, epilogue = SiLU(store to dst)
  //   pass 2: up   matmul in a 4-stage pipeline, epilogue = mul(dst, up)
  //
  // Compared with the earlier fused stages, this matches mat_mul.c more closely:
  // each pass has explicit prologue / steady-state loop / epilogue behavior and
  // keeps D as a thin post-matmul stage.
  for (size_t mr = 0; mr < (size_t) tuned.m; mr += tuned.m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) tuned.m - mr, tuned.m_chunk_n_rows);
    const float *activation_chunk = tuned.activation + mr * tuned.k;

    transfer_activation_chunk_fp32_to_fp16_local(tuned.vtcm_activation, activation_chunk, (int) n_rows, tuned.k, tuned.k);

    swiglu_gate_up_qk_stage_ctx_local_t chunk_ctx = tuned;
    chunk_ctx.dst = tuned.dst + mr * tuned.n;
    chunk_ctx.activation = activation_chunk;
    chunk_ctx.m = (int) n_rows;

    int ret = swiglu_gate_up_qk_pipeline_gate_pass_local(&chunk_ctx);
    if (ret != 0) {
      return ret;
    }

    ret = swiglu_gate_up_qk_pipeline_up_pass_local(&chunk_ctx);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

static int swiglu_gate_up_qk_stage6_local(const swiglu_gate_up_qk_stage_ctx_local_t *ctx) {
  static dma_desc_1d_t dma_desc __attribute__((aligned(64)));
  static swiglu_gate_up_core_dot_task_state_local_t hmx_task_state;
  static worker_pool_job_t hmx_task_job;

  swiglu_gate_up_qk_stage_ctx_local_t tuned = *ctx;

  choose_chunk_shape_high_m_pipeline_local(ctx->m,
                                           ctx->k,
                                           ctx->n,
                                           get_weight_area_size_local(),
                                           get_activation_area_size_local(),
                                           get_output_area_size_local(),
                                           get_qweight_area_size_local(),
                                           ctx->super_block_size,
                                           &tuned.m_chunk_n_rows,
                                           &tuned.n_chunk_n_cols);

  if (!tuned.vtcm_weight_aux) {
    return -1;
  }

  // Stage 6 is the joint single-function pipeline:
  //
  //   gate: A_g -> B_g -> C_g -> D_g
  //   up  : A_u -> B_u -> C_u -> D_u
  //
  // We keep one HMX stream and one DMA stream, then phase-shift the two 4-stage
  // sub-pipelines so that:
  //   - C_g(i) overlaps with A_u(i) / B_u(i) and D_u(i-1)
  //   - C_u(i) overlaps with A_g(i+1) / B_g(i+1) and D_g(i)
  //
  // Buffering scheme:
  //   A --> B: vtcm_qweight, 1 buffer
  //   B --> C: gate_weight / up_weight, 2 dedicated buffers
  //   C --> D: gate_out / up_out, 2 dedicated buffers
  //
  // Compared with stage5, this avoids pass-level serialization between gate and up.

  for (size_t mr = 0; mr < (size_t) tuned.m; mr += tuned.m_chunk_n_rows) {
    const size_t n_rows = smin((size_t) tuned.m - mr, tuned.m_chunk_n_rows);
    const float *activation_chunk = tuned.activation + mr * tuned.k;

    transfer_activation_chunk_fp32_to_fp16_local(tuned.vtcm_activation, activation_chunk, (int) n_rows, tuned.k, tuned.k);

    swiglu_gate_up_qk_stage_ctx_local_t chunk_ctx = tuned;
    chunk_ctx.dst = tuned.dst + mr * tuned.n;
    chunk_ctx.activation = activation_chunk;
    chunk_ctx.m = (int) n_rows;

    const int n_chunk_cnt = (int) ceil_div((size_t) chunk_ctx.n, chunk_ctx.n_chunk_n_cols);
    if (n_chunk_cnt <= 0) {
      continue;
    }

    __fp16 *gate_weight_buf = chunk_ctx.vtcm_weight;
    __fp16 *up_weight_buf   = chunk_ctx.vtcm_weight_aux;
    __fp16 *gate_out_buf    = chunk_ctx.vtcm_gate_out;
    __fp16 *up_out_buf      = chunk_ctx.vtcm_up_out;

    // prologue: A_g0, B_g0
    {
      const size_t nc0 = 0;
      const size_t n_cols0 = smin((size_t) chunk_ctx.n - nc0, chunk_ctx.n_chunk_n_cols);
      const size_t chunk_ne0 = n_cols0 * chunk_ctx.k;
      const size_t chunk_size0 = chunk_ne0 / QK_K * chunk_ctx.super_block_size;
      dma_issue_load_from_ddr_local(&dma_desc, chunk_ctx.vtcm_qweight, chunk_ctx.gate_weight, chunk_size0);
      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(gate_weight_buf, NULL, (int) chunk_ne0, chunk_ctx.k,
                                                        chunk_ctx.weight_type, chunk_ctx.vtcm_qweight);
    }

    for (int i = 0; i < n_chunk_cnt; ++i) {
      const size_t nc = (size_t) i * chunk_ctx.n_chunk_n_cols;
      const size_t n_cols = smin((size_t) chunk_ctx.n - nc, chunk_ctx.n_chunk_n_cols);
      const size_t chunk_ne = n_cols * chunk_ctx.k;
      const size_t up_offset = (nc * chunk_ctx.k / QK_K) * chunk_ctx.super_block_size;
      const size_t up_chunk_size = chunk_ne / QK_K * chunk_ctx.super_block_size;
      const int n_row_tiles = (int) ceil_div(n_rows, HMX_FP16_TILE_N_ROWS);
      const int n_col_tiles = (int) ceil_div(n_cols, HMX_FP16_TILE_N_COLS);
      const int n_dot_tiles = chunk_ctx.k / HMX_FP16_TILE_N_COLS;

      // C_g(i)
      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             gate_out_buf, chunk_ctx.vtcm_activation, gate_weight_buf, chunk_ctx.vtcm_scales,
                                             n_row_tiles, n_col_tiles, n_dot_tiles);

      // During C_g(i): A_u(i), D_u(i-1), B_u(i)
      dma_issue_load_from_ddr_local(&dma_desc, chunk_ctx.vtcm_qweight, chunk_ctx.up_weight + up_offset, up_chunk_size);

      if (i > 0) {
        const size_t prev_nc = (size_t) (i - 1) * chunk_ctx.n_chunk_n_cols;
        const size_t prev_n_cols = smin((size_t) chunk_ctx.n - prev_nc, chunk_ctx.n_chunk_n_cols);
        mul_dst_fp32_by_up_chunk_local(chunk_ctx.dst + prev_nc,
                                       up_out_buf,
                                       (int) n_rows,
                                       (int) prev_n_cols,
                                       chunk_ctx.n);
      }

      dma_wait_for_idle();
      dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(up_weight_buf, NULL, (int) chunk_ne, chunk_ctx.k,
                                                        chunk_ctx.weight_type, chunk_ctx.vtcm_qweight);

      // wait for C_g(i)
      worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);

      // C_u(i)
      submit_core_dot_chunk_fp16_async_local(&hmx_task_state, &hmx_task_job,
                                             up_out_buf, chunk_ctx.vtcm_activation, up_weight_buf, chunk_ctx.vtcm_scales,
                                             n_row_tiles, n_col_tiles, n_dot_tiles);

      // During C_u(i): A_g(i+1), D_g(i), B_g(i+1)
      if (i + 1 < n_chunk_cnt) {
        const size_t next_nc = (size_t) (i + 1) * chunk_ctx.n_chunk_n_cols;
        const size_t next_n_cols = smin((size_t) chunk_ctx.n - next_nc, chunk_ctx.n_chunk_n_cols);
        const size_t next_chunk_ne = next_n_cols * chunk_ctx.k;
        const size_t next_gate_offset = (next_nc * chunk_ctx.k / QK_K) * chunk_ctx.super_block_size;
        const size_t next_gate_chunk_size = next_chunk_ne / QK_K * chunk_ctx.super_block_size;

        dma_issue_load_from_ddr_local(&dma_desc, chunk_ctx.vtcm_qweight,
                                      chunk_ctx.gate_weight + next_gate_offset, next_gate_chunk_size);

        silu_gate_chunk_fp16_to_fp32_local(chunk_ctx.dst + nc,
                                           gate_out_buf,
                                           (int) n_rows,
                                           (int) n_cols,
                                           chunk_ctx.n,
                                           chunk_ctx.silu_lut,
                                           chunk_ctx.silu_lut_size,
                                           chunk_ctx.silu_lut_clamp,
                                           chunk_ctx.use_silu_lut);

        dma_wait_for_idle();
        dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(gate_weight_buf, NULL, (int) next_chunk_ne, chunk_ctx.k,
                                                          chunk_ctx.weight_type, chunk_ctx.vtcm_qweight);
      } else {
        silu_gate_chunk_fp16_to_fp32_local(chunk_ctx.dst + nc,
                                           gate_out_buf,
                                           (int) n_rows,
                                           (int) n_cols,
                                           chunk_ctx.n,
                                           chunk_ctx.silu_lut,
                                           chunk_ctx.silu_lut_size,
                                           chunk_ctx.silu_lut_clamp,
                                           chunk_ctx.use_silu_lut);
      }

      // wait for C_u(i)
      worker_pool_synctoken_wait(&hmx_task_state.sync_ctx);
    }

    // epilogue: D_u(last)
    {
      const size_t last_nc = (size_t) (n_chunk_cnt - 1) * chunk_ctx.n_chunk_n_cols;
      const size_t last_n_cols = smin((size_t) chunk_ctx.n - last_nc, chunk_ctx.n_chunk_n_cols);
      mul_dst_fp32_by_up_chunk_local(chunk_ctx.dst + last_nc,
                                     up_out_buf,
                                     (int) n_rows,
                                     (int) last_n_cols,
                                     chunk_ctx.n);
    }
  }

  return 0;
}

void dequantize_permuted_weight_chunk_qk_0_to_fp16_hvx(__fp16 *vtcm_dst, const void *src, int ne, int k,
                                                       enum ggml_type type, void *vtcm_scratch);
int hmx_mat_mul_permuted_w16a32(float *restrict dst, const float *restrict activation,
                                const __fp16 *restrict permuted_weight, int m, int k, int n);

int hmx_hvx_swiglu_gate_up_fused_w16a32(float *restrict dst,
                                        const float *restrict activation,
                                        const __fp16 *restrict gate_weight,
                                        const __fp16 *restrict up_weight,
                                        int m,
                                        int k,
                                        int n,
                                        int silu_lut_bits,
                                        float silu_lut_clamp,
                                        bool use_silu_lut) {
  if (!dst || !activation || !gate_weight || !up_weight || !m || !k || !n) {
    return -1;
  }
  if (k % 32 != 0 || n % 32 != 0) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(activation, VLEN) ||
      !is_aligned(gate_weight, VLEN) || !is_aligned(up_weight, VLEN)) {
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

  const size_t n_elems = (size_t) m * (size_t) n;
  float *gate_tmp = NULL;
  float *up_tmp   = NULL;
  if (posix_memalign((void **) &gate_tmp, VLEN, n_elems * sizeof(float)) != 0 ||
      posix_memalign((void **) &up_tmp, VLEN, n_elems * sizeof(float)) != 0) {
    free(gate_tmp);
    free(up_tmp);
    return -1;
  }

  int err = 0;
  err = hmx_mat_mul_permuted_w16a32(gate_tmp, activation, gate_weight, m, k, n);
  if (err == 0) {
    err = hmx_mat_mul_permuted_w16a32(up_tmp, activation, up_weight, m, k, n);
  }
  if (err == 0) {
    fuse_gate_up_fp32_local(dst, gate_tmp, up_tmp, n_elems, silu_lut, silu_lut_size, silu_lut_clamp, use_silu_lut);
  }

  free(gate_tmp);
  free(up_tmp);
  return err;
}

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

  const size_t weight_area_size = get_weight_area_size_local();
  const size_t activation_area_size = get_activation_area_size_local();
  const size_t output_area_size = get_output_area_size_local();
  const size_t qweight_area_size = get_qweight_area_size_local();

  uint8_t *vtcm_ptr        = (uint8_t *) vtcm_manager_get_vtcm_base();
  __fp16  *vtcm_weight     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size);
  __fp16  *vtcm_weight_aux = stage_uses_weight_aux_local() ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, weight_area_size) : NULL;
  __fp16  *vtcm_activation = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, activation_area_size);
  __fp16  *vtcm_gate_out   = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  __fp16  *vtcm_gate_out_aux = stage_uses_output_aux_local() ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size) : NULL;
  __fp16  *vtcm_up_out     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size);
  __fp16  *vtcm_up_out_aux = stage_uses_output_aux_local() ? (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, output_area_size) : NULL;
  void    *vtcm_qweight    = vtcm_seq_alloc(&vtcm_ptr, qweight_area_size);
  __fp16  *vtcm_scales     = (__fp16 *) vtcm_seq_alloc(&vtcm_ptr, 256);

  hmx_init_column_scales(vtcm_scales, Q6_V_vsplat_R(0x3c00));

  const size_t vec_dot_size       = k * sizeof(__fp16);
  const size_t m_chunk_max_n_rows = align_down(activation_area_size / vec_dot_size, HMX_FP16_TILE_N_ROWS);
  const size_t n_chunk_max_n_cols_by_weight = align_down(weight_area_size / vec_dot_size, HMX_FP16_TILE_N_COLS);
  const size_t qweight_bytes_per_col = ((size_t) k / QK_K) * super_block_size;
  const size_t n_chunk_max_n_cols_by_qweight = qweight_bytes_per_col == 0
                                                   ? 0
                                                   : align_down(qweight_area_size / qweight_bytes_per_col,
                                                                HMX_FP16_TILE_N_COLS);
  const size_t n_chunk_max_n_cols = smin(n_chunk_max_n_cols_by_weight, n_chunk_max_n_cols_by_qweight);

  size_t m_chunk_n_rows = 0;
  size_t n_chunk_n_cols = 0;
  find_chunk_size_local(m_chunk_max_n_rows, n_chunk_max_n_cols, output_area_size / sizeof(__fp16),
                        HMX_FP16_TILE_N_ROWS, HMX_FP16_TILE_N_COLS, &m_chunk_n_rows, &n_chunk_n_cols);

  if (m_chunk_n_rows == 0 || n_chunk_n_cols == 0) {
    return -1;
  }
  const swiglu_gate_up_qk_stage_ctx_local_t ctx = {
    .dst = dst,
    .activation = activation,
    .gate_weight = gate_weight,
    .up_weight = up_weight,
    .m = m,
    .k = k,
    .n = n,
    .weight_type = weight_type,
    .super_block_size = super_block_size,
    .vtcm_weight = vtcm_weight,
    .vtcm_weight_aux = vtcm_weight_aux,
    .vtcm_activation = vtcm_activation,
    .vtcm_gate_out = vtcm_gate_out,
    .vtcm_gate_out_aux = vtcm_gate_out_aux,
    .vtcm_up_out = vtcm_up_out,
    .vtcm_up_out_aux = vtcm_up_out_aux,
    .vtcm_qweight = vtcm_qweight,
    .vtcm_scales = vtcm_scales,
    .m_chunk_n_rows = m_chunk_n_rows,
    .n_chunk_n_cols = n_chunk_n_cols,
    .silu_lut = silu_lut,
    .silu_lut_size = silu_lut_size,
    .silu_lut_clamp = silu_lut_clamp,
    .use_silu_lut = use_silu_lut,
  };

  switch (SWIGLU_GATE_UP_ACTIVE_STAGE) {
    case 0:
      return swiglu_gate_up_qk_stage0_local(&ctx);
    case 1:
      return swiglu_gate_up_qk_stage1_local(&ctx);
    case 2:
      return swiglu_gate_up_qk_stage2_local(&ctx);
    case 3:
      return swiglu_gate_up_qk_stage3_local(&ctx);
    case 4:
      return swiglu_gate_up_qk_stage4_local(&ctx);
    case 5:
      return swiglu_gate_up_qk_stage5_local(&ctx);
    case 6:
      return swiglu_gate_up_qk_stage6_local(&ctx);
    default:
      return -1;
  }
}
