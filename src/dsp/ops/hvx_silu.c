#include <math.h>
#include <stdint.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"

// Standalone HVX SiLU kernels (DSP-side helpers).
// Not wired into build; intended to be called from op implementations.

static inline int32_t fp32_to_bits(float x) {
  union { float f; int32_t i; } u;
  u.f = x;
  return u.i;
}

static inline HVX_Vector hvx_silu_vec_f32(HVX_Vector v_x_sf) {
  const HVX_Vector v_zero   = Q6_V_vzero();
  const HVX_Vector v_one_sf = Q6_V_vsplat_R(0x3F800000);  // 1.0f
  const HVX_Vector v_log2e  = Q6_V_vsplat_R(0xBFB8AA3C);  // -1/ln(2) ~= -1.4426951

  // exp(-x) = exp2(x * (-1/ln2))
  HVX_Vector v_x_log2e_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_x_sf, v_log2e);
  HVX_Vector v_x_log2e_sf   = Q6_Vsf_equals_Vqf32(v_x_log2e_qf32);
  HVX_Vector v_exp_sf       = hvx_my_exp2_vsf(v_x_log2e_sf);

  // denom = 1 + exp(-x)
  HVX_Vector v_denom_qf32 = Q6_Vqf32_vadd_VsfVsf(v_exp_sf, v_one_sf);
  HVX_Vector v_denom_sf   = Q6_Vsf_equals_Vqf32(v_denom_qf32);

  // inv_denom
  HVX_Vector v_inv_qf32 = hvx_my_inv_vqf32_vsf(v_denom_sf);

  // silu = x * inv_denom
  HVX_Vector v_x_qf32   = Q6_Vqf32_vadd_VsfVsf(v_x_sf, v_zero);
  HVX_Vector v_out_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(v_x_qf32, v_inv_qf32);
  return Q6_Vsf_equals_Vqf32(v_out_qf32);
}

int hvx_silu_f32(float *restrict dst, const float *restrict src, int n) {
  if (!dst || !src || n <= 0) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  const int n_vecs   = n / 32;
  const int leftover = n & 31;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) dst;

  for (int i = 0; i < n_vecs; ++i) {
    const HVX_Vector v_x = *pv_in++;
    *pv_out++ = hvx_silu_vec_f32(v_x);
  }

  if (leftover > 0) {
    const float *tail = src + n_vecs * 32;
    float *tail_out   = dst + n_vecs * 32;
    for (int i = 0; i < leftover; ++i) {
      float x = tail[i];
      tail_out[i] = x / (1.0f + expf(-x));
    }
  }

  return 0;
}

int hvx_silu_lut_f32(float *restrict dst, const float *restrict src, int n,
                     const float *restrict lut, int lut_size, float clamp) {
  if (!dst || !src || !lut || n <= 0 || lut_size <= 0 || !(clamp > 0.0f)) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(src, VLEN)) {
    return -1;
  }

  const float step     = (2.0f * clamp) / (float) lut_size;
  const float inv_step = 1.0f / step;
  const float neg_clamp = -clamp;

  const HVX_Vector v_clamp_sf  = Q6_V_vsplat_R(fp32_to_bits(clamp));
  const HVX_Vector v_nclamp_sf = Q6_V_vsplat_R(fp32_to_bits(neg_clamp));
  const HVX_Vector v_inv_step  = Q6_V_vsplat_R(fp32_to_bits(inv_step));

  const int n_vecs   = n / 32;
  const int leftover = n & 31;

  const HVX_Vector *pv_in  = (const HVX_Vector *) src;
  HVX_Vector       *pv_out = (HVX_Vector *) dst;

  _Alignas(VLEN) int32_t idx[32];
  _Alignas(VLEN) float   tval[32];
  _Alignas(VLEN) float   out[32];

  for (int i = 0; i < n_vecs; ++i) {
    HVX_Vector v_x = *pv_in++;

    // clamp to [-clamp, clamp]
    HVX_VectorPred q_gt = Q6_Q_vcmp_gt_VsfVsf(v_x, v_clamp_sf);
    HVX_VectorPred q_lt = Q6_Q_vcmp_gt_VsfVsf(v_nclamp_sf, v_x);
    v_x = Q6_V_vmux_QVV(q_gt, v_clamp_sf, v_x);
    v_x = Q6_V_vmux_QVV(q_lt, v_nclamp_sf, v_x);

    // u = (x + clamp) * inv_step
    HVX_Vector v_x_plus_qf32 = Q6_Vqf32_vadd_VsfVsf(v_x, v_clamp_sf);
    HVX_Vector v_x_plus_sf   = Q6_Vsf_equals_Vqf32(v_x_plus_qf32);
    HVX_Vector v_u_qf32      = Q6_Vqf32_vmpy_VsfVsf(v_x_plus_sf, v_inv_step);
    HVX_Vector v_u_sf        = Q6_Vsf_equals_Vqf32(v_u_qf32);

    // i = floor(u)
    HVX_Vector v_i = qhmath_hvx_vw_truncate_vsf(v_u_sf);
    HVX_Vector v_i_sf = Q6_Vsf_equals_Vw(v_i);

    // t = u - i
    HVX_Vector v_t_qf32 = Q6_Vqf32_vsub_VsfVsf(v_u_sf, v_i_sf);
    HVX_Vector v_t_sf   = Q6_Vsf_equals_Vqf32(v_t_qf32);

    vmem(idx)  = v_i;
    vmem(tval) = v_t_sf;

    // NOTE: no HVX gather; do scalar LUT fetch for now.
    for (int j = 0; j < 32; ++j) {
      int ii = idx[j];
      if (ii < 0) {
        ii = 0;
      } else if (ii >= lut_size) {
        ii = lut_size - 1;
      }
      const float y0 = lut[ii];
      const float y1 = lut[ii + 1];
      out[j] = y0 + (y1 - y0) * tval[j];
    }

    *pv_out++ = vmem(out);
  }

  if (leftover > 0) {
    const float *tail = src + n_vecs * 32;
    float *tail_out   = dst + n_vecs * 32;
    for (int i = 0; i < leftover; ++i) {
      float x = tail[i];
      if (x <= -clamp) {
        tail_out[i] = lut[0];
        continue;
      }
      if (x >= clamp) {
        tail_out[i] = lut[lut_size];
        continue;
      }
      float u = (x + clamp) * inv_step;
      int   ii = (int) u;
      if (ii < 0) ii = 0;
      if (ii >= lut_size) ii = lut_size - 1;
      float t = u - (float) ii;
      float y0 = lut[ii];
      float y1 = lut[ii + 1];
      tail_out[i] = y0 + (y1 - y0) * t;
    }
  }

  return 0;
}

int hvx_silu_mul_f32(float *restrict dst, const float *restrict x, const float *restrict y, int n,
                     const float *restrict lut, int lut_size, float clamp, int use_lut) {
  if (!dst || !x || !y || n <= 0) {
    return -1;
  }
  if (!is_aligned(dst, VLEN) || !is_aligned(x, VLEN) || !is_aligned(y, VLEN)) {
    return -1;
  }

  const int n_vecs   = n / 32;
  const int leftover = n & 31;

  const HVX_Vector *pv_x = (const HVX_Vector *) x;
  const HVX_Vector *pv_y = (const HVX_Vector *) y;
  HVX_Vector       *pv_o = (HVX_Vector *) dst;

  if (!use_lut) {
    for (int i = 0; i < n_vecs; ++i) {
      const HVX_Vector v_x = *pv_x++;
      const HVX_Vector v_y = *pv_y++;
      const HVX_Vector v_silu = hvx_silu_vec_f32(v_x);
      HVX_Vector v_out_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_silu, v_y);
      *pv_o++ = Q6_Vsf_equals_Vqf32(v_out_qf32);
    }
  } else {
    // LUT path: reuse lut implementation for SiLU, then multiply.
    _Alignas(VLEN) float tmp_in[32];
    _Alignas(VLEN) float tmp_out[32];
    for (int i = 0; i < n_vecs; ++i) {
      HVX_Vector v_x = *pv_x++;
      HVX_Vector v_y = *pv_y++;

      vmem(tmp_in) = v_x;
      hvx_silu_lut_f32(tmp_out, tmp_in, 32, lut, lut_size, clamp);

      HVX_Vector v_silu = vmem(tmp_out);
      HVX_Vector v_out_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_silu, v_y);
      *pv_o++ = Q6_Vsf_equals_Vqf32(v_out_qf32);
    }
  }

  if (leftover > 0) {
    const float *xt = x + n_vecs * 32;
    const float *yt = y + n_vecs * 32;
    float *ot        = dst + n_vecs * 32;
    for (int i = 0; i < leftover; ++i) {
      float sx;
      if (use_lut && lut && lut_size > 0 && clamp > 0.0f) {
        float xx = xt[i];
        if (xx <= -clamp) {
          sx = lut[0];
        } else if (xx >= clamp) {
          sx = lut[lut_size];
        } else {
          float step     = (2.0f * clamp) / (float) lut_size;
          float inv_step = 1.0f / step;
          float u = (xx + clamp) * inv_step;
          int   ii = (int) u;
          if (ii < 0) ii = 0;
          if (ii >= lut_size) ii = lut_size - 1;
          float t = u - (float) ii;
          float y0 = lut[ii];
          float y1 = lut[ii + 1];
          sx = y0 + (y1 - y0) * t;
        }
      } else {
        float xx = xt[i];
        sx = xx / (1.0f + expf(-xx));
      }
      ot[i] = sx * yt[i];
    }
  }

  return 0;
}
