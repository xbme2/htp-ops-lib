#include <HAP_farf.h>
#include <HAP_perf.h>

#include "dsp/hvx_convert.h"
#include "dsp/hvx_math.h"
#include "dsp/vtcm_mgr.h"

static inline HVX_Vector silu_vec_f32_precompute_local(HVX_Vector v_x_sf) {
  const HVX_Vector v_zero   = Q6_V_vzero();
  const HVX_Vector v_one_sf = Q6_V_vsplat_R(0x3F800000);  // 1.0f
  const HVX_Vector v_log2e  = Q6_V_vsplat_R(0xBFB8AA3C);  // -1 / ln(2)

  HVX_Vector v_x_log2e_qf32 = Q6_Vqf32_vmpy_VsfVsf(v_x_sf, v_log2e);
  HVX_Vector v_x_log2e_sf   = Q6_Vsf_equals_Vqf32(v_x_log2e_qf32);
  HVX_Vector v_exp_sf       = hvx_my_exp2_vsf(v_x_log2e_sf);

  HVX_Vector v_denom_qf32 = Q6_Vqf32_vadd_VsfVsf(v_exp_sf, v_one_sf);
  HVX_Vector v_denom_sf   = Q6_Vsf_equals_Vqf32(v_denom_qf32);
  HVX_Vector v_inv_qf32   = hvx_my_inv_vqf32_vsf(v_denom_sf);

  HVX_Vector v_x_qf32   = Q6_Vqf32_vadd_VsfVsf(v_x_sf, v_zero);
  HVX_Vector v_out_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(v_x_qf32, v_inv_qf32);
  return Q6_Vsf_equals_Vqf32(v_out_qf32);
}

static void precompute_safe_softmax_exp2_table() {
  const int n_dup = 4;

  uint8_t *table = (uint8_t *) vtcm_manager_reserve_area("safe_softmax::exp2_hf_qf16", 65536 * n_dup, 65536);
  if (!table) {
    FARF(ALWAYS, "%s: VTCM reservation failed", __func__);
    return;
  }

  const int n = 32768;  // 32k fp16 elements in 64k area

  const int n_elems_per_vec = VLEN / sizeof(__fp16);
  const int n_vecs          = n / n_elems_per_vec;

  _Alignas(VLEN) uint16_t tmp[VLEN / sizeof(uint16_t)];

  HVX_Vector *pv_table = (HVX_Vector *) table;

  int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < n_vecs; ++i) {
    for (int j = 0; j < n_elems_per_vec; ++j) {
      int index = i * n_elems_per_vec + j;
      tmp[j]    = index | 0x8000;  // negative value
    }

    // *pv_table++ = hvx_my_exp2_vhf_vqf16(vmem(tmp));

    // promote computation precision
    HVX_VectorPair vp    = hvx_my_vqf16_to_wsf(vmem(tmp));
    HVX_Vector     v0_sf = hvx_my_exp2_vsf(Q6_V_lo_W(vp));
    HVX_Vector     v1_sf = hvx_my_exp2_vsf(Q6_V_hi_W(vp));
    pv_table[i]          = hvx_my_wsf_to_vhf(v1_sf, v0_sf);
  }
  int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);

  FARF(ALWAYS, "%s: precompute table took %lld us", __func__, elapsed_us);

  for (int dup = 1; dup < n_dup; ++dup) {
    HVX_Vector *dup_table = pv_table + n_vecs * dup;
    for (int i = 0; i < n_vecs; ++i) {
      dup_table[i] = pv_table[i];
    }
  }
}

static void precompute_swiglu_silu_neg_table() {
  const int n_dup = 1;

  uint8_t *table = (uint8_t *) vtcm_manager_reserve_area("swiglu::silu_neg_hf", 65536 * n_dup, 65536);
  if (!table) {
    FARF(ALWAYS, "%s: VTCM reservation failed", __func__);
    return;
  }

  const int n = 32768;  // 32k negative fp16 elements in 64k area
  const int n_elems_per_vec = VLEN / sizeof(__fp16);
  const int n_vecs          = n / n_elems_per_vec;

  _Alignas(VLEN) uint16_t tmp[VLEN / sizeof(uint16_t)];

  HVX_Vector *pv_table = (HVX_Vector *) table;

  int64_t t0 = HAP_perf_get_qtimer_count();
  for (int i = 0; i < n_vecs; ++i) {
    for (int j = 0; j < n_elems_per_vec; ++j) {
      int index = i * n_elems_per_vec + j;
      tmp[j]    = index | 0x8000;  // negative fp16 value
    }

    HVX_VectorPair vp = hvx_my_vhf_to_wsf(vmem(tmp));
    HVX_Vector v0_sf  = silu_vec_f32_precompute_local(Q6_V_lo_W(vp));
    HVX_Vector v1_sf  = silu_vec_f32_precompute_local(Q6_V_hi_W(vp));
    pv_table[i]       = hvx_my_wsf_to_vhf(v1_sf, v0_sf);
  }

  int64_t elapsed_us = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - t0);
  FARF(ALWAYS, "%s: precompute table took %lld us", __func__, elapsed_us);

  for (int dup = 1; dup < n_dup; ++dup) {
    HVX_Vector *dup_table = pv_table + n_vecs * dup;
    for (int i = 0; i < n_vecs; ++i) {
      dup_table[i] = pv_table[i];
    }
  }
}

void init_precomputed_tables();

void init_precomputed_tables() {
  precompute_safe_softmax_exp2_table();
  precompute_swiglu_silu_neg_table();
}
