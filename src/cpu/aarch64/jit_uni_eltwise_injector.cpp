/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_uni_eltwise_injector.hpp"

#ifndef DNNL_X64_IMPLEMENTATION
#ifdef CG
#undef CG
#endif
#define CG h->CodeGeneratorAArch64
#define IDX(a) static_cast<uint32_t>(a.getIdx())
#endif //#ifdef DNNL_X64_IMPLEMENTATION

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak;
#ifndef DNNL_X64_IMPLEMENTATION
namespace xa = Xbyak_aarch64;
#endif

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble(
#ifdef DNNL_X64_IMPLEMENTATION
        size_t start_idx, size_t end_idx) {
    using namespace Xbyak::util;
#else /* DNNL_X64_IMPLEMENTATION */
        const vmm_index_set_t &vmm_idxs) {
    using namespace alg_kind;
    //using namespace Xbyak_aarch64::util;
    using namespace Xbyak::util;
#endif /* DNNL_X64_IMPLEMENTATION */
    preserved_vecs_count = 0;
#ifdef DNNL_X64_IMPLEMENTATION
    vecs_to_preserve = aux_vecs_count();
    start_idx_tail = start_idx;
#else //#ifdef DNNL_X64_IMPLEMENTATION
    /* +1 for memory operand */
    //vecs_to_preserve = aux_vecs_count() + 1;
    vecs_to_preserve = aux_vecs_count();
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin()) + 1;
    start_idx_tail = vmm_idxs.begin();
#endif //#ifdef DNNL_X64_IMPLEMENTATION

    // For sse41 mask register has to be Xmm(0)
#ifdef DNNL_X64_IMPLEMENTATION
    if (isa == sse41 && vecs_to_preserve > 0) {
#else //#ifdef DNNL_X64_IMPLEMENTATION
    if (isa == simd && vecs_to_preserve > 0) {
#endif //#ifdef DNNL_X64_IMPLEMENTATION
        size_t idx = 0;
        assert(idx < start_idx);
        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
#ifdef DNNL_X64_IMPLEMENTATION
        preserved_vec_idxs[preserved_vecs_count++] = start_idx_tail++;
#else /* DNNL_X64_IMPLEMENTATION */
        preserved_vec_idxs[preserved_vecs_count++] = *start_idx_tail;
        ++start_idx_tail;
#endif //#ifdef DNNL_X64_IMPLEMENTATION
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    // Same logic but to allocate gprs
    size_t preserved_gprs_count = 0;
#ifdef DNNL_X64_IMPLEMENTATION
    for (size_t gpr_idx = 0; gpr_idx <= Operand::R15; ++gpr_idx) {
        int _idx = Operand::R15 - gpr_idx; // we allocate from the end
        if (preserved_gprs_count < aux_gprs_count()
                && !utils::one_of(_idx, p_table.getIdx(), Operand::RSP))
            preserved_gpr_idxs[preserved_gprs_count++] = _idx;
    }
#else /* DNNL_X64_IMPLEMENTATION */
    for (size_t gpr_idx = 0; gpr_idx <= 30; ++gpr_idx) {
        int _idx = 30 - gpr_idx; // we allocate from the end
        if (preserved_gprs_count < aux_gprs_count()
                && (((unsigned)_idx) != x_table.getIdx()))
            preserved_gpr_idxs[preserved_gprs_count++] = _idx;
    }
#endif /* DNNL_X64_IMPLEMENTATION */
    assert(preserved_gprs_count == aux_gprs_count());

#ifndef DNNL_X64_IMPLEMENTATION
    CG::ptrue(p_512.b);
    CG::ptrue(p_256.b, xa::VL32);
    CG::ptrue(p_128.b, xa::VL16);
    if (vlen == 32) {
        p_lsb = p_256;
    } else if (vlen == 16) {
        p_lsb = p_128;
    }
#endif

#ifdef DNNL_X64_IMPLEMENTATION
    if (save_state_) {
        h->push(p_table);
        for (size_t i = 0; i < preserved_gprs_count; ++i)
            h->push(Reg64(preserved_gpr_idxs[i]));

        if (preserved_vecs_count) h->sub(h->rsp, preserved_vecs_count * vlen);
#else /* DNNL_X64_IMPLEMENTATION */
    if (save_state_) {
      CG::str(x_table, pre_ptr(X_SP, -8));

        for (size_t i = 0; i < preserved_gprs_count; ++i) {
            /* This route has not been tested */
	  CG::str(xa::XReg(preserved_gpr_idxs[i]), xa::pre_ptr(X_SP, -8));
        }
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->uni_vmovups(
                    h->ptr[h->rsp + i * vlen], Vmm(preserved_vec_idxs[i]));
#else //#ifdef DNNL_X64_IMPLEMENTATION
	/*
        xa::XReg x_sp {IDX(h->rsp)};
        xa::XReg x_addr {h->xtDefaultAddrIdx};

        size_t i = 0;

        while (i < preserved_vecs_count) {
            int count = 0;
            int ii = i;
            do {
                CG::add_imm(x_tmp_vec[count++], x_sp, i * vlen, x_addr);
                i++;
            } while (i < preserved_vecs_count && count < x_tmp_vec_size);

            if (vlen != 32)
                for (int j = 0; j < count; j++)
                    CG::st1w(xa::ZRegS(preserved_vec_idxs[ii++]), p_lsb,
                            xa::ptr(x_tmp_vec[j]));
            else
                for (int j = 0; j < count; j++)
                    CG::str(xa::QReg(preserved_vec_idxs[ii++]),
                            xa::ptr(x_tmp_vec[j]));
        }
	*/
        size_t i = 0;

        while (i < preserved_vecs_count) {
            int count = 0;
            int ii = i;
            do {
	      CG::add_imm(x_tmp_vec[count++], X_SP, i * vlen,
                        X_DEFAULT_ADDR);
                i++;
            } while (i < preserved_vecs_count && count < x_tmp_vec_size);

            for (int j = 0; j < count; j++)
	      CG::st1w(xa::ZRegS(preserved_vec_idxs[ii++]), p_512,
		       xa::ptr(x_tmp_vec[j]));
        }
#endif //#ifdef DNNL_X64_IMPLEMENTATION
        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_preamble_tail(
#ifdef DNNL_X64_IMPLEMENTATION
        size_t start_idx) {
    size_t tail_vecs_to_preserve = start_idx_tail - start_idx;
#else /* DNNL_X64_IMPLEMENTATION */
        const vmm_index_set_iterator_t start_idx_it) {
    size_t tail_vecs_to_preserve = std::distance(start_idx_it, start_idx_tail);
#endif /* DNNL_X64_IMPLEMENTATION */
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
#ifdef DNNL_X64_IMPLEMENTATION
        if (idx_off) h->add(h->rsp, idx_off * vlen);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]),
                    h->ptr[h->rsp + i * vlen]);
#else //#ifdef DNNL_X64_IMPLEMENTATION
	/* This route has not been tested */
	/*
        if (idx_off) CG::add_imm(X_SP, X_SP, idx_off * vlen, X_TMP_0);
	
        xa::XReg x_sp {IDX(h->rsp)};
        xa::XReg x_addr {h->xtDefaultAddrIdx};
        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
                CG::add_imm(x_tmp_vec[count++], x_sp, i * vlen, x_addr);
                i++;
            } while (i < tail_vecs_to_preserve && count < x_tmp_vec_size);

            if (vlen != 32)
                for (int j = 0; j < count; j++)
                    CG::ld1w(xa::ZRegS(preserved_vec_idxs[idx_off + ii++]),
                            p_lsb / xa::T_z, xa::ptr(x_tmp_vec[j]));
            else
                for (int j = 0; j < count; j++)
                    CG::ldr(xa::QReg(preserved_vec_idxs[idx_off + ii++]),
                            xa::ptr(x_tmp_vec[j]));
        }
	*/
        if (idx_off) CG::add_imm(X_SP, X_SP, idx_off * vlen, X_TMP_0);

        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
	      CG::add_imm(x_tmp_vec[count++], X_SP, i * vlen,
                        X_DEFAULT_ADDR);
                i++;
            } while (i < tail_vecs_to_preserve && count < x_tmp_vec_size);

            for (int j = 0; j < count; j++)
	      CG::ld1w(xa::ZRegS(preserved_vec_idxs[idx_off + ii++]), p_512 / xa::T_z,
		       xa::ptr(x_tmp_vec[j]));
        }
#endif //#ifdef DNNL_X64_IMPLEMENTATION
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_) {
#ifdef DNNL_X64_IMPLEMENTATION
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[idx_off + i]));
#else //#ifdef DNNL_X64_IMPLEMENTATION
	/*
        xa::XReg x_sp {IDX(h->rsp)};
        xa::XReg x_addr {h->xtDefaultAddrIdx};
        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
                CG::add_imm(x_tmp_vec[count++], x_sp, i * vlen, x_addr);
                i++;
            } while (i < tail_vecs_to_preserve && count < x_tmp_vec_size);

            if (vlen != 32)
                for (int j = 0; j < count; j++)
                    CG::st1w(xa::ZRegS(preserved_vec_idxs[idx_off + ii++]),
                            p_lsb / xa::T_z, xa::ptr(x_tmp_vec[j]));
            else
                for (int j = 0; j < count; j++)
                    CG::str(xa::QReg(preserved_vec_idxs[idx_off + ii++]),
                            xa::ptr(x_tmp_vec[j]));
        }
	*/
        size_t i = 0;

        while (i < tail_vecs_to_preserve) {
            int count = 0;
            int ii = i;
            do {
	      CG::add_imm(x_tmp_vec[count++], X_SP, i * vlen,
                        X_DEFAULT_ADDR);
                i++;
            } while (i < tail_vecs_to_preserve && count < x_tmp_vec_size);

            for (int j = 0; j < count; j++)
	      CG::st1w(xa::ZRegS(preserved_vec_idxs[idx_off + ii++]), p_512 / xa::T_z,
		       xa::ptr(x_tmp_vec[j]));
        }
#endif //#ifdef DNNL_X64_IMPLEMENTATION

#ifdef DNNL_X64_IMPLEMENTATION
        if (idx_off) h->sub(h->rsp, idx_off * vlen);
#else /* DNNL_X64_IMPLEMENTATION */
        if (idx_off) {
	  CG::sub_imm(X_SP, X_SP, idx_off * vlen, X_TMP_0);
        }
#endif /* DNNL_X64_IMPLEMENTATION */
    }

    assign_regs();
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::injector_postamble() {
#ifdef DNNL_X64_IMPLEMENTATION
    using namespace Xbyak::util;
    if (!save_state_) return;

    for (size_t i = 0; i < preserved_vecs_count; ++i)
        h->uni_vmovups(Vmm(preserved_vec_idxs[i]), h->ptr[h->rsp + i * vlen]);

    if (preserved_vecs_count) h->add(h->rsp, preserved_vecs_count * vlen);

    for (int i = aux_gprs_count() - 1; i >= 0; --i)
        h->pop(Reg64(preserved_gpr_idxs[i]));
    h->pop(p_table);
#else /* DNNL_X64_IMPLEMENTATION */
    //using namespace Xbyak_aarch64::util;
    using namespace Xbyak::util;
    if (!save_state_) return;

    size_t i = 0;

    while (i < preserved_vecs_count) {
        int count = 0;
        int ii = i;
        do {
	  CG::add_imm(x_tmp_vec[count++], X_SP, i * vlen,
		      X_DEFAULT_ADDR);
            i++;
        } while (i < preserved_vecs_count && count < x_tmp_vec_size);

        for (int j = 0; j < count; j++)
	  CG::ld1w(xa::ZRegS(preserved_vec_idxs[ii++]), p_512 / xa::T_z,
		   xa::ptr(x_tmp_vec[j]));
    }

    if (preserved_vecs_count)
      CG::add_imm(X_SP, X_SP, preserved_vecs_count * vlen, X_TMP_0);

    for (int i = aux_gprs_count() - 1; i >= 0; --i)
      CG::ldr(xa::XReg(preserved_gpr_idxs[i]), xa::pre_ptr(X_SP, 8));
    CG::ldr(x_table, xa::pre_ptr(X_SP, 8));
#endif /* DNNL_X64_IMPLEMENTATION */
}

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    vmm_mask = Vmm(preserved_vec_idxs[0]);
    vmm_aux0 = Vmm(preserved_vec_idxs[0]);
    vmm_aux1 = Vmm(preserved_vec_idxs[1]);
    vmm_aux2 = Vmm(preserved_vec_idxs[2]);
    vmm_aux3 = Vmm(preserved_vec_idxs[3]);
    vmm_aux4 = Vmm(preserved_vec_idxs[4]);
}
#else //#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::assign_regs() {
    /* For translation of x64's memory operand instructions */
    z_tmp = xa::ZRegS(static_cast<uint32_t>(preserved_vec_idxs[0]));

    vmm_mask = xa::ZRegS(preserved_vec_idxs[1]);
    vmm_aux0 = xa::ZRegS(preserved_vec_idxs[1]);
    vmm_aux1 = xa::ZRegS(preserved_vec_idxs[2]);
    vmm_aux2 = xa::ZRegS(preserved_vec_idxs[3]);
    vmm_aux3 = xa::ZRegS(preserved_vec_idxs[4]);
    vmm_aux4 = xa::ZRegS(preserved_vec_idxs[5]);
    vmm_aux5 = xa::ZRegS(preserved_vec_idxs[6]);
    vmm_aux6 = xa::ZRegS(preserved_vec_idxs[7]);
    vmm_aux7 = xa::ZRegS(preserved_vec_idxs[8]);
}
#endif //#ifdef DNNL_X64_IMPLEMENTATION

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Stores a mask by applying cmpps on two inputs w/ a given predicate.
#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(const Vmm &vmm_src,
        const Xbyak::Operand &compare_operand, int cmp_predicate) {

    if (has_avx512()) {
        h->vcmpps(k_mask, vmm_src, compare_operand, cmp_predicate);
    } else {
        h->uni_vcmpps(vmm_mask, vmm_src, compare_operand, cmp_predicate);
    }
}
#else //#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_cmp_mask(
	 const xa::ZRegS &vmm_src, const xa::ZRegS &compare_operand, int cmp_predicate) {

    enum {
        EQ_OQ = 0,
        LT_OS = 1,
        LE_OS = 2,
        UNORD_Q = 3,
        NEQ_UQ = 4,
        NLT_US = 5,
        NLE_US = 6,
        ORD_Q = 7,
        EQ_UQ = 8,
        NGE_US = 9,
        NGT_US = 10,
        FALSE_OQ = 11,
        NEQ_OQ = 12,
        GE_OS = 13,
        GT_OS = 14,
        TRUE_UQ = 15,
        EQ_OS = 16,
        LT_OQ = 17,
        LE_OQ = 18,
        UNORD_S = 19,
        NEQ_US = 20,
        NLT_UQ = 21,
        NLE_UQ = 22,
        ORD_S = 23,
        EQ_US = 24,
        NGE_UQ = 25,
        NGT_UQ = 26,
        FALSE_OS = 27,
        NEQ_OS = 28,
        GE_OQ = 29,
        GT_OQ = 30,
        TRUE_US = 31,
    };

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE / xa::T_z, P_ALL_ONE.b);
    switch (cmp_predicate) {
        case EQ_OQ:
	  CG::fcmeq(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case LT_OS:
	  CG::fcmlt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case LE_OS:
	  CG::fcmle(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NEQ_UQ:
	  CG::fcmne(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NLT_US:
	  CG::fcmge(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NLE_US:
	  CG::fcmgt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case EQ_UQ:
	  CG::fcmeq(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NGE_US:
	  CG::fcmlt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NGT_US:
	  CG::fcmle(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NEQ_OQ:
	  CG::fcmne(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case GE_OS:
	  CG::fcmge(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case GT_OS:
	  CG::fcmgt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case EQ_OS:
	  CG::fcmeq(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case LT_OQ:
	  CG::fcmlt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case LE_OQ:
	  CG::fcmle(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NEQ_US:
	  CG::fcmne(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NLT_UQ:
	  CG::fcmge(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NLE_UQ:
	  CG::fcmgt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case EQ_US:
	  CG::fcmeq(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NGE_UQ:
	  CG::fcmlt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NGT_UQ:
	  CG::fcmle(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case NEQ_OS:
	  CG::fcmne(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case GE_OQ:
	  CG::fcmge(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;
        case GT_OQ:
	  CG::fcmgt(
                    xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, vmm_src, compare_operand);
            break;

        case UNORD_Q:
        case ORD_Q:
        case FALSE_OQ:
        case TRUE_UQ:
        case UNORD_S:
        case ORD_S:
        case FALSE_OS:
        case TRUE_US:
        default: assert(!"Unsupported compare mode"); break;
    }
}
#endif //#ifdef DNNL_X64_IMPLEMENTATION
 
#ifndef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::uni_ldr(
        const Vmm &vmm_dst, const Xbyak::Operand &addr) {
    RegExp exp = addr.getAddress().getRegExp();
    uint32_t base = IDX(exp.getBase());
    uint32_t index = IDX(exp.getIndex());
    bool isIndexExist = exp.getIndex().isREG();
    int scale = exp.getScale();
    int disp = exp.getDisp();
    xa::XReg x_base {base};
    xa::XReg x_index {index};
    xa::XReg x_tmp_addr {h->xtDefaultAddrIdx};
    xa::ZReg z_dst {IDX(vmm_dst)};

    /* At this time, ignore disp only addressing. */
    assert(exp.getBase().getBit());
    assert(scale == 1);
    (void)scale;

    /* Address calculation */
    if (isIndexExist == false && disp == 0) {
        x_tmp_addr = x_base;
    } else if (isIndexExist && disp == 0) {
        CG::add(x_tmp_addr, x_base, x_index);
    } else if (isIndexExist == false && disp) {
        CG::add_imm(x_tmp_addr, x_base, disp, h->X_TMP_0);
    } else {
        CG::add(x_tmp_addr, x_base, x_index);
        CG::add_imm(x_tmp_addr, x_base, disp, h->X_TMP_0);
    }

    CG::ld1w(z_dst.s, p_lsb / xa::T_z, xa::ptr(x_tmp_addr));
}
#endif //#ifdef DNNL_X64_IMPLEMENTATION

// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Blends a result of second input into a first input w/ a stored mask.
#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
        const Vmm &vmm_dst, const Xbyak::Operand &src) {
    if (has_avx512()) {
        h->vblendmps(vmm_dst | k_mask, vmm_dst, src);
    } else {
        h->uni_vblendvps(vmm_dst, vmm_dst, src, vmm_mask);
    }
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::blend_with_mask(
	const xa::ZRegS &vmm_dst, const xa::ZRegS &src) {
  CG::sel(vmm_dst, p_mask / xa::T_m, src, vmm_dst);
}
#endif /* DNNL_X64_IMPLEMENTATION */
 
// Uses injector masks objects: k_mask (>= avx512_common) or vmm_mask (<= avx2).
// Tests a mask for all zeros. If all zeroes occur, set ZF = 1.
// Nicely combines with jump_if_zero (jz).
#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::test_mask() {
    if (has_avx512()) {
        h->kortestw(k_mask, k_mask);
    } else {
        h->uni_vtestps(vmm_mask, vmm_mask);
    }
}
#else /* DNNL_X64_IMPLEMENTATION */

#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
        const Vmm &vmm_src) {
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux2, vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux2);

    // x = x - fx * ln2
    h->uni_vfnmadd231ps(vmm_aux1, vmm_aux2, table_val(ln2f));

    // compute 2^n
    h->uni_vcvtps2dq(vmm_aux2, vmm_src);
    h->uni_vpaddd(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits); //Vmm(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    h->uni_vpxor(vmm_src, vmm_src, vmm_src);
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux2, vmm_src);

    // compute polynomial
    h->uni_vmovups(vmm_src, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    // y = y * 2^n
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_fwd(
      const xa::ZRegS &vmm_src) {

    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(exp_ln_flt_max_f))));
    CG::fminnm(z_tmp, p_tmp0, vmm_src);
    CG::fmin(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));

    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(exp_ln_flt_min_f))));
    CG::fmaxnm(z_tmp, p_tmp0, vmm_src);
    CG::fmax(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));

    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_src)));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(exp_log2ef))));
    CG::fadd(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(half))));

    // tmp = floorf(fx)

    CG::frintm(vmm_aux2, p_tmp0 / xa::T_m, vmm_src);

    // keep vmm_src = fx for further computations
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux2)));

    // x = x - fx * ln2
    CG::fmls(vmm_aux1, p_tmp0 / xa::T_m, vmm_aux2, xa::ZRegS(IDX(table_val(ln2f))));

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    CG::fsub(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(one))));
    CG::frinti(vmm_aux2, p_tmp0 / xa::T_m, vmm_src);
    CG::fcvtzs(vmm_aux2, p_tmp0 / xa::T_m, vmm_aux2);
    CG::add(vmm_aux2, vmm_aux2, xa::ZRegS(IDX(table_val(exponent_bias))));
    CG::lsl(vmm_aux2, vmm_aux2,
            n_mantissa_bits); //TRegS(6) = 2^-fx

    // use vmm_src as tmp vmm_zero when applying mask
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)));
    // set zeroes at those points which were < log(FLT_MIN)
    blend_with_mask(vmm_aux2, vmm_src);

    // compute polynomial
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(exp_pol, 4))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(exp_pol, 3))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(exp_pol, 2))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(exp_pol, 1))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(exp_pol, 0))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));

    // y = y * 2^n
    CG::fmul(vmm_src, vmm_src, vmm_aux2);
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(two))));
}

#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux1, vmm_src);
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, vmm_aux1);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
  CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    blend_with_mask(vmm_src, vmm_aux1);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_zero_ns_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
  CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(zero))));
    CG::fmaxnm(z_tmp, p_tmp0, vmm_src);
    CG::fmax(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    h->uni_vsubps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));

    // combine with mask
    compute_cmp_mask(vmm_aux3, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux3);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_fwd(
       const xa::ZRegS &vmm_src) {
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
  CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux3, p_tmp0 / xa::T_m, 0);

    // compute exponent
    exp_compute_vector_fwd(vmm_src);

    // alpha * (exp(x) - 1)
    CG::fsub(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(one))));
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));

    // combine with mask
    compute_cmp_mask(vmm_aux3, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux3);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    // we add a check as the avx2 code cannot be used for avx
    assert(IMPLICATION(isa == avx2, mayiuse(avx2)));

    using namespace Xbyak::util;
    const int XMM_float_lanes_count = 4;
    const int tanh_n_polynomials = 32;

    // register mapping
    // TODO: put sign on stack and alias zmm_table2 with vmm_sign to save a reg ?
    Vmm vmm_dst = vmm_aux1, vmm_src_shift = vmm_aux1, vmm_coeff = vmm_aux1,
        vmm_pol = vmm_aux2, vmm_indices = vmm_aux3, vmm_src_original = vmm_aux4,
        vmm_sign = vmm_aux4;
    Reg64 gpr_idx[XMM_float_lanes_count];

    if (isa == sse41) {
        assert(aux_gprs_count() >= XMM_float_lanes_count);
        for (int i = 0; i < XMM_float_lanes_count; i++)
            gpr_idx[i] = Reg64(preserved_gpr_idxs[i]);
    }

    // We split the positive domain in 33 intervals:
    // a) [0; linear_ubound]: in this interval tanh(x) = x
    // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
    //    half binade
    // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
    //    one interval for each half binade, there are 29 of those
    // d) [0x1.0p3; saturation_ubound]:
    //    This interval spans part of a half binade
    // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
    // For b-d, we need 31 polynomials and will do a table lookup for those.
    // To simplify the logic, we will also put a) in the table.

    // The polynomials are of degree 6, so we need to gather 7 coefficients.
    // - sse4.1: we do it the naive way using vextract/vinsert.
    //           Here we will extract the indices in gpr only once and
    //           reuse them as there are only 4 of them.
    // - avx2: we use vpermps and blend for each coefficient.
    //         This needs an extra vmm to store the mask
    // - avx512: because the table fits in 2 registers, we can use vpermi2d.
    auto coeffs_off = [&](int coeff_off, int off = 0) {
        return table_off(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto coeffs_address = [&](int coeff_off, int off = 0) {
        return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto gather_coefficient_init = [&](Vmm vmm_pol_idx, int nelems) {
        switch (isa) {
            case sse41:
                for (int i = 0; i < XMM_float_lanes_count; ++i)
                    h->pextrd(gpr_idx[i].cvt32(), vmm_pol_idx, i);
                break;
            case avx2:
                // needed for gather instruction
                h->uni_vxorps(vmm_mask, vmm_mask, vmm_mask);
                break;
            case avx512_common:
            case avx512_core: break;
            default: assert(!"unimplemented");
        }
    };
    auto gather_coefficient = [&](Vmm vmm_coeff, int coeff_idx,
                                      Vmm vmm_pol_idx) {
        switch (isa) {
            case sse41:
                for (int idx = 0; idx < 4; ++idx) {
                    Xbyak::Address coeff_addr
                            = ptr[p_table + coeffs_off(coeff_idx)
                                    + gpr_idx[idx] * sizeof(float)];
                    h->pinsrd(vmm_coeff, coeff_addr, idx);
                }
                break;
            case avx2: {
                Xbyak::Address idx_addr = ptr[p_table + coeffs_off(coeff_idx)
                        + vmm_pol_idx * sizeof(float)];
                // we set the mask to all ones to gather full
                // register.  needs to be done after each gather since
                // since the gather instructions zeros the mask if
                // successful
                h->uni_vcmpps(vmm_mask, vmm_mask, vmm_mask, _cmp_eq_oq);
                h->vgatherdps(vmm_coeff, idx_addr, vmm_mask);
                break;
            }
                // use gather instruction
            case avx512_common:
            case avx512_core:
                // we use vpermt2ps to not override the indices
                // this also enables to save a register for table loading
                {
                    Zmm zmm_coeff(vmm_coeff.getIdx());
                    Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
                    h->uni_vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
                    h->vpermt2ps(zmm_coeff, zmm_pol_idx,
                            coeffs_address(coeff_idx, 16));
                    break;
                }
            default: assert(!"unimplemented");
        }
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    h->uni_vmovups(vmm_src_original, vmm_src);
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // We compute the indices for the table lookup
    h->uni_vmovups(vmm_indices, vmm_src);
    h->uni_vpsubd(vmm_indices, vmm_indices, table_val(tanh_idx_bias));
    h->uni_vandps(vmm_indices, vmm_indices, table_val(tanh_idx_mask));
    h->uni_vpsrld(vmm_indices, vmm_indices, 22);

    // we do the argument reduction
    h->uni_vmovups(vmm_src_shift, vmm_src);
    h->uni_vandps(vmm_src_shift, vmm_src_shift, table_val(tanh_idx_mask));
    h->uni_vsubps(vmm_src, vmm_src, vmm_src_shift);

    // we gather and evaluate the polynonials
    gather_coefficient_init(vmm_indices, vlen / sizeof(float));
    gather_coefficient(vmm_pol, 6, vmm_indices);
    for (int deg = 5; deg >= 0; --deg) {
        gather_coefficient(vmm_coeff, deg, vmm_indices);
        h->uni_vfmadd213ps(vmm_pol, vmm_src, vmm_coeff);
    }

    // we restore src with cleared sign, and keep sign
    assert(vmm_sign.getIdx() == vmm_src_original.getIdx());
    h->uni_vmovups(vmm_src, vmm_src_original);
    h->uni_vandps(vmm_sign, vmm_sign, table_val(sign_mask));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    h->uni_vmovups(vmm_dst, table_val(one));
    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    h->uni_vmovups(vmm_mask, table_val(tanh_saturation_lbound));
    compute_cmp_mask(vmm_mask, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_pol);
    // [0; linear_ubound]  : we return x
    h->uni_vmovups(vmm_mask, table_val(tanh_linear_ubound));
    compute_cmp_mask(vmm_mask, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_src);

    // We reapply the sign and return
    h->uni_vxorps(vmm_dst, vmm_dst, vmm_sign);
    h->uni_vmovups(vmm_src, vmm_dst);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
    // we add a check as the avx2 code cannot be used for avx
    //using namespace Xbyak_aarch64::util;
  using namespace Xbyak::util;
    const int tanh_n_polynomials = 32;

    // register mapping
    // TODO: put sign on stack and alias zmm_table2 with vmm_sign to save a reg ?
    xa::ZRegS vmm_dst = vmm_aux1, vmm_src_shift = vmm_aux1, vmm_coeff = vmm_aux1,
          vmm_pol = vmm_aux2, vmm_indices = vmm_aux3,
          vmm_src_original = vmm_aux4, vmm_sign = vmm_aux4;

    auto vpermt2ps_aarch64 =
      [&](const xa::ZRegS &d, const xa::ZRegS &s, const xa::ZRegS &s2, const xa::ZRegS &t,
                    const xa::ZRegS &t2, const xa::ZRegS &t3, const xa::PReg &p) {
                CG::ptrue(p.b);
                CG::mov(t, 0x1f);
                CG::and_(xa::ZRegB(t.getIdx()), p, xa::ZRegB(s.getIdx()));
                for (int i = 0; i < 16; i++) {
                    CG::cmpeq(P_TMP_0.s, p, t, 0);
                    CG::sub(t, 0x1);
                    CG::dup(t2, d[i]);
                    CG::mov(t3, h->P_TMP_0 / xa::T_m, t2);
                }
                for (int i = 0; i < 16; i++) {
                    CG::cmpeq(P_TMP_0.s, p, t, i);
                    CG::dup(t2, s2[i]);
                    CG::mov(t3, P_TMP_0 / xa::T_m, t2);
                }
                CG::mov(xa::ZRegD(d.getIdx()), xa::ZRegD(t3.getIdx()));
            };

    // We split the positive domain in 33 intervals:
    // a) [0; linear_ubound]: in this interval tanh(x) = x
    // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
    //    half binade
    // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
    //    one interval for each half binade, there are 29 of those
    // d) [0x1.0p3; saturation_ubound]:
    //    This interval spans part of a half binade
    // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
    // For b-d, we need 31 polynomials and will do a table lookup for those.
    // To simplify the logic, we will also put a) in the table.

    // The polynomials are of degree 6, so we need to gather 7 coefficients.
    // - sse4.1: we do it the naive way using vextract/vinsert.
    //           Here we will extract the indices in gpr only once and
    //           reuse them as there are only 4 of them.
    // - avx2: we use vpermps and blend for each coefficient.
    //         This needs an extra vmm to store the mask
    // - avx512: because the table fits in 2 registers, we can use vpermi2d.
    auto coeffs_address = [&](int coeff_off, int off = 0) {
        return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto gather_coefficient_init = [&](xa::ZRegS vmm_pol_idx, int nelems) {
        switch (isa) {
            case sve: break;
            default: assert(!"unimplemented");
        }
    };
    auto gather_coefficient = [&](xa::ZRegS vmm_coeff, int coeff_idx,
                                      xa::ZRegS vmm_pol_idx) {
        switch (isa) {
                // use gather instruction
            case sve:
                // we use vpermt2ps to not override the indices
                // this also enables to save a register for table loading
                {
                    xa::ZReg zmm_coeff(vmm_coeff.getIdx());
                    xa::ZReg zmm_pol_idx(vmm_pol_idx.getIdx());
                    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
                    CG::mov(xa::ZRegD(IDX(zmm_coeff)),
                            xa::ZRegD(IDX(coeffs_address(coeff_idx, 0))));
                    CG::mov(xa::ZRegS(IDX(zmm_coeff)), p_tmp0 / xa::T_m, 0);

                    vpermt2ps_aarch64(xa::ZRegS(IDX(zmm_coeff)),
                            xa::ZRegS(IDX(zmm_pol_idx)),
                            xa::ZRegS(IDX(coeffs_address(coeff_idx, 16))), vmm_aux5,
                            vmm_aux6, vmm_aux7, p_tmp0);
                    break;
                }
            default: assert(!"unimplemented");
        }
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x postive
    // and reapply sign at the end
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src_original)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(xa::ZRegS(IDX(vmm_src_original)), p_tmp0 / xa::T_m, 0);
    CG::and_(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(positive_mask))));

    // We compute the indices for the table lookup
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_indices)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(xa::ZRegS(IDX(vmm_indices)), p_tmp0 / xa::T_m, 0);
    CG::sub(xa::ZRegS(IDX(vmm_indices)), xa::ZRegS(IDX(vmm_indices)),
            xa::ZRegS(IDX(table_val(tanh_idx_bias))));
    CG::and_(xa::ZRegD(IDX(vmm_indices)), xa::ZRegD(IDX(vmm_indices)),
            xa::ZRegD(IDX(table_val(tanh_idx_mask))));
    CG::lsr(xa::ZRegS(IDX(vmm_indices)), xa::ZRegS(IDX(vmm_indices)), 22);

    // we do the argument reduction
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src_shift)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(xa::ZRegS(IDX(vmm_src_shift)), p_tmp0 / xa::T_m, 0);
    CG::and_(xa::ZRegD(IDX(vmm_src_shift)), xa::ZRegD(IDX(vmm_src_shift)),
            xa::ZRegD(IDX(table_val(tanh_idx_mask))));
    CG::fsub(vmm_src, vmm_src, xa::ZRegS(IDX(vmm_src_shift)));

    // we gather and evaluate the polynonials
    gather_coefficient_init(vmm_indices, vlen / sizeof(float));
    gather_coefficient(vmm_pol, 6, vmm_indices);

    for (int deg = 5; deg >= 0; --deg) {
        gather_coefficient(vmm_coeff, deg, vmm_indices);
        CG::fmad(xa::ZRegS(IDX(vmm_pol)), p_512 / xa::T_m, vmm_src,
                xa::ZRegS(IDX(vmm_coeff)));
    }

    // we restore src with cleared sign, and keep sign
    assert(vmm_sign.getIdx() == vmm_src_original.getIdx());
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src_original)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    CG::and_(xa::ZRegD(IDX(vmm_sign)), xa::ZRegD(IDX(vmm_sign)),
            xa::ZRegD(IDX(table_val(sign_mask))));
    CG::and_(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(positive_mask))));

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_dst)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_dst, p_tmp0 / xa::T_m, 0);

    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_mask)), xa::ZRegD(IDX(table_val(tanh_saturation_lbound))));
    CG::mov(vmm_mask, p_tmp0 / xa::T_m, 0);

    compute_cmp_mask(vmm_mask, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_pol);

    // [0; linear_ubound]  : we return x
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_mask)), xa::ZRegD(IDX(table_val(tanh_linear_ubound))));
    CG::mov(vmm_mask, p_tmp0 / xa::T_m, 0);

    compute_cmp_mask(vmm_mask, vmm_src, _cmp_gt_os);
    blend_with_mask(vmm_dst, vmm_src);

    // We reapply the sign and return
    CG::eor(xa::ZRegD(IDX(vmm_dst)), xa::ZRegD(IDX(vmm_dst)), xa::ZRegD(IDX(vmm_sign)));

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_dst)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vmovups(vmm_aux1, table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

    // save x on stack as tanh uses vmm_aux0
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux0);

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->uni_vaddps(vmm_src, vmm_src, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_fwd(
	     const xa::ZRegS &vmm_src) {
  CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    CG::fmul(vmm_src, vmm_src, vmm_src);
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)),
            xa::ZRegD(IDX(table_val(gelu_tanh_fitting_const))));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));
    CG::fmul(vmm_src, vmm_src, vmm_aux0);
    CG::fmul(vmm_src, vmm_src,
            xa::ZRegS(IDX(table_val(gelu_tanh_sqrt_two_over_pi))));

    // save x on stack as tanh uses vmm_aux0
    CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);

    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::str(xa::ZReg(IDX(vmm_aux0)), xa::ptr(X_TMP_0));

    // compute tanh(G(x))
    tanh_compute_vector_fwd(vmm_src);

    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux0)), xa::ptr(X_TMP_0));
    CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);

    // compute 0.5 * x * (1 + tanh(G(x)))
    CG::fadd(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(one))));
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(half))));
    CG::fmul(vmm_src, vmm_src, vmm_aux0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_fwd(
	  const xa::ZRegS &vmm_src) {
  CG::fmul(vmm_src, vmm_src, vmm_src);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
    h->uni_vandps(vmm_src, vmm_src, table_val(positive_mask));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_fwd(
       const xa::ZRegS &vmm_src) {
    // compute abs(x) = _mm_and_ps(x, 01111..111));
  CG::and_(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
	   xa::ZRegD(IDX(table_val(positive_mask))));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vsqrtps(vmm_src, vmm_src);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
  CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b, P_ALL_ONE.b);
    CG::fsqrt(vmm_src, p_tmp0 / xa::T_m, vmm_src);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
        const Vmm &vmm_src) {
    // compute x = alpha * x + beta;
    h->uni_vmovups(vmm_aux0, table_val(alpha));
    h->uni_vfmadd213ps(vmm_src, vmm_aux0, table_val(beta));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_fwd(
	  const xa::ZRegS &vmm_src) {
    // compute x = alpha * x + beta;
  CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(table_val(alpha))));
    CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux0, xa::ZRegS(IDX(table_val(beta))));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
    h->uni_vminps(vmm_src, vmm_src, table_val(alpha));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(zero))));
    CG::fmaxnm(z_tmp, p_tmp0, vmm_src);
    CG::fmax(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(alpha))));
    CG::fminnm(z_tmp, p_tmp0, vmm_src);
    CG::fmin(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vmaxps(vmm_src, vmm_src, table_val(alpha));
    h->uni_vminps(vmm_src, vmm_src, table_val(beta));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
  CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(alpha))));
    CG::fmaxnm(z_tmp, p_tmp0, vmm_src);
    CG::fmax(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(beta))));
    CG::fminnm(z_tmp, p_tmp0, vmm_src);
    CG::fmin(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
        const Vmm &vmm_src) {
    // keep src for further computations
    h->uni_vmovups(vmm_aux2, vmm_src);

    h->uni_vminps(vmm_src, vmm_src, table_val(exp_ln_flt_max_f));
    h->uni_vmaxps(vmm_src, vmm_src, table_val(exp_ln_flt_min_f));
    h->uni_vmovups(vmm_aux1, vmm_src);
    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->uni_vmulps(vmm_src, vmm_src, table_val(exp_log2ef));
    h->uni_vaddps(vmm_src, vmm_src, table_val(half));

    // tmp = floorf(fx)
    h->uni_vroundps(vmm_aux0, vmm_src, _op_floor);

    // keep vmm_src = fx for further computations
    h->uni_vmovups(vmm_src, vmm_aux0);

    // x = x - fx * ln2
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(ln2f));
    h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial
    h->uni_vmovups(vmm_aux3, table_val(exp_pol, 4));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 3));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 2));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 1));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(exp_pol, 0));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, table_val(one));

    // compute 2^(-n)
    if (has_avx512()) {
        h->vmulps(vmm_aux1, vmm_src, table_val(minus_one));
        h->vcvtps2dq(vmm_aux1, vmm_aux1);
    } else {
        h->uni_vcvtps2dq(vmm_aux1, vmm_src);
        h->uni_vpsignd(vmm_aux1, vmm_aux1, table_val(minus_one));
    }

    h->uni_vpaddd(vmm_aux1, vmm_aux1, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux1, vmm_aux1, n_mantissa_bits); //vmm_aux1 = 2^-fx
    // calculate ln(1 + y)
    h->uni_vaddps(vmm_aux3, vmm_aux3, vmm_aux1);
    // frexp()
    h->uni_vpsrld(vmm_src, vmm_aux3, n_mantissa_bits);
    h->uni_vcvtdq2ps(vmm_src, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    h->uni_vsubps(vmm_src, vmm_src, table_val(soft_relu_one_twenty_six));

    // and with mask (to get 0.5 * mantissa)
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(soft_relu_mantissa_sign_mask));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    h->uni_vorps(vmm_aux3, vmm_aux3, table_val(half));
    // y  = y - 1
    h->uni_vsubps(vmm_aux3, vmm_aux3, table_val(one));

    // compute log1p polynomial
    h->uni_vmovups(vmm_aux1, table_val(soft_relu_pol, 8));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 7));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 6));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 5));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux3, table_val(soft_relu_pol, 0));
    //calculate ln(2) * n
    h->uni_vmulps(vmm_src, vmm_src, table_val(ln2f));
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_fwd(
     const xa::ZRegS &vmm_src) {
    // ln(1 + exp(x)) =
    // = ln(1 + exp(n * ln(2) + r)) // divide x by ln(2) and get quot and rem
    // = ln(1 + 2^n * exp(r)) // simplify the exp(n*ln(2)) expression
    // = ln(2 ^ 0 + 2^n * exp(r)) // note 1 = 2^0
    // = ln(2 ^ (n - n) + 2^n * exp(r)) // 2^0 = 2^(n-n)
    // = ln(2 ^ n * (2^-n + exp(r))) // factorize with 2^n
    // = n * ln(2) + ln(2^-n + exp(r)) // take the 2^n factor out of the ln

  CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);

    // keep src for further computations
    CG::mov(xa::ZRegD(IDX(vmm_aux2)), xa::ZRegD(IDX(vmm_src)));

    CG::fminnm(xa::ZRegS(IDX(table_val(exp_ln_flt_max_f))), p_tmp0, vmm_src);
    CG::fmin(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));
    CG::fmaxnm(xa::ZRegS(IDX(table_val(exp_ln_flt_min_f))), p_tmp0, vmm_src);
    CG::fmax(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_src)));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(exp_log2ef))));
    CG::fadd(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(half))));

    // tmp = floorf(fx)
    CG::frintm(vmm_aux0, p_tmp0 / xa::T_m, vmm_src);

    // keep vmm_src = fx for further computations
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));

    // x = x - fx * ln2
    CG::fmul(vmm_aux0, vmm_aux0, xa::ZRegS(IDX(table_val(ln2f))));
    CG::fsub(vmm_aux1, vmm_aux1, vmm_aux0);
    // compute exponent polynomial
    CG::mov(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(table_val(exp_pol, 4))));
    CG::fmad(vmm_aux3, p_tmp0 / xa::T_m, vmm_aux1,
            xa::ZRegS(IDX(table_val(exp_pol, 3))));
    CG::fmad(vmm_aux3, p_tmp0 / xa::T_m, vmm_aux1,
            xa::ZRegS(IDX(table_val(exp_pol, 2))));
    CG::fmad(vmm_aux3, p_tmp0 / xa::T_m, vmm_aux1,
            xa::ZRegS(IDX(table_val(exp_pol, 1))));
    CG::fmad(vmm_aux3, p_tmp0 / xa::T_m, vmm_aux1,
            xa::ZRegS(IDX(table_val(exp_pol, 0))));
    CG::fmad(vmm_aux3, p_tmp0 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));

    // We do not count 2^-n here, because n can reach 128 and 2^(-128) is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^-n + exp(r) will be counted (2^-(n-1) + 2*exp(r))/2, because 2^(-127)
    // and 2 are numbers representable in fp32.

    // compute 2^-(n-1)
    // vmm_src now represents n-1
    CG::fsub(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(one))));
    CG::fmul(vmm_aux1, vmm_src, xa::ZRegS(IDX(table_val(minus_one))));

    CG::frinti(vmm_aux1, p_tmp0 / xa::T_m, vmm_aux1);
    CG::fcvtzs(vmm_aux1, p_tmp0 / xa::T_m, vmm_aux1);
    // restore vmm_src to n
    CG::fadd(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(one))));

    CG::add(vmm_aux1, vmm_aux1, xa::ZRegS(IDX(table_val(exponent_bias))));
    CG::lsl(vmm_aux1, vmm_aux1, n_mantissa_bits);
    // calculate ln(1 + y)
    CG::fmul(vmm_aux3, vmm_aux3,
            xa::ZRegS(IDX(table_val(two)))); // 2*exp(r)
    CG::fadd(vmm_aux3, vmm_aux3,
            vmm_aux1); // 2^-(n-1) + 2*exp(r)
    CG::fdiv(vmm_aux3, p_tmp0 / xa::T_m,
            xa::ZRegS(IDX(table_val(two)))); // (2^-(n-1) + 2*exp(r))/2

    // frexp()
    CG::lsr(vmm_src, vmm_aux3, n_mantissa_bits);
    CG::scvtf(vmm_src, p_tmp0 / xa::T_m, vmm_src);
    // got n. where n is x = 2^n * y. y = 0.5 .. 1
    CG::fsub(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(soft_relu_one_twenty_six))));

    // and with mask (to get 0.5 * mantissa)
    CG::and_(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_aux3)),
            xa::ZRegD(IDX(table_val(soft_relu_mantissa_sign_mask))));
    // got y. (mantisa)  0.5 < y < 1 (or with (to get 0.5 * mantissa))
    CG::orr(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_aux3)),
            xa::ZRegD(IDX(table_val(half))));
    // y  = y - 1
    CG::fsub(vmm_aux3, vmm_aux3, xa::ZRegS(IDX(table_val(one))));

    // compute log1p polynomial
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(table_val(soft_relu_pol, 8))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 7))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 6))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 5))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 4))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 3))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 2))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 1))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux3,
            xa::ZRegS(IDX(table_val(soft_relu_pol, 0))));
    //calculate ln(2) * n
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(ln2f))));
    CG::fadd(vmm_src, vmm_src, vmm_aux1);
    CG::fadd(vmm_src, vmm_src, vmm_aux0);

    // get vmm_mask = src > max logf
    // y = (x < max log f) ? soft_relu(x) : x
    compute_cmp_mask(vmm_aux2, table_val(exp_ln_flt_max_f), _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_aux2);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
        const Vmm &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.

    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);
    // we store the original sign and make x negative
    h->uni_vandps(vmm_aux3, vmm_aux3, table_val(sign_mask));
    h->uni_vorps(vmm_src, vmm_src, table_val(sign_mask));

    exp_compute_vector_fwd(vmm_src);
    // dup exp(x)
    h->uni_vmovups(vmm_aux1, vmm_src);
    // (exp(x) + 1)
    h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(one));
    // y = exp(x) / (exp(x) + 1)
    h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->uni_vmovups(vmm_aux2, table_val(one));
    h->uni_vsubps(vmm_aux2, vmm_aux2, vmm_src);
    if (has_avx512()) {
        h->vptestmd(k_mask, vmm_aux3, vmm_aux3);
    } else {
        h->uni_vmovups(vmm_mask, vmm_aux3);
    }
    blend_with_mask(vmm_aux2, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux2);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_fwd(
	    const xa::ZRegS &vmm_src) {
    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_aux3 for the mask as exp_compute does not use it.
  CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux3, p_tmp0 / xa::T_m, 0);
    // we store the original sign and make x negative
    CG::and_(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_aux3)),
            xa::ZRegD(IDX(table_val(sign_mask))));
    CG::orr(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(sign_mask))));

    exp_compute_vector_fwd(vmm_src);

    // dup exp(x)
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    // (exp(x) + 1)
    CG::fadd(vmm_aux1, vmm_aux1, xa::ZRegS(IDX(table_val(one))));
    // y = exp(x) / (exp(x) + 1)
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE, P_ALL_ONE.b);
    CG::fdiv(vmm_src, p_tmp0, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux2)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_aux2, p_tmp0 / xa::T_m, 0);
    CG::fsub(vmm_aux2, vmm_aux2, vmm_src);

    CG::movs(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::and_(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_aux3)));
    CG::cmpne(xa::PRegS(IDX(p_mask)), p_tmp0 / xa::T_z, z_tmp, 0);

    blend_with_mask(vmm_aux2, vmm_src);

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux2)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Save src data on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // x*alpha
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_fwd(
	 const xa::ZRegS &vmm_src) {
    // Save src data on stack for later usage
    CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));
    // x*alpha
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    // sigmoid(x*alpha)
    logistic_compute_vector_fwd(vmm_src);
    // x*sigmoid(alpha*x)
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux0)), xa::ptr(X_TMP_0));
    CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);

    CG::fmul(vmm_src, vmm_src, vmm_aux0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
        const Vmm &vmm_src) {
    // From J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010
    // Here is a brief mathematics to approximate log(x):
    // log(x) = E * log(2) + log(y), where -log(2)/2 <= log(y) <= log(2)/2;
    // log(y) = log(1 + z) - log(r_i), where z = y * r_i - 1, r_i approximates
    //   1 / y, i is index of one of precomputed values;
    // log(1 + z) ~~ polynomial(z), =>
    // if (x is normal)
    //     log(x) ~~ E * log(2) + polynomial(z) - log(r_i),
    // where log(r_i) is table value.
    //
    // If (x == 0) result = -inf;
    // If (x < 0) result = qnan;

    // save source on stack to check neg and zero values at the end
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // compute i
    const int approx_order = 5;
    h->uni_vpsrld(vmm_aux1, vmm_src, n_mantissa_bits - approx_order);
    h->uni_vandps(vmm_aux1, vmm_aux1, table_val(log_five_bit_offset));
    h->uni_vpslld(vmm_aux1, vmm_aux1, 1); // multiply i by 2

    // compute anticancellation i
    h->uni_vpsrld(vmm_aux2, vmm_aux1, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    h->uni_vpsrld(vmm_aux3, vmm_src, n_mantissa_bits);
    h->uni_vpaddd(vmm_aux3, vmm_aux3, vmm_aux2);
    h->uni_vcvtdq2ps(vmm_aux3, vmm_aux3);

    // get m (mantissa)
    h->uni_vxorps(vmm_aux2, vmm_aux2, table_val(exponent_bias));
    h->uni_vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits);
    h->uni_vandps(vmm_src, vmm_src, table_val(log_mantissa_mask));
    h->uni_vorps(vmm_src, vmm_src, vmm_aux2);

    // At first, adjust indices for table structure which broadcasts elements
    if (has_avx512()) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 4); // multiply by simd_w = 16
    } else if (isa == avx2) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 3); // multiply by simd_w = 8
    } else if (isa == sse41) {
        h->uni_vpslld(vmm_aux1, vmm_aux1, 2); // multiply by simd_w = 4
    }

    const auto it = entry_map_.find(log_predefined_vals);
    assert(it != entry_map_.end());
    const auto table_start_idx = (*it).second.off;

    auto gather_table_values = [&](const Vmm &vmm_dst, const Vmm &vmm_idxs,
                                       size_t offt = 0) {
        Xbyak::Address table_idx = h->ptr[p_table + table_start_idx + offt
                + vmm_idxs * sizeof(float)];
        if (has_avx512()) {
	  //#ifdef DNNL_X64_IMPLEMENTATION
            h->kmovw(k_mask, table_val(log_full_k_reg_mask));
	    /*
#else //#ifdef DNNL_X64_IMPLEMENTATION
            CG::ptrue(xa::PRegS {IDX(k_mask)}, xa::VL16);
#endif //#ifdef DNNL_X64_IMPLEMENTATION
	    */
            h->vgatherdps(vmm_dst | k_mask, table_idx);
        } else if (isa == avx2) {
            h->uni_vmovups(vmm_mask, table_val(sign_mask));
            h->vgatherdps(vmm_dst, table_idx, vmm_mask);
        } else if (isa == sse41) {
            Xbyak::Reg64 reg_tmp
                    = p_table.getIdx() != h->r9.getIdx() ? h->r9 : h->r10;

            const int gpr_size = 8;
            // save reg_tmp state as we are not allowed to spoil it.
            h->sub(h->rsp, gpr_size);
            h->mov(h->ptr[h->rsp], reg_tmp);

            // rest of code puts indices on stack, fetching a table number based
            // on an index, replaces index with the value, and, finally, moves
            // fetched values into vector register.
            h->sub(h->rsp, vlen);
            h->uni_vmovups(h->ptr[h->rsp], vmm_idxs);

            for (size_t i = 0; i < vlen / sizeof(float); ++i) {
                h->mov(reg_tmp.cvt32(), h->ptr[h->rsp + i * sizeof(float)]);
                h->shl(reg_tmp.cvt32(), 2); // multiply by simd_w
                table_idx = h->ptr[p_table + table_start_idx + offt + reg_tmp];
                h->mov(reg_tmp.cvt32(), table_idx);
                h->mov(h->ptr[h->rsp + i * sizeof(float)], reg_tmp.cvt32());
            }

            h->uni_vmovups(vmm_dst, h->ptr[h->rsp]);
            h->add(h->rsp, vlen);
            // restore GPR state
            h->mov(reg_tmp, h->ptr[h->rsp]);
            h->add(h->rsp, gpr_size);
        }
    };

    // get r_i, same as table(i)
    gather_table_values(vmm_aux2, vmm_aux1, 0);

    // compute relative error (rel_err = m * r_i - 1)
    h->uni_vfmsub213ps(vmm_aux2, vmm_src, table_val(one));

    // compute polynomial(rel_err)
    h->uni_vmovups(vmm_src, table_val(log_pol, 3));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 2));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 1));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(log_pol, 0));
    h->uni_vfmadd213ps(vmm_src, vmm_aux2, table_val(one));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux2);

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux2, vmm_aux1, vlen);

    // compute partial result (pres = E * ln(2) - log(r_i))
    h->uni_vfmadd231ps(vmm_aux2, vmm_aux3, table_val(ln2f));

    // compute (result = polynomial + pres) w/ TwoSum algorithm
    // TODO: restore this instead of version below when asserts are gone
    // h->uni_vaddps(vmm_aux1, vmm_src, vmm_aux2); // res_hi = pol + pres
    // h->uni_vsubps(vmm_aux3, vmm_aux1, vmm_aux2); // res_lo = res_hi - pres
    // h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    // h->uni_vaddps(vmm_src, vmm_aux1, vmm_aux3); // res_hi = pol + pres

    h->uni_vmovups(vmm_aux1, vmm_src);
    h->uni_vaddps(vmm_aux1, vmm_aux1, vmm_aux2); // res_hi = pol + pres
    h->uni_vmovups(vmm_aux3, vmm_aux1);
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_aux2); // res_lo = res_hi - pres
    h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    h->uni_vmovups(vmm_src, vmm_aux1);
    h->uni_vaddps(vmm_src, vmm_src, vmm_aux3); // res_hi = pol + pres

    // Check original source for zero and neg values. skip blend w/ extreme
    // values if all src values were positive.
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    Xbyak::Label end_log_label;
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_le_os);

#ifndef DNNL_INDIRECT_JIT_AARCH64
    std::cout << "hoge:" << __LINE__ << std::endl;
    test_mask();
#endif
    /*
#else
    h->CodeGeneratorAArch64::orrs(h->P_TMP_0.b,
            h->P_ALL_ONE / Xbyak_aarch64::T_z,
            Xbyak_aarch64::PRegB(k_mask.getIdx()),
            Xbyak_aarch64::PRegB(k_mask.getIdx()));
#endif
    */
    h->jz(end_log_label);

    // Blend extreme values into src if reach here.
    // First zero for -inf values...
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(log_minus_inf));

    // ...then negative for qnan values.
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(log_qnan));
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_fwd(
       const xa::ZRegS &vmm_src) {
    // From J.-M. Muller and others, Handbook of Floating-Point Arithmetic, 2010
    // Here is a brief mathematics to approximate log(x):
    // log(x) = E * log(2) + log(y), where -log(2)/2 <= log(y) <= log(2)/2;
    // log(y) = log(1 + z) - log(r_i), where z = y * r_i - 1, r_i approximates
    //   1 / y, i is index of one of precomputed values;
    // log(1 + z) ~~ polynomial(z), =>
    // if (x is normal)
    //     log(x) ~~ E * log(2) + polynomial(z) - log(r_i),
    // where log(r_i) is table value.
    //
    // If (x == 0) result = -inf;
    // If (x < 0) result = qnan;

    // save source on stack to check neg and zero values at the end
    CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));

    // compute i
    const int approx_order = 5;
    CG::lsr(vmm_aux1, vmm_src, n_mantissa_bits - approx_order);
    CG::and_(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_aux1)),
            xa::ZRegD(IDX(table_val(log_five_bit_offset))));
    CG::lsl(vmm_aux1, vmm_aux1,
            1); // multiply i by 2

    // compute anticancellation i
    CG::lsr(vmm_aux2, vmm_aux1, approx_order);

    // get E, don't care about sign as only positive numbers are considered
    CG::lsr(vmm_aux3, vmm_src, n_mantissa_bits);
    CG::add(vmm_aux3, vmm_aux3, vmm_aux2);
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::scvtf(vmm_aux3, p_tmp0 / xa::T_m, vmm_aux3);

    // get m (mantissa)
    CG::eor(xa::ZRegD(IDX(vmm_aux2)), xa::ZRegD(IDX(vmm_aux2)),
            xa::ZRegD(IDX(table_val(exponent_bias))));
    CG::lsl(vmm_aux2, vmm_aux2, n_mantissa_bits);
    CG::and_(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(log_mantissa_mask))));
    CG::orr(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux2)));

    // At first, adjust indices for table structure which broadcasts elements
    CG::lsl(vmm_aux1, vmm_aux1,
            4); // multiply by simd_w = 16

    const auto it = entry_map_.find(log_predefined_vals);
    assert(it != entry_map_.end());
    const auto table_start_idx = (*it).second.off;

    auto gather_table_values = [&](const xa::ZRegS &vmm_dst, const xa::ZRegS &vmm_idxs,
                                       size_t offt = 0) {
        CG::ptrue(xa::PRegS(IDX(p_mask)), xa::VL16);
        CG::add_imm(
                X_DEFAULT_ADDR, x_table, table_start_idx + offt, X_TMP_1);

        CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(vmm_idxs)));
        CG::mul(z_tmp, 4);

        CG::ld1w(z_tmp, p_mask / xa::T_z, xa::ptr(X_DEFAULT_ADDR, z_tmp, xa::SXTW));
        CG::mov(vmm_dst, p_mask / xa::T_m, z_tmp);
        CG::pfalse(xa::PRegB(IDX(p_mask)));
    };

    // get r_i, same as table(i)
    gather_table_values(vmm_aux2, vmm_aux1, 0);

    // compute relative error (rel_err = m * r_i - 1)
    /* [info]Expand from the content of the process, not from the instruction. */
    CG::fmul(vmm_aux2, vmm_aux2, vmm_src);
    CG::fsub(vmm_aux2, vmm_aux2, xa::ZRegS(IDX(table_val(one))));

    // compute polynomial(rel_err)
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(log_pol, 3))));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux2, xa::ZRegS(IDX(table_val(log_pol, 2))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux2, xa::ZRegS(IDX(table_val(log_pol, 1))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux2, xa::ZRegS(IDX(table_val(log_pol, 0))));
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux2, xa::ZRegS(IDX(table_val(one))));
    CG::fmul(vmm_src, vmm_src, vmm_aux2);

    // get log(r_i) = table(i+1)
    gather_table_values(vmm_aux2, vmm_aux1, vlen);

    // compute partial result (pres = E * ln(2) - log(r_i))
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmla(vmm_aux2, p_tmp0 / xa::T_m, vmm_aux3, xa::ZRegS(IDX(table_val(ln2f))));

    // compute (result = polynomial + pres) w/ TwoSum algorithm
    // TODO: restore this instead of version below when asserts are gone
    // h->uni_vaddps(vmm_aux1, vmm_src, vmm_aux2); // res_hi = pol + pres
    // h->uni_vsubps(vmm_aux3, vmm_aux1, vmm_aux2); // res_lo = res_hi - pres
    // h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src); // res_lo = res_lo - pol
    // h->uni_vaddps(vmm_src, vmm_aux1, vmm_aux3); // res_hi = pol + pres

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    CG::fadd(vmm_aux1, vmm_aux1, vmm_aux2);
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_aux1)));
    CG::mov(vmm_aux3, p_tmp0 / xa::T_m, 0);
    CG::fsub(vmm_aux3, vmm_aux3,
            vmm_aux2); // res_lo = res_hi - pres
    CG::fsub(vmm_aux3, vmm_aux3,
            vmm_src); // res_lo = res_lo - pol
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux1)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    CG::fadd(vmm_src, vmm_src, vmm_aux3);

    // Check original source for zero and neg values. skip blend w/ extreme
    // values if all src values were positive.
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux1)), xa::ptr(X_TMP_0));
    CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);

    Label end_log_label;
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_le_os);

    CG::orrs(P_TMP_0.b, P_ALL_ONE / xa::T_z,
            xa::PRegB(p_mask.getIdx()),
            xa::PRegB(p_mask.getIdx()));

    CG::b(xa::EQ, end_log_label);

    // Blend extreme values into src if reach here.
    // First zero for -inf values...
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
    blend_with_mask(vmm_src, table_val(log_minus_inf));

    // ...then negative for qnan values.
    compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(log_qnan));

#endif /* DNNL_X64_IMPLEMENTATION */
    h->L(end_log_label);
}


#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_fwd(
        const Vmm &vmm_src) {
    // dispatch between special cases.
    if (beta_ == -1) { // alpha / x
        h->uni_vmovups(vmm_aux0, table_val(alpha));
        h->uni_vdivps(vmm_src, vmm_aux0, vmm_src, vmm_aux0);
    } else if (beta_ == 0) { // alpha
        h->uni_vmovups(vmm_src, table_val(alpha));
    } else if (beta_ == 0.5) { // alpha * sqrt(x)
        sqrt_compute_vector_fwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 1) { // alpha * x
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 2) { // alpha * x^2
        square_compute_vector_fwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else { // general path
        // caller obligation to save gprs as callee may use them
        size_t gpr_size = 8;
        Xbyak::Operand gprs_to_save[] = {h->r8, h->r9, h->r10, h->r11, h->rax,
                h->rcx, h->rdx, h->rdi, h->rsi, h->rbp, h->rbx};
        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        h->sub(h->rsp, n_gprs_to_save * gpr_size);
        for (size_t i = 0; i < n_gprs_to_save; ++i)
            h->mov(h->ptr[h->rsp + i * gpr_size], gprs_to_save[i]);

        // caller obligation to save k-regs as callee may use them
        size_t n_k_regs_to_save = 8;
        if (has_avx512()) {
            h->sub(h->rsp, n_k_regs_to_save * k_mask_size);
            for (size_t i = 0; i < n_k_regs_to_save; ++i) {
	      //#ifdef DNNL_X64_IMPLEMENTATION
                if (mayiuse(avx512_core))
                    h->kmovq(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
                else
                    h->kmovw(h->ptr[h->rsp + i * k_mask_size], Opmask(i));
		/*
#else //#ifdef DNNL_X64_IMPLEMENTATION
                CG::add_imm(h->X_TMP_0, xa::XReg {IDX(h->rsp)}, i * k_mask_size,
                        h->X_TMP_1);
                CG::str(xa::PReg {static_cast<uint32_t>(i)},
                        xa::ptr(h->X_TMP_0));
#endif //#ifdef DNNL_X64_IMPLEMENTATION
		*/
            }
        }

        // 1. Caller obligation to save vector registers as callee may use them.
        // 2. Additionally save space for vmm_src, to put the answer in-place on
        // this space and space for beta.
        // 3. There is an implicit assumption that the host code uses the same
        // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
        // `vlen` should be replaced with `host_isa::vlen` and
        // `host_isa::vecs_count`.
        h->sub(h->rsp, (vecs_count + 2) * vlen);
        for (size_t i = 2; i < vecs_count + 2; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen], Vmm(i - 2));
        h->uni_vmovups(h->ptr[h->rsp + 0 * vlen], vmm_src); // src
        h->uni_vmovups(vmm_src, table_val(beta));
        h->uni_vmovups(h->ptr[h->rsp + 1 * vlen], vmm_src); // beta

        // save function address in gpr to pass in in call instruction
        h->mov(h->rbp, reinterpret_cast<uintptr_t>(powf));

        // align stack on 16-byte as ABI requires
        h->mov(h->rbx, h->rsp);
        h->and_(h->rbx, 0xf);
        h->sub(h->rsp, h->rbx);

        // Take src, apply powf on it and replace value on a stack with dst.
        Xmm xmm0 = Xmm(0), xmm1 = Xmm(1);
        for (size_t i = 0; i < vlen / sizeof(float); ++i) {
            const Address &source = h->ptr[h->rsp + h->rbx + i * sizeof(float)];
            h->uni_vmovss(xmm0, source);
            h->uni_vmovss(xmm1, h->ptr[h->rsp + h->rbx + vlen]); // beta
            h->call(h->rbp);
            h->uni_vmovss(source, xmm0);
        }

        h->add(h->rsp, h->rbx);

        // restore vector registers
        for (size_t i = vecs_count + 1; i >= 2; --i)
            h->uni_vmovups(Vmm(i - 2), h->ptr[h->rsp + i * vlen]);
        h->uni_vmovups(vmm_src, h->ptr[h->rsp + 0 * vlen]);
        h->add(h->rsp, (vecs_count + 2) * vlen);

        // restore k registers
        if (has_avx512()) {
            for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
	      //#ifdef DNNL_X64_IMPLEMENTATION
                if (mayiuse(avx512_core))
                    h->kmovq(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
                else
                    h->kmovw(Opmask(i), h->ptr[h->rsp + i * k_mask_size]);
		/*
#else //#ifdef DNNL_X64_IMPLEMENTATION
                CG::add_imm(h->X_TMP_0, xa::XReg {IDX(h->rsp)}, i * k_mask_size,
                        h->X_TMP_1);
                CG::ldr(xa::PReg {static_cast<uint32_t>(i)},
                        xa::ptr(h->X_TMP_0));
#endif //#ifdef DNNL_X64_IMPLEMENTATION
		*/
            }
            h->add(h->rsp, n_k_regs_to_save * k_mask_size);
        }

        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i)
            h->mov(gprs_to_save[i], h->ptr[h->rsp + i * gpr_size]);
        h->add(h->rsp, n_gprs_to_save * gpr_size);

        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    }
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_fwd(
	       const xa::ZRegS &vmm_src) {
    // dispatch between special cases.
    if (beta_ == -1) { // alpha / x
        CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
        CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(table_val(alpha))));
        CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);

        CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(vmm_src)));
        CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));
        CG::fdiv(vmm_src, p_512 / xa::T_m, z_tmp);
    } else if (beta_ == 0) { // alpha
        CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
        CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(alpha))));
        CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    } else if (beta_ == 0.5) { // alpha * sqrt(x)
        sqrt_compute_vector_fwd(vmm_src);
        CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    } else if (beta_ == 1) { // alpha * x
        CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    } else if (beta_ == 2) { // alpha * x^2
        square_compute_vector_fwd(vmm_src);
        CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    } else { // general path
        // caller obligation to save gprs as callee may use them
        size_t gpr_size = 5;
        Xbyak_aarch64::XReg gprs_to_save[]
	  = {xa::XReg(8), xa::XReg(9), xa::XReg(10), xa::XReg(11), xa::XReg(0)};

        size_t n_gprs_to_save = sizeof(gprs_to_save) / sizeof(gprs_to_save[0]);

        CG::sub_imm(X_SP, X_SP, n_gprs_to_save * gpr_size, X_TMP_0);
        for (size_t i = 0; i < n_gprs_to_save; ++i) {
            CG::add_imm(X_TMP_0, X_SP, i * gpr_size, X_TMP_1);
            CG::str(xa::XReg(IDX(gprs_to_save[i])), xa::ptr(X_TMP_0));
        }

        // caller obligation to save k-regs as callee may use them
        size_t n_k_regs_to_save = 8;
        CG::sub_imm(
                X_SP, X_SP, n_k_regs_to_save * k_mask_size, X_TMP_0);
        for (size_t i = 0; i < n_k_regs_to_save; ++i) {
            CG::add_imm(X_TMP_0, X_SP, i * k_mask_size, X_TMP_1);
            CG::str(xa::PReg(static_cast<uint32_t>(i)), xa::ptr(X_TMP_0));
        }

        // 1. Caller obligation to save vector registers as callee may use them.
        // 2. Additionally save space for vmm_src, to put the answer in-place on
        // this space and space for beta.
        // 3. There is an implicit assumption that the host code uses the same
        // `isa` as the injector. Once the assumption is wrong, `vecs_count` and
        // `vlen` should be replaced with `host_isa::vlen` and
        // `host_isa::vecs_count`.
        CG::sub_imm(X_SP, X_SP, (vecs_count + 2) * vlen, X_TMP_0);

        for (size_t i = 2; i < vecs_count + 2; ++i) {
            CG::add_imm(X_TMP_0, X_SP, i * vlen, X_TMP_1);
            CG::str(xa::ZReg(IDX(xa::ZRegS(i - 2))), xa::ptr(X_TMP_0));
        }

        CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
        CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));

        CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
        CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(beta))));
        CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);

        CG::add_imm(X_TMP_0, X_SP, vlen, X_TMP_1);
        CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));

        // save function address in gpr to pass in in call instruction
        CG::mov_imm(xa::XReg(0), reinterpret_cast<uintptr_t>(powf));

        // align stack on 16-byte as ABI requires
        CG::mov(xa::XReg(0), X_SP);

        uint64_t mask = ~uint64_t(0xffffffff);
        unsigned bits = (mask & 0xf) ? 64 : 32;
        CG::mov_imm(X_TMP_0, int64_t(bits));
        CG::and_(xa::XReg(0), xa::XReg(0), X_TMP_0);

        CG::sub(X_SP, X_SP, xa::XReg(0));

        // Take src, apply powf on it and replace value on a stack with dst.
        xa::VReg xmm0 = xa::VReg(0), xmm1 = xa::VReg(1);
        for (size_t i = 0; i < vlen / sizeof(float); ++i) {
            CG::add_imm(X_TMP_0, X_SP, i * sizeof(float), X_TMP_1);
            CG::add(X_TMP_0, X_TMP_0, xa::XReg(0));
            CG::ld1(xa::VReg(IDX(xmm0)).s[0], xa::ptr(X_TMP_0));
            CG::mov(z_tmp, 0);
            for (int ii = 1; ii < 4; ii++) {
                CG::mov(xa::VReg(IDX(xmm0)).s[ii], xa::VReg(IDX(z_tmp)).s[0]);
            }
            // beta
            CG::add_imm(X_TMP_0, X_SP, vlen, X_TMP_1);
            CG::add(X_TMP_0, X_TMP_0, xa::XReg(0));
            CG::ld1(xa::VReg(IDX(xmm1)).s[0], xa::ptr(X_TMP_0));
            CG::mov(z_tmp, 0);
            for (int ii = 1; ii < 4; ii++) {
                CG::mov(xa::VReg(IDX(xmm1)).s[ii], xa::VReg(IDX(z_tmp)).s[0]);
            }

            CG::br(xa::XReg(0));

            CG::add_imm(X_TMP_0, X_SP, i * sizeof(float), X_TMP_1);
            CG::add(X_TMP_0, X_TMP_0, xa::XReg(0));
            CG::st1(xa::VReg(IDX(xmm0)).s[0], xa::ptr(X_TMP_0));
        }

        CG::add(X_SP, X_SP, xa::XReg(0));

        // restore vector registers
        for (size_t i = vecs_count + 1; i >= 2; --i) {
            CG::add_imm(X_TMP_0, X_SP, i * vlen, X_TMP_1);
            CG::ldr(xa::ZReg(IDX(xa::ZRegS(i - 2))), xa::ptr(X_TMP_0));
        }

        CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
        CG::ldr(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));

        CG::add_imm(X_SP, X_SP, (vecs_count + 2) * vlen, X_TMP_0);
        // restore k registers

        for (int i = n_k_regs_to_save - 1; i >= 0; --i) {
            CG::add_imm(X_TMP_0, X_SP, i * k_mask_size, X_TMP_1);
            CG::ldr(xa::PReg(static_cast<uint32_t>(i)), xa::ptr(X_TMP_0));
        }
        CG::add_imm(
                X_SP, X_SP, n_k_regs_to_save * k_mask_size, X_TMP_0);

        // restore gpr registers
        for (int i = n_gprs_to_save - 1; i >= 0; --i) {
            CG::add_imm(X_TMP_0, X_SP, i * gpr_size, X_TMP_1);
            CG::ldr(gprs_to_save[i], xa::ptr(X_TMP_0));
        }
        CG::add_imm(X_SP, X_SP, n_gprs_to_save * gpr_size, X_TMP_0);
	CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    }
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
        const Vmm &vmm_src) {
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    // x = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_erf_one_over_sqrt_two));

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    h->uni_vmovups(vmm_aux3, vmm_src);

    // -exp(-x*x)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux0, vmm_aux3);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux1, vmm_aux3);
    abs_compute_vector_fwd(vmm_aux1);

    // t = 1 / (p*x + 1)
    h->uni_vmovups(vmm_aux2, table_val(gelu_erf_approx_const));
    h->uni_vfmadd213ps(vmm_aux2, vmm_aux1, table_val(one));
    h->uni_vmovups(vmm_aux4, table_val(one));
    h->uni_vdivps(vmm_aux4, vmm_aux4, vmm_aux2);

    // -exp(-x*x)*t
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    h->uni_vmovups(vmm_aux1, table_val(gelu_erf_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux0);

    // S = 0.5 * s = x / sqrt^2(2)
    h->uni_vmulps(vmm_aux3, vmm_aux3, table_val(gelu_erf_one_over_sqrt_two));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->uni_vfmadd213ps(vmm_src, vmm_aux3, vmm_aux3);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_fwd(
	    const xa::ZRegS &vmm_src) {
    // Here we approximate erf(x) using the expression by
    // Abramowitz and Stegun from ``Handbook of Mathematical
    // Functions''
    // NOTE: The performance of this kernel can be further improved
    // with a minimax polynomialial expansion, thereby avoiding division
    // and exp. However, so far, this has costed larger accuracy
    // differences with respect to glibc erf based GELU, in particular
    // ~1.0e-5 -- 1.0e-3 absolute error at s = -5.

    // x = s / sqrt(2)
    CG::fmul(vmm_src, vmm_src,
            xa::ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two))));

    // IMPORTANT: we use vmm_aux3 to save `x` as exp_compute does not use it.
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux3, p_tmp0 / xa::T_m, 0);

    // -exp(-x*x)
    CG::fmul(vmm_src, vmm_src, vmm_src);
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(sign_mask))));

    exp_compute_vector_fwd(vmm_src);
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(sign_mask))));

    // get sign
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(vmm_aux3)));
    CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);
    CG::and_(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(vmm_aux0)),
            xa::ZRegD(IDX(table_val(sign_mask))));

    // abs(x)
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(vmm_aux3)));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    abs_compute_vector_fwd(vmm_aux1);

    // t = 1 / (p*x + 1)
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux2)), xa::ZRegD(IDX(table_val(gelu_erf_approx_const))));
    CG::mov(vmm_aux2, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_aux2, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux4)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_aux4, p_tmp0 / xa::T_m, 0);
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE, P_ALL_ONE.b);
    CG::fdiv(vmm_aux4, p_tmp0, vmm_aux2);

    // -exp(-x*x)*t
    CG::fmul(vmm_src, vmm_src, vmm_aux4);

    // compute polynomialial r
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(table_val(gelu_erf_pol, 4))));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);

    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 3))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 2))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 1))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));

    // S = 0.5 * s = x / sqrt^2(2)
    CG::fmul(vmm_aux3, vmm_aux3,
            xa::ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two))));
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux3, vmm_aux3);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // invariant to whether `s` or `d` is passed.
    // get mask of `s` > 0
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    // fill with alpha, then blend with 1.f
    h->uni_vmovups(vmm_src, table_val(alpha));
    blend_with_mask(vmm_src, table_val(one));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::relu_compute_vector_bwd(
	const xa::ZRegS &vmm_src) {
    // invariant to whether `s` or `d` is passed.
    // get mask of `s` > 0
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    // fill with alpha, then blend with 1.f
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(alpha))));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    blend_with_mask(vmm_src, table_val(one));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
        const Vmm &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
        h->uni_vaddps(vmm_src, vmm_src, table_val(alpha));
    }
    blend_with_mask(vmm_src, table_val(one));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::elu_compute_vector_bwd(
      const xa::ZRegS &vmm_src) {
    if (!use_dst_) {
        // R = exp(s)
        exp_compute_vector_fwd(vmm_src);
        // after exponentiation, get mask by comparing with exp(0)=1.f, not 0.f
        compute_cmp_mask(vmm_src, table_val(one), _cmp_gt_os);
        // R * alpha, then blend with 1.f
        CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    } else {
        // get mask of `d` > 0
        compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
        // R = `d` + alpha, then blend with 1.f
	CG::fadd(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    }
    blend_with_mask(vmm_src, table_val(one));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, table_val(one));
    h->uni_vfnmadd231ps(vmm_aux0, vmm_src, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::tanh_compute_vector_bwd(
	const xa::ZRegS &vmm_src) {
    // res = 1 - d^2 = 1 - tanh^2(s)
    if (!use_dst_) tanh_compute_vector_fwd(vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(table_val(one))));

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmls(vmm_aux0, p_tmp0 / xa::T_m, vmm_src, vmm_src);

    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_aux0, vmm_src);

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    h->uni_vmovups(vmm_aux2, table_val(gelu_tanh_fitting_const_times_three));
    h->uni_vfmadd213ps(vmm_aux2, vmm_src, table_val(one));

    h->uni_vmovups(vmm_aux1, table_val(gelu_tanh_fitting_const));
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vmulps(vmm_aux0, vmm_aux0, table_val(gelu_tanh_sqrt_two_over_pi));
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_aux2);

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    if (isa == sse41) {
        h->uni_vmovups(vmm_aux3, table_val(one));
        h->uni_vsubps(vmm_aux3, vmm_aux3, vmm_src);
        h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_aux3);
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_src);
        h->uni_vaddps(vmm_src, vmm_src, vmm_aux2);
    } else {
        // 1) R = G2 * (1 - T) = G2 - G2 * T
        h->uni_vfnmadd231ps(vmm_aux2, vmm_aux2, vmm_src);
        // 2) Q = 1 + T
        h->uni_vaddps(vmm_src, vmm_src, table_val(one));
        // 3) res = Q * (1 + R) = Q + Q * R
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux2);
    }
    h->uni_vmulps(vmm_src, vmm_src, table_val(half));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_tanh_compute_vector_bwd(
	     const xa::ZRegS &vmm_src) {
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(vmm_src)));
    CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);

    // compute G1(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x^2)
    // compute G2(x) = sqrt_root_two_over_pi * x * (1 + 3 * fitting_const * x^2)
    CG::fmul(vmm_src, vmm_src, vmm_src);

    // keep G2 in a separate register
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux2)),
            xa::ZRegD(IDX(table_val(gelu_tanh_fitting_const_times_three))));
    CG::mov(vmm_aux2, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_aux2, p_512 / xa::T_m, vmm_src, xa::ZRegS(IDX(table_val(one))));

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)),
            xa::ZRegD(IDX(table_val(gelu_tanh_fitting_const))));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));
    CG::fmul(vmm_aux0, vmm_aux0,
            xa::ZRegS(IDX(table_val(gelu_tanh_sqrt_two_over_pi))));
    CG::fmul(vmm_src, vmm_src, vmm_aux0);
    CG::fmul(vmm_aux2, vmm_aux2, vmm_aux0);

    // save G2 on stack as tanh uses all available registers
    CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::str(xa::ZReg(IDX(vmm_aux2)), xa::ptr(X_TMP_0));

    // T = tanh(G1(x))
    tanh_compute_vector_fwd(vmm_src);

    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux2)), xa::ptr(X_TMP_0));
    CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);

    // compute 0.5 * (1 + T) * (1 + G2 * (1 - T))
    // 1) R = G2 * (1 - T) = G2 - G2 * T
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmls(vmm_aux2, p_tmp0 / xa::T_m, vmm_aux2, vmm_src);
    // 2) Q = 1 + T
    CG::fadd(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(one))));
    // 3) res = Q * (1 + R) = Q + Q * R
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmla(vmm_src, p_tmp0 / xa::T_m, vmm_src, vmm_aux2);

    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(half))));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 2 * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(two));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::square_compute_vector_bwd(
	  const xa::ZRegS &vmm_src) {
    // res = 2 * s
  CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(two))));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
        const Vmm &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::abs_compute_vector_bwd(
       const xa::ZRegS &vmm_src) {
    // replace positive values with 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
    // replace negative values with -1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_lt_os);
    blend_with_mask(vmm_src, table_val(minus_one));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, table_val(half));
    // h->uni_vdivps(vmm_src, vmm_aux0, vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::sqrt_compute_vector_bwd(
		const xa::ZRegS &vmm_src) {
    // res = 0.5 / d = 0.5 / sqrt(s)
    if (!use_dst_) sqrt_compute_vector_fwd(vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(table_val(half))));

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE, P_ALL_ONE.b);
    CG::fdiv(vmm_aux0, p_tmp0, vmm_src);

    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
        const Vmm &vmm_src) {
    h->uni_vmovups(vmm_src, table_val(alpha));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::linear_compute_vector_bwd(
	  const xa::ZRegS &vmm_src) {
  CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(alpha))));
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_bwd(
        const Vmm &vmm_src) {
    // get mask of values > alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(zero));
    // make all negative values zeros
    h->uni_vmaxps(vmm_src, vmm_src, table_val(zero));
    // everything bigger than 0.f should be 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::bounded_relu_compute_vector_bwd(
	const xa::ZRegS &vmm_src) {
    // get mask of values > alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(zero));
    // make all negative values zeros
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::mov(xa::ZRegD(IDX(z_tmp)), xa::ZRegD(IDX(table_val(zero))));
    CG::fmaxnm(z_tmp, p_tmp0, vmm_src);
    CG::fmax(z_tmp, p_tmp0, vmm_src);
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(z_tmp)));

    // everything bigger than 0.f should be 1.f
    compute_cmp_mask(vmm_src, table_val(zero), _cmp_gt_os);
    blend_with_mask(vmm_src, table_val(one));
}
#endif /* DNNL_X64_IMPLEMENTATION */

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::soft_relu_compute_vector_bwd(
#ifdef DNNL_X64_IMPLEMENTATION
        const Vmm &vmm_src) {
#else /* DNNL_X64_IMPLEMENTATION */
        const xa::ZRegS &vmm_src) {
#endif /* DNNL_X64_IMPLEMENTATION */
    logistic_compute_vector_fwd(vmm_src);
}

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    // h->uni_vfnmadd231ps(vmm_src, vmm_src, vmm_src); // bless sse41
    h->uni_vmovups(vmm_aux0, table_val(one));
    h->uni_vsubps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux0);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::logistic_compute_vector_bwd(
	    const xa::ZRegS &vmm_src) {
    // res = d * (1 - d) = d - d * d; d = logistic(s)
    if (!use_dst_) logistic_compute_vector_fwd(vmm_src);
    // h->uni_vfnmadd231ps(vmm_src, vmm_src, vmm_src); // bless sse41
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);

    CG::fsub(vmm_aux0, vmm_aux0, vmm_src);

    CG::fmul(vmm_src, vmm_src, vmm_aux0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::exp_compute_vector_bwd(
#ifdef DNNL_X64_IMPLEMENTATION
        const Vmm &vmm_src) {
#else /* DNNL_X64_IMPLEMENTATION */
        const xa::ZRegS &vmm_src) {
#endif /* DNNL_X64_IMPLEMENTATION */
    if (!use_dst_) exp_compute_vector_fwd(vmm_src);
}

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = alpha * s
    h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    // Save R on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);
    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    // compute Q * (1 + R * (1 - Q))
    if (isa == sse41) {
        h->uni_vmovups(vmm_aux1, table_val(one));
        h->uni_vsubps(vmm_aux1, vmm_aux1, vmm_src);
        h->uni_vmulps(vmm_aux1, vmm_aux1, vmm_aux0);
        h->uni_vaddps(vmm_aux1, vmm_aux1, table_val(one));
        h->uni_vmulps(vmm_src, vmm_src, vmm_aux1);
    } else {
        // T = R * (1 - Q) = R - R * Q
        h->uni_vfnmadd231ps(vmm_aux0, vmm_aux0, vmm_src);
        // Q * (1 + T) = Q + Q * T
        h->uni_vfmadd231ps(vmm_src, vmm_src, vmm_aux0);
    }
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::swish_compute_vector_bwd(
	 const xa::ZRegS &vmm_src) {
    // R = alpha * s
    CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));

    // Save R on stack for later usage
    CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);

    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));

    // Q = sigmoid(alpha * s)
    logistic_compute_vector_fwd(vmm_src);

    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux0)), xa::ptr(X_TMP_0));

    CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);

    // compute Q * (1 + R * (1 - Q))
    // T = R * (1 - Q) = R - R * Q
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmls(vmm_aux0, p_tmp0 / xa::T_m, vmm_aux0, vmm_src);

    // Q * (1 + T) = Q + Q * T
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmla(vmm_src, p_tmp0 / xa::T_m, vmm_src, vmm_aux0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
        const Vmm &vmm_src) {
    // res = 1 / s
    h->uni_vmovups(vmm_aux0, table_val(one));
    // h->uni_vdivps(vmm_src, vmm_aux0, vmm_src); // bless sse41
    h->uni_vdivps(vmm_aux0, vmm_aux0, vmm_src);
    h->uni_vmovups(vmm_src, vmm_aux0);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::log_compute_vector_bwd(
	       const xa::ZRegS &vmm_src) {
    // res = 1 / s
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_aux0, p_tmp0 / xa::T_m, 0);

    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE, P_ALL_ONE.b);
    CG::fdiv(vmm_aux0, p_tmp0, vmm_src);

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
        const Vmm &vmm_src) {
    // set result with 1.f
    h->uni_vmovups(vmm_aux1, table_val(one));
    // get mask of values > beta and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(beta), _cmp_gt_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    // get mask of values <= alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_le_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    h->uni_vmovups(vmm_src, vmm_aux1);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::clip_compute_vector_bwd(
	const xa::ZRegS &vmm_src) {
    // set result with 1.f
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);

    // get mask of values > beta and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(beta), _cmp_gt_os);
    blend_with_mask(vmm_aux1, table_val(zero));
    // get mask of values <= alpha and blend with 0.f
    compute_cmp_mask(vmm_src, table_val(alpha), _cmp_le_os);
    blend_with_mask(vmm_aux1, table_val(zero));

    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux1)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_bwd(
        const Vmm &vmm_src) {
    // dispatch some special cases.
    if (beta_ == 0) { // zero
        h->uni_vmovups(vmm_src, table_val(zero));
    } else if (beta_ == 0.5) { // 0.5 * alpha / sqrt(s)
        sqrt_compute_vector_bwd(vmm_src);
        h->uni_vmulps(vmm_src, vmm_src, table_val(alpha));
    } else if (beta_ == 1) { // alpha
        h->uni_vmovups(vmm_src, table_val(alpha));
    } else {
        // Save `s` on stack for later usage
        h->sub(h->rsp, vlen);
        h->uni_vmovups(h->ptr[h->rsp], vmm_src);
        // R = alpha * pow(s, beta)
        pow_compute_vector_fwd(vmm_src);
        // Restore `s` from stack
        h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
        h->add(h->rsp, vlen);
        // Save mask of zero elements to convert them into zeros at the end
        if (beta_ >= 1) compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
        // res = alpha * beta * pow(s, beta - 1) = beta * R / s;
        h->uni_vdivps(vmm_src, vmm_src, vmm_aux1);
        h->uni_vmulps(vmm_src, vmm_src, table_val(beta));

        // beta < 1 leads to NaN as `s` appears in denominator, but beta >= 1
        // should lead to zero, when `s` is zero.
        if (beta_ >= 1) blend_with_mask(vmm_src, table_val(zero));
    }
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::pow_compute_vector_bwd(
      const xa::ZRegS &vmm_src) {
    // dispatch some special cases.
    if (beta_ == 0) { // zero
        /* This route has not been tested */
        CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
        CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(zero))));
        CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    } else if (beta_ == 0.5) { // 0.5 * alpha / sqrt(s)
        sqrt_compute_vector_bwd(vmm_src);
        CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(alpha))));
    } else if (beta_ == 1) { // alpha
        CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
        CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(table_val(alpha))));
        CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
    } else {
        // Save `s` on stack for later usage
        CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);
        CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
        CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));
        // R = alpha * pow(s, beta)
        pow_compute_vector_fwd(vmm_src);
        // Restore `s` from stack
        CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
        CG::ldr(xa::ZReg(IDX(vmm_aux1)), xa::ptr(X_TMP_0));
        CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);
        // Save mask of zero elements to convert them into zeros at the end
        if (beta_ >= 1) compute_cmp_mask(vmm_aux1, table_val(zero), _cmp_eq_oq);
        // res = alpha * beta * pow(s, beta - 1) = beta * R / s;
        CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE, P_ALL_ONE.b);
        CG::fdiv(vmm_src, p_tmp0, vmm_aux1);
	CG::fmul(vmm_src, vmm_src, xa::ZRegS(IDX(table_val(beta))));

        // beta < 1 leads to NaN as `s` appears in denominator, but beta >= 1
        // should lead to zero, when `s` is zero.
        if (beta_ >= 1) blend_with_mask(vmm_src, table_val(zero));
    }
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
        const Vmm &vmm_src) {
    // R = s / sqrt(2)
    h->uni_vmulps(vmm_src, vmm_src, table_val(gelu_erf_one_over_sqrt_two));

    // Save R on stack for later usage
    h->sub(h->rsp, vlen);
    h->uni_vmovups(h->ptr[h->rsp], vmm_src);

    // Q = exp(-R*R)
    h->uni_vmulps(vmm_src, vmm_src, vmm_src);
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    h->uni_vmovups(vmm_aux2, h->ptr[h->rsp]);
    h->uni_vmulps(vmm_aux2, vmm_aux2, table_val(gelu_erf_one_over_sqrt_pi));
    h->uni_vmulps(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    h->uni_vxorps(vmm_src, vmm_src, table_val(sign_mask));

    // get sign
    h->uni_vmovups(vmm_aux0, h->ptr[h->rsp]);
    h->uni_vandps(vmm_aux0, vmm_aux0, table_val(sign_mask));

    // abs(x)
    h->uni_vmovups(vmm_aux1, h->ptr[h->rsp]);
    h->add(h->rsp, vlen);
    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    h->uni_vmovups(vmm_aux3, table_val(gelu_erf_approx_const));
    h->uni_vmovups(vmm_aux4, table_val(one));
    h->uni_vfmadd213ps(vmm_aux3, vmm_aux1, vmm_aux4);
    h->uni_vdivps(vmm_aux4, vmm_aux4, vmm_aux3);

    // Q * W
    h->uni_vmulps(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    h->uni_vmovups(vmm_aux1, table_val(gelu_erf_pol, 4));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 3));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 2));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 1));
    h->uni_vfmadd213ps(vmm_aux1, vmm_aux4, table_val(gelu_erf_pol, 0));

    // erf = sign * (1 - r * t * exp(-x*x))
    h->uni_vfmadd213ps(vmm_src, vmm_aux1, table_val(one));
    h->uni_vxorps(vmm_src, vmm_src, vmm_aux0);

    // P = T + 0.5
    h->uni_vaddps(vmm_aux2, vmm_aux2, table_val(half));
    // res = P + 0.5 * erf
    h->uni_vfmadd231ps(vmm_aux2, vmm_src, table_val(half));
    h->uni_vmovups(vmm_src, vmm_aux2);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::gelu_erf_compute_vector_bwd(
		    const xa::ZRegS &vmm_src) {
    // R = s / sqrt(2)
    CG::fmul(vmm_src, vmm_src,
            xa::ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_two))));

    // Save R on stack for later usage
    CG::sub_imm(X_SP, X_SP, vlen, X_TMP_0);
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::str(xa::ZReg(IDX(vmm_src)), xa::ptr(X_TMP_0));

    // Q = exp(-R*R)
    CG::fmul(vmm_src, vmm_src, vmm_src);
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(sign_mask))));
    exp_compute_vector_fwd(vmm_src);

    // T = R / sqrt(pi) * Q
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux2)), xa::ptr(X_TMP_0));
    CG::fmul(vmm_aux2, vmm_aux2,
            xa::ZRegS(IDX(table_val(gelu_erf_one_over_sqrt_pi))));
    CG::fmul(vmm_aux2, vmm_aux2, vmm_src);

    // -Q
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)),
            xa::ZRegD(IDX(table_val(sign_mask))));

    // get sign
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux0)), xa::ptr(X_TMP_0));
    CG::and_(xa::ZRegD(IDX(vmm_aux0)), xa::ZRegD(IDX(vmm_aux0)),
            xa::ZRegD(IDX(table_val(sign_mask))));

    // abs(x)
    CG::add_imm(X_TMP_0, X_SP, 0, X_TMP_1);
    CG::ldr(xa::ZReg(IDX(vmm_aux1)), xa::ptr(X_TMP_0));
    CG::add_imm(X_SP, X_SP, vlen, X_TMP_0);

    abs_compute_vector_fwd(vmm_aux1);

    // W = 1 / (p * s + 1)
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux3)), xa::ZRegD(IDX(table_val(gelu_erf_approx_const))));
    CG::mov(vmm_aux3, p_tmp0 / xa::T_m, 0);
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux4)), xa::ZRegD(IDX(table_val(one))));
    CG::mov(vmm_aux4, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_aux3, p_512 / xa::T_m, vmm_aux1, vmm_aux4);
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE, P_ALL_ONE.b);
    CG::fdiv(vmm_aux4, p_tmp0, vmm_aux3);

    // Q * W
    CG::fmul(vmm_src, vmm_src, vmm_aux4);

    // compute polynomial r
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_aux1)), xa::ZRegD(IDX(table_val(gelu_erf_pol, 4))));
    CG::mov(vmm_aux1, p_tmp0 / xa::T_m, 0);
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 3))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 2))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 1))));
    CG::fmad(vmm_aux1, p_512 / xa::T_m, vmm_aux4,
            xa::ZRegS(IDX(table_val(gelu_erf_pol, 0))));

    // erf = sign * (1 - r * t * exp(-x*x))
    CG::fmad(vmm_src, p_512 / xa::T_m, vmm_aux1, xa::ZRegS(IDX(table_val(one))));
    CG::eor(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux0)));

    // P = T + 0.5
    CG::fadd(vmm_aux2, vmm_aux2, xa::ZRegS(IDX(table_val(half))));
    // res = P + 0.5 * erf
    CG::mov(xa::PRegB(IDX(p_tmp0)), P_ALL_ONE.b);
    CG::fmla(vmm_aux2, p_tmp0 / xa::T_m, vmm_src, xa::ZRegS(IDX(table_val(half))));
    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(p_512)));
    CG::mov(xa::ZRegD(IDX(vmm_src)), xa::ZRegD(IDX(vmm_aux2)));
    CG::mov(vmm_src, p_tmp0 / xa::T_m, 0);
}
#endif /* DNNL_X64_IMPLEMENTATION */

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_gprs_count() {
    using namespace alg_kind;
    switch (alg_) {
        case eltwise_tanh_use_dst_for_bwd:
        case eltwise_tanh:
#ifdef DNNL_X64_IMPLEMENTATION
        case eltwise_gelu_tanh: return isa == sse41 ? 4 : 0;
#else /* DNNL_X64_IMPLEMENTATION */
        case eltwise_gelu_tanh: return 0;
#endif /* DNNL_X64_IMPLEMENTATION */
        default: return 0;
    }
    return 0;
};

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::round_compute_vector_fwd(
        const Vmm &vmm_src) {
    h->uni_vroundps(vmm_src, vmm_src, _op_mxcsr);
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::round_compute_vector_fwd(
	const xa::ZRegS &vmm_src) {
  CG::frintn(vmm_src, p_512 / xa::T_m, vmm_src);
}
#endif /* DNNL_X64_IMPLEMENTATION */

template <cpu_isa_t isa>
size_t jit_uni_eltwise_injector_f32<isa>::aux_vecs_count() {
      using namespace alg_kind;
      
#ifdef DNNL_X64_IMPLEMENTATION
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 0 : 2;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4;
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 5;
            case eltwise_square: return 0;
            case eltwise_abs: return 0;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 1;
            case eltwise_bounded_relu: return 0;
            case eltwise_soft_relu: return 4;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 4;
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 3;
            case eltwise_gelu_tanh: return 5;
            case eltwise_swish: return 4;
            case eltwise_log: return 5;
            case eltwise_clip: return 0;
            case eltwise_pow: return 2;
            case eltwise_gelu_erf: return 5;
            case eltwise_round: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 1;
            case eltwise_elu_use_dst_for_bwd: return 1;

            case eltwise_elu: return 3;
            case eltwise_tanh_use_dst_for_bwd: return 1;
            case eltwise_tanh: return 5;
            case eltwise_square: return 0;
            case eltwise_abs: return 0;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 1;
            case eltwise_linear: return 0;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 4;
            case eltwise_logistic_use_dst_for_bwd: return 1;
            case eltwise_logistic: return 4;
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 3;
            case eltwise_gelu_tanh: return 5;
            case eltwise_swish: return 4;
            case eltwise_log: return 1;
            case eltwise_clip: return 2;
            case eltwise_pow: return 2;
            case eltwise_gelu_erf: return 5;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
#else /* DNNL_X64_IMPLEMENTATION */
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return (alpha_ == 0.f) ? 1 : 3;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 5; /* = exp + 1 */
            case eltwise_tanh_use_dst_for_bwd:
            case eltwise_tanh: return 9;
            case eltwise_square: return 0;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 0;
            case eltwise_linear: return 2;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 5;
            case eltwise_logistic_use_dst_for_bwd:
            case eltwise_logistic: return 5; /* = exp + 1 */
            case eltwise_exp_use_dst_for_bwd:
            case eltwise_exp: return 4;
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 5;
            case eltwise_clip: return 1;
            case eltwise_pow: return 3;
            case eltwise_gelu_erf: return 6;
            case eltwise_round: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu_use_dst_for_bwd:
            case eltwise_relu: return 2;
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_elu: return 4; /* = exp */
            case eltwise_tanh_use_dst_for_bwd: return 2;
            case eltwise_tanh: return 9;
            case eltwise_square: return 1;
            case eltwise_abs: return 1;
            case eltwise_sqrt_use_dst_for_bwd:
            case eltwise_sqrt: return 2;
            case eltwise_linear: return 1;
            case eltwise_bounded_relu: return 1;
            case eltwise_soft_relu: return 5; /* = logistic */
            case eltwise_logistic_use_dst_for_bwd: return 2;
            case eltwise_logistic: return 5; /* = logistic */
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_exp: return 4; /* = exp */
            case eltwise_gelu_tanh: return 9; /* = tanh */
            case eltwise_swish: return 6; /* = logistic */
            case eltwise_log: return 2;
            case eltwise_clip: return 3;
            case eltwise_pow: return 3;
            case eltwise_gelu_erf: return 6;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
#endif /* DNNL_X64_IMPLEMENTATION */

    return 0;
}

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        size_t start_idx, size_t end_idx) {
    using namespace alg_kind;
    for (size_t idx = start_idx; idx < end_idx; idx++) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
                        relu_zero_ns_compute_vector_fwd(Vmm(idx));
                    else
                        relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_square: square_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_abs: abs_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_linear: linear_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_clip: clip_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_fwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(Vmm(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(Vmm(idx)); break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_square: square_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_abs: abs_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_linear: linear_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(Vmm(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_log: log_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_clip: clip_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_pow: pow_compute_vector_bwd(Vmm(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(Vmm(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
            h->uni_vmulps(Vmm(idx), Vmm(idx), table_val(scale));
        }
    }
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_body(
        const vmm_index_set_iterator_t &start_idx_it,
        const vmm_index_set_iterator_t &end_idx_it) {
    using namespace alg_kind;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        if (is_fwd_) {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu:
                    if (alpha_ == 0.f)
		      relu_zero_ns_compute_vector_fwd(xa::ZRegS(idx));
                    else
                        relu_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_square:
                    square_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_abs: abs_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_swish: swish_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_log: log_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_clip: clip_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_pow: pow_compute_vector_fwd(xa::ZRegS(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_fwd(xa::ZRegS(idx));
                    break;
                case eltwise_round: round_compute_vector_fwd(xa::ZRegS(idx)); break;
                default: assert(!"unsupported eltwise algorithm");
            }
        } else {
            switch (alg_) {
                case eltwise_relu_use_dst_for_bwd:
                case eltwise_relu: relu_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu: elu_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_square:
                    square_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                case eltwise_abs: abs_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_sqrt_use_dst_for_bwd:
                case eltwise_sqrt: sqrt_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_linear:
                    linear_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                case eltwise_bounded_relu:
                    bounded_relu_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                case eltwise_soft_relu:
                    soft_relu_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                    logistic_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp: exp_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_gelu_tanh:
                    gelu_tanh_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                case eltwise_swish: swish_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_log: log_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_clip: clip_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_pow: pow_compute_vector_bwd(xa::ZRegS(idx)); break;
                case eltwise_gelu_erf:
                    gelu_erf_compute_vector_bwd(xa::ZRegS(idx));
                    break;
                default: assert(!"unsupported eltwise algorithm");
            }
        }
        if (scale_ != 1.f) {
	  CG::fmul(xa::ZRegS(IDX(xa::ZRegS(idx))), xa::ZRegS(IDX(xa::ZRegS(idx))),
                    xa::ZRegS(IDX(table_val(scale))));
        }
    });
}
#endif /* DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    assert(start_idx < end_idx && end_idx <= vecs_count);

    injector_preamble(start_idx, end_idx);
    compute_body(start_idx_tail, end_idx);
    injector_preamble_tail(start_idx);
    compute_body(start_idx, start_idx_tail);
    injector_postamble();
}
#else /* DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::compute_vector_range(
        const vmm_index_set_t &vmm_idxs) {
    const auto &start_idx_it = vmm_idxs.begin();
    const auto &end_idx_it = vmm_idxs.end();
    assert(*start_idx_it < *vmm_idxs.rbegin() + 1
            && *vmm_idxs.rbegin() <= vecs_count);

    injector_preamble(vmm_idxs);
    compute_body(start_idx_tail, end_idx_it);
    injector_preamble_tail(start_idx_it);
    compute_body(start_idx_it, start_idx_tail);
    injector_postamble();
}
#endif /* DNNL_X64_IMPLEMENTATION */


template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::prepare_table(bool gen_table) {
    if (!gen_table) return;

    h->align(64);
    h->L(l_table);

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Assumption: iterating on entry_map_ here has the same order as
    // when we set the offsets. We verify that in asserts.
    // table_entry_val_t is assumed to be 32 bits
#ifndef NDEBUG
    size_t off = 0;
    key_t curr_key = undef_key;
    int key_occurences = 0;
#endif

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? vlen : sizeof(table_entry_val_t);
#ifdef DNNL_X64_IMPLEMENTATION
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->CodeGeneratorAArch64::dd(te.val);
#else /* DNNL_X64_IMPLEMENTATION */
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
	  //h->dw(te.val);
	  CG::dd(te.val);
#endif /* DNNL_X64_IMPLEMENTATION */

#ifndef NDEBUG
        // we check that the precomputed offsets match the registered ones
        const auto &key = (*it).first; // get map entry key
        if (key != curr_key) {
            curr_key = key;
            key_occurences = 0;
        }
        key_occurences++;
        auto expected_off = table_off(key, key_occurences - 1);
        assert(off == expected_off);
        MAYBE_UNUSED(expected_off);
        off += len;
#endif
    }
}

template <cpu_isa_t isa>
void jit_uni_eltwise_injector_f32<isa>::register_table_entries() {
    // This function is responsible to pick all necessary constants
    // for a given algorithm, compute right offset for them to be used
    // in table_val() and save the hexadecimal value of them, which
    // will be finally used in prepare_table(). We rely on fact that
    // the map iterator order is deterministic for a fixed map.

    // common values used in several algorithms
    static const table_t common_values {{zero, {0x00000000, true}},
            {half, {0x3f000000, true}}, {one, {0x3f800000, true}},
            {two, {0x40000000, true}}, {minus_one, {0xbf800000, true}},
            {minus_two, {0xc0000000, true}}, {ln2f, {0x3f317218, true}},
            {positive_mask, {0x7fffffff, true}},
            {sign_mask, {0x80000000, true}},
            {exponent_bias, {0x0000007f, true}}};

    // exp(x) constants
    static const table_t exp_consts {{exp_log2ef, {0x3fb8aa3b, true}},
            {exp_ln_flt_max_f, {0x42b17218, true}},
            {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    // exp(x) polynomial approximation
    static const table_t exp_polynomial {
            {exp_pol, {0x3f7ffffb, true}}, // p1 = 0.999999701f
            {exp_pol, {0x3efffee3, true}}, // p2 = 0.499991506f
            {exp_pol, {0x3e2aad40, true}}, // p3 = 0.166676521f
            {exp_pol, {0x3d2b9d0d, true}}, // p4 = 0.0418978221f
            {exp_pol, {0x3c07cfce, true}} // p5 = 0.00828929059f
    };

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts {{tanh_idx_bias, {0x39800000, true}},
            {tanh_idx_mask, {0xffc00000, true}},
            {tanh_linear_ubound, {0x39ddb3d7, true}},
            {tanh_saturation_lbound, {0x41102cb3, true}}};

    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
    static const table_t tanh_polynomial_table {
            // coefficients of degree 0
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x39bfffff, false}},
            {tanh_pol_table, {0x39ffffff, false}},
            {tanh_pol_table, {0x3a3ffffe, false}},
            {tanh_pol_table, {0x3a7ffffb, false}},
            {tanh_pol_table, {0x3abffff7, false}},
            {tanh_pol_table, {0x3affffeb, false}},
            {tanh_pol_table, {0x3b3fffdc, false}},
            {tanh_pol_table, {0x3b7fffab, false}},
            {tanh_pol_table, {0x3bbfff70, false}},
            {tanh_pol_table, {0x3bfffeab, false}},
            {tanh_pol_table, {0x3c3ffdc0, false}},
            {tanh_pol_table, {0x3c7ffaab, false}},
            {tanh_pol_table, {0x3cbff701, false}},
            {tanh_pol_table, {0x3cffeaad, false}},
            {tanh_pol_table, {0x3d3fdc08, false}},
            {tanh_pol_table, {0x3d7faacd, false}},
            {tanh_pol_table, {0x3dbf7081, false}},
            {tanh_pol_table, {0x3dfeacc9, false}},
            {tanh_pol_table, {0x3e3dc7fd, false}},
            {tanh_pol_table, {0x3e7acbf5, false}},
            {tanh_pol_table, {0x3eb77a9f, false}},
            {tanh_pol_table, {0x3eec9a9f, false}},
            {tanh_pol_table, {0x3f22991f, false}},
            {tanh_pol_table, {0x3f42f7d6, false}},
            {tanh_pol_table, {0x3f67b7cc, false}},
            {tanh_pol_table, {0x3f76ca83, false}},
            {tanh_pol_table, {0x3f7ebbe9, false}},
            {tanh_pol_table, {0x3f7fd40c, false}},
            {tanh_pol_table, {0x3f7fff32, false}},
            {tanh_pol_table, {0x3f7ffffc, false}},
            {tanh_pol_table, {0x3f800000, false}},
            // coefficients of degree 1
            {tanh_pol_table, {0x3f800000, false}},
            {tanh_pol_table, {0x3f800018, false}},
            {tanh_pol_table, {0x3f7fffe8, false}},
            {tanh_pol_table, {0x3f7fffda, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffdc, false}},
            {tanh_pol_table, {0x3f7fffac, false}},
            {tanh_pol_table, {0x3f7fff70, false}},
            {tanh_pol_table, {0x3f7ffeec, false}},
            {tanh_pol_table, {0x3f7ffdc0, false}},
            {tanh_pol_table, {0x3f7ffbed, false}},
            {tanh_pol_table, {0x3f7ff704, false}},
            {tanh_pol_table, {0x3f7feff5, false}},
            {tanh_pol_table, {0x3f7fdbca, false}},
            {tanh_pol_table, {0x3f7fbfff, false}},
            {tanh_pol_table, {0x3f7f7041, false}},
            {tanh_pol_table, {0x3f7f009b, false}},
            {tanh_pol_table, {0x3f7dc36c, false}},
            {tanh_pol_table, {0x3f7c0aa8, false}},
            {tanh_pol_table, {0x3f7734b8, false}},
            {tanh_pol_table, {0x3f70a4de, false}},
            {tanh_pol_table, {0x3f5f1fd8, false}},
            {tanh_pol_table, {0x3f495493, false}},
            {tanh_pol_table, {0x3f18b9ec, false}},
            {tanh_pol_table, {0x3ed706cb, false}},
            {tanh_pol_table, {0x3e390b06, false}},
            {tanh_pol_table, {0x3d90b11f, false}},
            {tanh_pol_table, {0x3c21a053, false}},
            {tanh_pol_table, {0x3aaf7fdb, false}},
            {tanh_pol_table, {0x37ccc1a3, false}},
            {tanh_pol_table, {0x355c6733, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 2
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xbe4e0ff1, false}},
            {tanh_pol_table, {0x3d25b1b1, false}},
            {tanh_pol_table, {0x3d6b6dab, false}},
            {tanh_pol_table, {0x3c9fb1d5, false}},
            {tanh_pol_table, {0xbabff06f, false}},
            {tanh_pol_table, {0x3c07b3f6, false}},
            {tanh_pol_table, {0xbb3fc1bc, false}},
            {tanh_pol_table, {0x3a9f5921, false}},
            {tanh_pol_table, {0xbbbf06f2, false}},
            {tanh_pol_table, {0xbbb0f402, false}},
            {tanh_pol_table, {0xbc47db9e, false}},
            {tanh_pol_table, {0xbc73d5e7, false}},
            {tanh_pol_table, {0xbca25bda, false}},
            {tanh_pol_table, {0xbcfca780, false}},
            {tanh_pol_table, {0xbd40e07c, false}},
            {tanh_pol_table, {0xbd7dab03, false}},
            {tanh_pol_table, {0xbdbe4a0f, false}},
            {tanh_pol_table, {0xbdfb14a5, false}},
            {tanh_pol_table, {0xbe36cc8d, false}},
            {tanh_pol_table, {0xbe6bd102, false}},
            {tanh_pol_table, {0xbe9fe7c5, false}},
            {tanh_pol_table, {0xbeba0f10, false}},
            {tanh_pol_table, {0xbec206a8, false}},
            {tanh_pol_table, {0xbea3c388, false}},
            {tanh_pol_table, {0xbe277d62, false}},
            {tanh_pol_table, {0xbd8b7960, false}},
            {tanh_pol_table, {0xbc209f49, false}},
            {tanh_pol_table, {0xbaad44ca, false}},
            {tanh_pol_table, {0xb7c6eeac, false}},
            {tanh_pol_table, {0xb663aa41, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 3
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x45b3ae96, false}},
            {tanh_pol_table, {0xc414eb20, false}},
            {tanh_pol_table, {0xc450e02e, false}},
            {tanh_pol_table, {0xc3152b4e, false}},
            {tanh_pol_table, {0xbead2f56, false}},
            {tanh_pol_table, {0xc2162e02, false}},
            {tanh_pol_table, {0xbeb4bd5a, false}},
            {tanh_pol_table, {0xc11a59a4, false}},
            {tanh_pol_table, {0xbed2f507, false}},
            {tanh_pol_table, {0xc020d32c, false}},
            {tanh_pol_table, {0x3dd0f506, false}},
            {tanh_pol_table, {0xbf2a75e2, false}},
            {tanh_pol_table, {0xbff950e3, false}},
            {tanh_pol_table, {0xbed47334, false}},
            {tanh_pol_table, {0xbe809b8c, false}},
            {tanh_pol_table, {0xbeb64532, false}},
            {tanh_pol_table, {0xbe961a5b, false}},
            {tanh_pol_table, {0xbe9b63ac, false}},
            {tanh_pol_table, {0xbea0d4b2, false}},
            {tanh_pol_table, {0xbe828a77, false}},
            {tanh_pol_table, {0xbe378612, false}},
            {tanh_pol_table, {0xbdc20908, false}},
            {tanh_pol_table, {0x3d2d3957, false}},
            {tanh_pol_table, {0x3dd46e89, false}},
            {tanh_pol_table, {0x3db3f629, false}},
            {tanh_pol_table, {0x3d2c5e7b, false}},
            {tanh_pol_table, {0x3bd20403, false}},
            {tanh_pol_table, {0x3a59dfae, false}},
            {tanh_pol_table, {0x3770af45, false}},
            {tanh_pol_table, {0x372cc014, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 4
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xcc981a1b, false}},
            {tanh_pol_table, {0x4a7edd3d, false}},
            {tanh_pol_table, {0x4ab1007c, false}},
            {tanh_pol_table, {0x48fedd9c, false}},
            {tanh_pol_table, {0x41a557b5, false}},
            {tanh_pol_table, {0x477ee32a, false}},
            {tanh_pol_table, {0x422557f5, false}},
            {tanh_pol_table, {0x45ff3ce4, false}},
            {tanh_pol_table, {0x42a55641, false}},
            {tanh_pol_table, {0x446e0867, false}},
            {tanh_pol_table, {0xc33dc19a, false}},
            {tanh_pol_table, {0x42915214, false}},
            {tanh_pol_table, {0x43af4fad, false}},
            {tanh_pol_table, {0x4110fe88, false}},
            {tanh_pol_table, {0xc1099b75, false}},
            {tanh_pol_table, {0x3fc8a8dc, false}},
            {tanh_pol_table, {0xbfbeaef5, false}},
            {tanh_pol_table, {0xbe365aad, false}},
            {tanh_pol_table, {0x3f4d9652, false}},
            {tanh_pol_table, {0x3ddfa08f, false}},
            {tanh_pol_table, {0x3e34e9b8, false}},
            {tanh_pol_table, {0x3e2d07a6, false}},
            {tanh_pol_table, {0x3dc63567, false}},
            {tanh_pol_table, {0x3cdaeb78, false}},
            {tanh_pol_table, {0xbcd17537, false}},
            {tanh_pol_table, {0xbc92829c, false}},
            {tanh_pol_table, {0xbb43ab99, false}},
            {tanh_pol_table, {0xb9b471dd, false}},
            {tanh_pol_table, {0xb6baad5a, false}},
            {tanh_pol_table, {0xb78bafc7, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 5
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0x52f688d5, false}},
            {tanh_pol_table, {0xd0505c72, false}},
            {tanh_pol_table, {0xd08f98e3, false}},
            {tanh_pol_table, {0xce505cc9, false}},
            {tanh_pol_table, {0xc7162b8a, false}},
            {tanh_pol_table, {0xcc5061d6, false}},
            {tanh_pol_table, {0xc7162bdf, false}},
            {tanh_pol_table, {0xca50b37f, false}},
            {tanh_pol_table, {0xc7162a3a, false}},
            {tanh_pol_table, {0xc8422086, false}},
            {tanh_pol_table, {0x471a714e, false}},
            {tanh_pol_table, {0xc5ece1f1, false}},
            {tanh_pol_table, {0xc70e3d90, false}},
            {tanh_pol_table, {0xc3eba94a, false}},
            {tanh_pol_table, {0x43e0c424, false}},
            {tanh_pol_table, {0xc21f4552, false}},
            {tanh_pol_table, {0x42217cc8, false}},
            {tanh_pol_table, {0x405e7dc4, false}},
            {tanh_pol_table, {0xc10dd401, false}},
            {tanh_pol_table, {0x3e96b602, false}},
            {tanh_pol_table, {0xbd1a6d2f, false}},
            {tanh_pol_table, {0xbd393883, false}},
            {tanh_pol_table, {0xbd674682, false}},
            {tanh_pol_table, {0xbd310016, false}},
            {tanh_pol_table, {0xb961e269, false}},
            {tanh_pol_table, {0x3ba32495, false}},
            {tanh_pol_table, {0x3a7680d5, false}},
            {tanh_pol_table, {0x38b3173c, false}},
            {tanh_pol_table, {0x35a9deea, false}},
            {tanh_pol_table, {0x375c3f2a, false}},
            {tanh_pol_table, {0x00000000, false}},
            // coefficients of degree 6
            {tanh_pol_table, {0x00000000, false}},
            {tanh_pol_table, {0xd8995ed1, false}},
            {tanh_pol_table, {0x558285ea, false}},
            {tanh_pol_table, {0x55b2cd69, false}},
            {tanh_pol_table, {0x53028625, false}},
            {tanh_pol_table, {0x4bc9991f, false}},
            {tanh_pol_table, {0x5082898a, false}},
            {tanh_pol_table, {0x4b4999b3, false}},
            {tanh_pol_table, {0x4e02c07c, false}},
            {tanh_pol_table, {0x4ac99764, false}},
            {tanh_pol_table, {0x4b72c822, false}},
            {tanh_pol_table, {0xca40c0e1, false}},
            {tanh_pol_table, {0x489413e4, false}},
            {tanh_pol_table, {0x49b12224, false}},
            {tanh_pol_table, {0x46134c4e, false}},
            {tanh_pol_table, {0xc60c2d57, false}},
            {tanh_pol_table, {0x43c83910, false}},
            {tanh_pol_table, {0xc3c872d1, false}},
            {tanh_pol_table, {0xc186bc9e, false}},
            {tanh_pol_table, {0x42325bc3, false}},
            {tanh_pol_table, {0xbf2ffa4a, false}},
            {tanh_pol_table, {0x3d9a203c, false}},
            {tanh_pol_table, {0xbc545a43, false}},
            {tanh_pol_table, {0xbae08fee, false}},
            {tanh_pol_table, {0x3c80225d, false}},
            {tanh_pol_table, {0x3b1fd1df, false}},
            {tanh_pol_table, {0xba36b9d1, false}},
            {tanh_pol_table, {0xb91de544, false}},
            {tanh_pol_table, {0xb71f100f, false}},
            {tanh_pol_table, {0xb408e2ed, false}},
            {tanh_pol_table, {0xb685fec8, false}},
            {tanh_pol_table, {0x00000000, false}},
    };

    // soft_relu(x) constants
    static const table_t soft_relu_consts {
            {soft_relu_one_twenty_six, {0x42fc0000, true}},
            {soft_relu_mantissa_sign_mask, {0x807fffff, true}},
    };

    // soft_relu ln(1 + x) polynomial approximation
    static const table_t soft_relu_polynomial {
            {soft_relu_pol, {0xb2b4637d, true}}, // p0 = 0.0000000244f
            {soft_relu_pol, {0x3f7fff8e, true}}, // p1 = 0.9999976971f
            {soft_relu_pol, {0xbf001759, true}}, // p2 = -0.5002478215f
            {soft_relu_pol, {0x3ea70608, true}}, // p3 = 0.3272714505f
            {soft_relu_pol, {0xbea3d7bf, true}}, // p4 = -0.3153830071f
            {soft_relu_pol, {0xbe361d04, true}}, // p5 = -0.1701777461f
            {soft_relu_pol, {0xbfa8f1e6, true}}, // p6 = -1.3254635147f
            {soft_relu_pol, {0xbfe1e812, true}}, // p7 = -1.7971917960f
            {soft_relu_pol, {0xbfc4d30e, true}}, // p8 = -1.5652673123f
    };

    // gelu_tanh(x) constants (formula defined)
    static const table_t gelu_tanh_consts {
            {gelu_tanh_fitting_const, {0x3d372713, true}},
            {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
            {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
    };

    // gelu_erf(x) constants (formula defined)
    static const table_t gelu_erf_consts {
            {gelu_erf_approx_const, {0x3ea7ba05, true}},
            {gelu_erf_one_over_sqrt_two, {0x3f3504f3, true}},
            {gelu_erf_one_over_sqrt_pi, {0x3f106eba, true}},
    };

    // gelu_erf(x) polynomial approximation
    static const table_t gelu_erf_polynomial {
            {gelu_erf_pol, {0x3e827906, true}}, // p1 = 0.254829592f
            {gelu_erf_pol, {0xbe91a98e, true}}, // p2 = -0.284496736f
            {gelu_erf_pol, {0x3fb5f0e3, true}}, // p3 = 1.421413741f
            {gelu_erf_pol, {0xbfba00e3, true}}, // p4 = -1.453152027f
            {gelu_erf_pol, {0x3f87dc22, true}}, // p5 = 1.061405429f
    };

    // log(x) constants
    static const table_t log_consts {
            {log_minus_inf, {0xff800000, true}},
            {log_qnan, {0x7fc00000, true}},
            {log_mantissa_mask, {0x007fffff, true}},
            {log_full_k_reg_mask, {0x0000ffff, true}},
            {log_five_bit_offset, {0x0000001f, true}},
    };

    // log(x) polynomial approximation
    static const table_t log_polynomial {
            {log_pol, {0xbf000000, true}}, // p1 = -0.5f
            {log_pol, {0x3eaaaaab, true}}, // p2 =  0.333333343f
            {log_pol, {0xbe8004ab, true}}, // p3 = -0.250035613f
            {log_pol, {0x3e4cc8a3, true}}, // p4 =  0.199984118f
    };

    // log(x) pre-defined values. First goes index}, then val[index].
    static const table_t log_predefined_values {
            {log_predefined_vals, {0x3f800000, true}}, //  0: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, //  1: -88.029693603515625
            {log_predefined_vals, {0x3f780000, true}}, //  2: 0.96875
            {log_predefined_vals,
                    {0xc2affef2, true}}, //  3: -87.9979400634765625
            {log_predefined_vals, {0x3f700000, true}}, //  4: 0.9375
            {log_predefined_vals,
                    {0xc2afee29, true}}, //  5: -87.9651565551757812
            {log_predefined_vals, {0x3f680000, true}}, //  6: 0.90625
            {log_predefined_vals,
                    {0xc2afdccd, true}}, //  7: -87.9312515258789062
            {log_predefined_vals, {0x3f600000, true}}, //  8: 0.875
            {log_predefined_vals,
                    {0xc2afcad6, true}}, //  9: -87.8961639404296875
            {log_predefined_vals, {0x3f580000, true}}, // 10: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 11: -87.859794616699218
            {log_predefined_vals, {0x3f580000, true}}, // 12: 0.84375
            {log_predefined_vals,
                    {0xc2afb837, true}}, // 13: -87.859794616699218
            {log_predefined_vals, {0x3f500000, true}}, // 14: 0.8125
            {log_predefined_vals,
                    {0xc2afa4e4, true}}, // 15: -87.822052001953125
            {log_predefined_vals, {0x3f480000, true}}, // 16: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 17: -87.782829284667968
            {log_predefined_vals, {0x3f480000, true}}, // 18: 0.78125
            {log_predefined_vals,
                    {0xc2af90cf, true}}, // 19: -87.782829284667968
            {log_predefined_vals, {0x3f400000, true}}, // 20: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 21: -87.742012023925781
            {log_predefined_vals, {0x3f400000, true}}, // 22: 0.75
            {log_predefined_vals,
                    {0xc2af7be9, true}}, // 23: -87.742012023925781
            {log_predefined_vals, {0x3f380000, true}}, // 24: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 25: -87.699447631835937
            {log_predefined_vals, {0x3f380000, true}}, // 26: 0.71875
            {log_predefined_vals,
                    {0xc2af661e, true}}, // 27: -87.699447631835937
            {log_predefined_vals, {0x3f300000, true}}, // 28: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 29: -87.654998779296875
            {log_predefined_vals, {0x3f300000, true}}, // 30: 0.6875
            {log_predefined_vals,
                    {0xc2af4f5c, true}}, // 31: -87.654998779296875
            {log_predefined_vals, {0x3fa80000, true}}, // 32: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 33: -88.301628112792968
            {log_predefined_vals, {0x3fa80000, true}}, // 34: 1.3125
            {log_predefined_vals,
                    {0xc2b09a6f, true}}, // 35: -88.301628112792968
            {log_predefined_vals, {0x3fa00000, true}}, // 36: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 37: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 38: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 39: -88.252838134765625
            {log_predefined_vals, {0x3fa00000, true}}, // 40: 1.25
            {log_predefined_vals,
                    {0xc2b08174, true}}, // 41: -88.252838134765625
            {log_predefined_vals, {0x3f980000, true}}, // 42: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 43: -88.201545715332031
            {log_predefined_vals, {0x3f980000, true}}, // 44: 1.1875
            {log_predefined_vals,
                    {0xc2b06731, true}}, // 45: -88.201545715332031
            {log_predefined_vals, {0x3f900000, true}}, // 46: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 47: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 48: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 49: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 50: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 51: -88.147476196289062
            {log_predefined_vals, {0x3f900000, true}}, // 52: 1.125
            {log_predefined_vals,
                    {0xc2b04b82, true}}, // 53: -88.147476196289062
            {log_predefined_vals, {0x3f880000, true}}, // 54: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 55: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 56: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 57: -88.090316772460937
            {log_predefined_vals, {0x3f880000, true}}, // 58: 1.0625
            {log_predefined_vals,
                    {0xc2b02e3e, true}}, // 59: -88.090316772460937
            {log_predefined_vals, {0x3f800000, true}}, // 60: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 61: -88.029693603515625
            {log_predefined_vals, {0x3f800000, true}}, // 62: 1
            {log_predefined_vals,
                    {0xc2b00f34, true}}, // 63: -88.029693603515625
    };

    // This object takes care about which constants and polynomials to include.
    struct need_t {
        need_t(alg_kind_t alg) {
            using namespace alg_kind;
            switch (alg) {
                case eltwise_elu_use_dst_for_bwd:
                case eltwise_elu:
                case eltwise_exp_use_dst_for_bwd:
                case eltwise_exp:
                case eltwise_logistic_use_dst_for_bwd:
                case eltwise_logistic:
                case eltwise_swish: exp_ = true; break;
                case eltwise_gelu_erf: gelu_erf_ = true; break;
                case eltwise_gelu_tanh: gelu_tanh_ = true; break;
                case eltwise_log: log_ = true; break;
                case eltwise_soft_relu: soft_relu_ = true; break;
                case eltwise_tanh_use_dst_for_bwd:
                case eltwise_tanh: tanh_ = true; break;
                default: break;
            }
        }

        bool exp_ = false;
        bool tanh_ = false;
        bool soft_relu_ = false;
        bool gelu_tanh_ = false;
        bool gelu_erf_ = false;
        bool log_ = false;

        bool exp() const { return exp_ || soft_relu_ || gelu_erf_; }
        bool tanh() const { return tanh_ || gelu_tanh_; }
        bool soft_relu() const { return soft_relu_; }
        bool gelu_tanh() const { return gelu_tanh_; }
        bool gelu_erf() const { return gelu_erf_; }
        bool log() const { return log_; }
    };

    need_t need(alg_);

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val,
                                     const bool broadcast) {
        mapped_table_entry_t te {0, val, broadcast};
        entry_map_.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t &t) {
        for (auto it = t.begin(); it != t.end(); it++) {
            auto key = (*it).first;
            auto te = (*it).second; // copy values from table
            push_arg_entry_of(key, te.val, te.bcast);
        }
    };

    push_arg_entry_of(scale, float2int(scale_), true);
    push_arg_entry_of(alpha, float2int(alpha_), true);
    push_arg_entry_of(beta, float2int(beta_), true);
    push_entries_of(common_values);
    if (need.exp()) push_entries_of(exp_consts);
    if (need.exp()) push_entries_of(exp_polynomial);
    if (need.tanh()) push_entries_of(tanh_consts);
    if (need.tanh()) push_entries_of(tanh_polynomial_table);
    if (need.soft_relu()) push_entries_of(soft_relu_consts);
    if (need.soft_relu()) push_entries_of(soft_relu_polynomial);
    if (need.gelu_tanh()) push_entries_of(gelu_tanh_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_consts);
    if (need.gelu_erf()) push_entries_of(gelu_erf_polynomial);
    if (need.log()) push_entries_of(log_consts);
    if (need.log()) push_entries_of(log_polynomial);
    if (need.log()) push_entries_of(log_predefined_values);

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? vlen : sizeof(table_entry_val_t);
    }
}

#ifdef DNNL_X64_IMPLEMENTATION
template struct jit_uni_eltwise_injector_f32<avx512_core>;
template struct jit_uni_eltwise_injector_f32<avx512_common>;
template struct jit_uni_eltwise_injector_f32<avx2>;
template struct jit_uni_eltwise_injector_f32<sse41>;
#else /* DNNL_X64_IMPLEMENTATION */
template struct jit_uni_eltwise_injector_f32<sve>;
#endif /* DNNL_X64_IMPLEMENTATION */

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
