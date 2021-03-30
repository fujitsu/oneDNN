/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_UNI_LRN_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_LRN_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/aarch64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct bf16_emulation_t;
struct jit_args_fwd_t {
    const void *src;
    void *dst, *scratch, *bwd_intermediate_res;
};

struct jit_args_bwd_t {
    const void *src, *diff_dst, *scratch, *bwd_intermediate_res;
    void *diff_src;
};

struct nchw8c_across {
    /*  version:
    *  -1: channels 0..7,
    *   1: channels C-8 .. C-1,
    *   0: other channels
    *   3: channels only for this kernel(without prev and next)
    */
    int H, W, version;
    nchw8c_across(int h, int w, int v) : H(h), W(w), version(v) {}
};

struct within_config {
    const int H, W, C, size;
    const format_tag_t dat_tag;
    within_config(int h, int w, int c, int s, format_tag_t dat_tag)
        : H(h), W(w), C(c), size(s), dat_tag(dat_tag) {}
};

struct nchw_across {
    int C, HW, tail;
    nchw_across(int c, int hw, int t) : C(c), HW(hw), tail(t) {}
};

struct nhwc_across {
    int C;
    nhwc_across(int c) : C(c) {}
};

template <class Derived>
class jit_uni_lrn_kernel_t; // primary template

template <template <cpu_isa_t isa, data_type_t d_type> class Derived,
        cpu_isa_t isa, data_type_t d_type>
class jit_uni_lrn_kernel_t<Derived<isa, d_type>> : public jit_generator {
public:
    jit_uni_lrn_kernel_t(
            void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE);
    jit_uni_lrn_kernel_t(const within_config &J, void *code_ptr = nullptr,
            size_t code_size = MAX_CODE_SIZE);

    ~jit_uni_lrn_kernel_t();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_kernel_t);
    static constexpr int VECTOR_LENGTH = (isa == avx512_common ? 16 : 8);

protected:
    using Vmm = typename utils::conditional<isa == avx2, Xbyak::Ymm,
            Xbyak::Zmm>::type;

    void load_constant(float constant, const Vmm &v_constant,
            const Xbyak::Xmm &x_constant);
    void load_data(const Vmm &reg, const Xbyak::Address &p);
    void store_data(const Xbyak::Address &p, const Vmm &reg);
    void within_loop(
            const within_config &config, int max_reg_blocks, prop_kind_t pk);
    void within_body_reg_blocked(int loop_count, int max_reg_block, int hoff,
            int Hoff, int woff, int Woff, int stride, prop_kind_t pk);

    const bool emulate_bfloat_;
    const Xbyak::Zmm bf16_emu_reserv_1_ = Xbyak::Zmm(28);
    const Xbyak::Zmm bf16_emu_reserv_2_ = Xbyak::Zmm(29);
    const Xbyak::Reg64 bf16_emu_scratch_ = this->rax;
    const Xbyak::Zmm bf16_emu_reserv_3_ = Xbyak::Zmm(30);
    const Xbyak::Zmm bf16_emu_reserv_4_ = Xbyak::Zmm(31);
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    const Xbyak::Reg64 h_ = this->r9;
    const Xbyak::Reg64 w_ = this->r10;
    const Xbyak::Reg64 imm_addr64_ = this->rbx;
    int single_pixel_offset_
            = VECTOR_LENGTH * sizeof(typename prec_traits<d_type>::type);
    int tempIdx_ = 0;
    int reg_block_idx_ = 0;
};

template <cpu_isa_t isa, data_type_t d_type>
class jit_uni_lrn_fwd_kernel_t
    : public jit_uni_lrn_kernel_t<jit_uni_lrn_fwd_kernel_t<isa, d_type>> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_fwd_kernel_t)

    jit_uni_lrn_fwd_kernel_t(const within_config &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 4 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel_t(const nchw8c_across &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel_t(const nhwc_across &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel_t(const nchw_across &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    ~jit_uni_lrn_fwd_kernel_t();
    void operator()(jit_args_fwd_t *arg) { ker(arg); }
    void (*ker)(jit_args_fwd_t *);

private:
    using Base = jit_uni_lrn_kernel_t<jit_uni_lrn_fwd_kernel_t<isa, d_type>>;

public:
    using Base::VECTOR_LENGTH;

private:
    friend Base;
    using typename Base::Vmm;

    void within_body(int hoff, int Hoff, int woff, int Woff, int stride,
            prop_kind_t pk, int reg_block = 1, int single_pixel_offset = 0);
    void nchw_body(int tail, int HW, prop_kind_t pk, Xbyak::Ymm ymask,
            Xbyak::Ymm ya, Xbyak::Ymm yb, Xbyak::Ymm yc, Xbyak::Ymm yd,
            Xbyak::Ymm ye, Xbyak::Ymm ysum);
    void nchw_body_sse41(int tail, int HW, prop_kind_t pk, Xbyak::Xmm xe_lo,
            Xbyak::Xmm xe_hi, Xbyak::Xmm xsum_lo, Xbyak::Xmm xsum_hi);
    void nchw_tail_sse41(int tail, Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo,
            Xbyak::Xmm xtail_hi);
    void move_data_pointers(int pixel_count, prop_kind_t pk);

    const Xbyak::Reg64 src_ = this->rax;
    const Xbyak::Reg64 dst_ = this->r8;
    const Xbyak::Reg64 scratch_ = this->r14;
    const Xbyak::Reg64 bwd_intermediate_res_ = this->rdx;
    const Xbyak::Reg64 store_addr_ = this->rbp;

    const Xbyak::Xmm xalpha_ = this->xmm0;
    const Xbyak::Xmm xk_ = this->xmm1;
    const Xbyak::Ymm yk_ = this->ymm1;
    const Vmm valpha_ = Vmm(0);
    const Vmm vk_ = Vmm(1);

    float alpha_;
    float k_;
    static constexpr int stack_space_needed_ = 11 * 4 * sizeof(float) + 16;
};

template <cpu_isa_t isa, data_type_t d_type>
class jit_uni_lrn_bwd_kernel_t
    : public jit_uni_lrn_kernel_t<jit_uni_lrn_bwd_kernel_t<isa, d_type>> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_bwd_kernel_t)

    jit_uni_lrn_bwd_kernel_t(const nchw8c_across &J, float A, float B,
            int use_h_parallel, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_bwd_kernel_t(const within_config &J, float A, float B,
            void *code_ptr = nullptr,
            size_t code_size = 4 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    void operator()(jit_args_bwd_t *arg) { ker(arg); }
    void (*ker)(jit_args_bwd_t *);

private:
    using Base = jit_uni_lrn_kernel_t<jit_uni_lrn_bwd_kernel_t<isa, d_type>>;

public:
    using Base::VECTOR_LENGTH;

private:
    friend Base;
    using typename Base::Vmm;

    void within_body(int hoff, int Hoff, int woff, int Woff, int stride,
            prop_kind_t pk, int reg_block = 1, int single_pixel_offset = 0);
    void move_data_pointers(int pixel_count, prop_kind_t pk);

    float nalphabeta_;
    int use_h_parallelizm_;
    const Xbyak::Reg64 src_ = this->rax;
    const Xbyak::Reg64 diffsrc_ = this->r13;
    const Xbyak::Reg64 diffdst_ = this->r14;
    const Xbyak::Reg64 scratch_ = this->r15;
    const Xbyak::Reg64 bwd_intermediate_res_ = this->rdx;
    const Xbyak::Xmm xnalphabeta_ = this->xmm0;
    const Vmm vnalphabeta_ = Vmm(0);
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
