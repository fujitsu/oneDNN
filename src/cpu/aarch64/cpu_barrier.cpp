/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <assert.h>

#include "cpu/aarch64/cpu_barrier.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace simple_barrier {

void generate(jit_generator &code, Xbyak_aarch64::XReg reg_ctx,
        Xbyak_aarch64::XReg reg_nthr) {
#define BAR_CTR_OFF offsetof(ctx_t, ctr)
#define BAR_SENSE_OFF offsetof(ctx_t, sense)
    using namespace Xbyak_aarch64;

    XReg x_tmp_0(23);
    XReg x_tmp_1(24);
    XReg x_tmp_2(25);

    XReg reg_tmp(27);
    XReg sp_tmp1(28);
    XReg sp_tmp2(29);
    XReg sp(31);

    Label barrier_exit_label, barrier_exit_restore_label, spin_label;

    code.cmp(reg_nthr, 1);
    code.b(LS, barrier_exit_label);

    code.mov(sp_tmp1, reg_tmp);

    /* take and save current sense */
    code.add_imm(x_tmp_0, reg_ctx, BAR_SENSE_OFF, x_tmp_0);
    code.ldr(reg_tmp, ptr(x_tmp_0));
    code.mov(sp_tmp2, reg_tmp);
    code.mov(reg_tmp, 1);

#if 0
    if (mayiuse(avx512_mic)) {
        code.prefetchwt1(code.ptr[reg_ctx + BAR_CTR_OFF]);
        code.prefetchwt1(code.ptr[reg_ctx + BAR_CTR_OFF]);
    }
#endif // #if 0

    code.add_imm(x_tmp_1, reg_ctx, BAR_CTR_OFF, x_tmp_1);
    code.ldaddal(reg_tmp, reg_tmp, ptr(x_tmp_1));
    code.add_imm(reg_tmp, reg_tmp, 1, x_tmp_0);
    code.cmp(reg_tmp, reg_nthr);
    code.mov(reg_tmp, sp_tmp2);
    code.b(NE, spin_label);

    /* the last thread {{{ */
    code.mov_imm(x_tmp_2, 0);
    code.str(x_tmp_2, ptr(x_tmp_1));

    // notify waiting threads
    code.mvn(reg_tmp, reg_tmp);
    code.str(reg_tmp, ptr(x_tmp_0));
    code.b(barrier_exit_restore_label);
    /* }}} the last thread */

    code.CodeGenerator::L(spin_label);
    code.ldr(x_tmp_1, ptr(x_tmp_0));
    code.cmp(reg_tmp, x_tmp_1);
    code.b(EQ, spin_label);

//#ifdef DNNL_INDIRECT_JIT_AARCH64
    code.CodeGenerator::dmb(ISH);
//#endif //#ifdef DNNL_INDIRECT_JIT_AARCH64

    code.CodeGenerator::L(barrier_exit_restore_label);
    code.mov(reg_tmp, sp_tmp1);

    code.CodeGenerator::L(barrier_exit_label);
#undef BAR_CTR_OFF
#undef BAR_SENSE_OFF
}

/** jit barrier generator */
struct jit_t : public jit_generator {

    void generate() override {
        simple_barrier::generate(*this, abi_param1, abi_param2);
        ret();
    }

    // TODO: Need to check status
    jit_t() { create_kernel(); }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_t)
};

} // namespace simple_barrier

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
