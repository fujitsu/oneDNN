/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_BINARY_INJECTOR_HPP
#define CPU_AARCH64_JIT_UNI_BINARY_INJECTOR_HPP

#include <array>
#include <cassert>
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <unordered_set>

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/binary_injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace binary_injector {
using dnnl::impl::cpu::binary_injector_utils::prepare_binary_args;

struct Address_t {
    Xbyak_aarch64::XReg base_reg = Xbyak_aarch64::XReg(0);
    bool isBroadcast_ = 0;
    bool isBroadcast() const { return isBroadcast_; }
};

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops);

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

bool binary_args_tail_supported(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d, int vlen);

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d);

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> predicate);

/*
 * Represents params related to all binary post-ops right-hand side arguments
 * (arg1) that don't change during jit_uni_binary_injector_t object lifetime
 * and between compute_vector_range calls.
 *
 * @param rhs_dt_helper_treg_idx - index of treg helper used when loading data for
 * calculations. Treated as hint from user. If inside compute_vector_range hint
 * turns out to be invalid, it will be overwriten by register preserving logic inside
 * binary injector.
 * @param rhs_addr_reg - gpr register, used as the currently processed address of
 * rhs tensor slice. Data of rhs(arg1) for the binary operation is loaded from address
 * stored inside rhs_addr_reg.
 * @param rhs_helper_reg - gpr register used as helper for calculations during data
 * loading phase.
 * @param preserve_gpr_helpers - determines whether gpr registers specified above
 * should be preserved (pushed to stack and poped back afterwords) between
 * compute_vector_range calls.
 * @param preserve_treg_helper - determines whether treg helper register specified
 * above should be preserved between compute_vector_range calls.
 * @param abi_param_offset - offset to rhs tensor from first binary post-op operation
 * specified by user from runtime structure passed to kernel as abi param 1.
 * @param dst_d - descriptor of destination tensor (result after applying all post-ops
 * operations).
 * @param tail_opmask - register with loaded by user mask, used in avx512 for load with
 * tail handling.
 * @oaram tail_size - size of processed tail in elements.
 * @param use_exact_tail_scalar_bcast - in case of scalar broadcast user can disable
 * loading data with tail, usually bcast through entire vector is faster (usually 1 instruction)
 * vs. broadcasting limited by tail size (potentially several instructions). In case
 * when user during storing ignores values from treg above tail size, setting this option to
 * false can result in better performance.
 */
struct rhs_arg_static_params_t {
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_treg_idx,
            const Xbyak_aarch64::XReg &rhs_addr_reg,
            const Xbyak_aarch64::XReg &rhs_helper_reg,
            bool preserve_gpr_helpers, bool preserve_treg_helper,
            std::size_t abi_param_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size = 0u,
            bool use_exact_tail_scalar_bcast = false);
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_treg_idx,
            const Xbyak_aarch64::XReg &rhs_addr_reg,
            const Xbyak_aarch64::XReg &rhs_helper_reg,
            bool preserve_gpr_helpers, bool preserve_treg_helper,
            std::size_t abi_param_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
            bool use_exact_tail_scalar_bcast);

    bool is_opmask_set() const noexcept { return is_opmask_set_; }

    mutable std::size_t rhs_dt_helper_treg_idx;

    Xbyak_aarch64::XReg rhs_addr_reg;
    Xbyak_aarch64::XReg rhs_helper_reg;
    bool preserve_gpr_helpers;
    bool preserve_treg_helper;
    std::size_t abi_param_offset;
    memory_desc_wrapper dst_d;
    std::size_t tail_size;
    Xbyak_aarch64::PReg tail_opmask;
    bool use_exact_tail_scalar_bcast;

private:
    rhs_arg_static_params_t(std::size_t rhs_dt_helper_treg_idx,
            const Xbyak_aarch64::XReg &rhs_addr_reg,
            const Xbyak_aarch64::XReg &rhs_helper_reg,
            bool preserve_gpr_helpers, bool preserve_treg_helper,
            std::size_t abi_param_offset, const memory_desc_wrapper &dst_d,
            std::size_t tail_size, const Xbyak_aarch64::PReg &tail_opmask,
            bool use_exact_tail_scalar_bcast, bool is_opmask_set);

    bool is_opmask_set_;
};

/*
 * Represents params required by jit_uni_binary_injector_t that don't change
 * during it's entire lifetime.
 *
 * @param param1 - register storing abi param1. At the moment of calling
 * compute_vector_range method can be different than the default one defined
 * inside jit_generator.
 * @param bcast_set_t supported_strategy_set - set allowing disabling particular
 * bcast strategies
 * @param rhs_arg_static_params - params related to all binary post-ops right-hand side
 * arguments that don't change during entire lifetime of jit_uni_binary_injector_t
 * object.
 */
struct static_params_t {
    static_params_t(const Xbyak_aarch64::XReg &param1,
            const bcast_set_t &supported_strategy_set,
            const rhs_arg_static_params_t &rhs_arg_static_params);
    static_params_t(const Xbyak_aarch64::XReg &param1,
            const rhs_arg_static_params_t &rhs_arg_static_params);

    Xbyak_aarch64::XReg param1;
    const bcast_set_t supported_strategy_set;
    rhs_arg_static_params_t rhs_arg_static_params;
};

/*
 * Represents params passed to compute_vector_range method of
 * jit_uni_binary_injector_t that can be different for each call.
 * Contains configurable std::maps where key is treg index and value is
 * offset in elements. The offset value identifies tensor slice in particular
 * treg. This is utilized by broadcasting mechanism. Offset, depending on the
 * implementation particular kernels, can be passed as value (usually during
 * unrolling), inside operand, under memory address.
 *
 * @param treg_idx_to_out_elem_off_addr - treg mapped to offset in elements stored under
 * memory address intended to use in no_broadcast strategy.
 * @param treg_idx_to_out_elem_off_addr - treg mapped to offset in elements passed as raw
 * value intended to use in no_broadcast strategy
 * @param treg_idx_to_out_elem_off_addr - treg mapped to offset in elements inside operand
 * intended to use in no_broadcast strategy
 * @param treg_idx_to_oc_elem_off_addr - treg mapped to output channel offset in elements
 * stored under memory address intended to use in per_oc broadcast strategies.
 * @param treg_idx_to_oc_elem_off_val - treg mapped to  output channel offset in elements
 * passed as raw value intended to use in per_oc broadcast strategies.
 * @param treg_idx_to_oc_off_oprnd - treg mapped to output channel offset in elements inside
 * operand intended to use in per_oc broadcast strategies.
 * @param treg_tail_idx - treg idxes that contains data don't fill the whole vector (tail).
 */

struct rhs_arg_dynamic_params_t {
    std::map<int, Xbyak_aarch64::AdrNoOfs> treg_idx_to_out_elem_off_addr;
    std::map<int, int> treg_idx_to_out_elem_off_val;
    std::map<int, Xbyak_aarch64::XReg> treg_idx_to_out_off_oprnd;

    std::map<int, Xbyak_aarch64::AdrNoOfs> treg_idx_to_oc_elem_off_addr;
    std::map<int, int> treg_idx_to_oc_elem_off_val;
    std::map<int, Xbyak_aarch64::XReg> treg_idx_to_oc_off_oprnd;
    std::unordered_set<int> treg_tail_idx_;
};

/*
 * Checks if src1 data type is supported by binary injector.
 */
bool is_data_supported(cpu_isa_t isa, data_type_t data_type);

/*
 * Checks if broadcast of src1 is supported by binary injector.
 */
bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Checks if binary injection for given args is supported.
 */
bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set);

/*
 * Main mechanism responsible for injecting binary postops supporting various
 * isa: sse41, avx, avx2, avx512 with core, bf16 extensions as well as data
 * types: f32, bf16, s32, u8, s8.
 */
template <cpu_isa_t isa>
class jit_uni_binary_injector_t {
public:
    jit_uni_binary_injector_t(
            jit_generator *host, const static_params_t &static_params);

    using TReg = typename cpu_isa_traits<isa>::TReg;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * ordered set of vector registers' indexes. Function loads appropriate
     * slice of rhs tensor for computations based on internally determined
     * broadcast strategy and information about stored data in particular treg
     * described inside rhs_arg_params.
     */
    void compute_vector_range(const injector_utils::treg_index_set_t &treg_idxs,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * range <start_idx, end_idx) of vector registers' indexes. Function loads
     * appropriate slice of rhs tensor for computations based on internally
     * determined broadcast strategy and information about stored data in particular
     * treg described inside rhs_arg_params.
     */
    void compute_vector_range(size_t start_idx, size_t end_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

    /*
     * Generates code of binary post_op injected to host primitive. Applied to
     * a single vector register index. Function loads appropriate slice of rhs tensor
     * for computations based on internally determined broadcast strategy and information
     * about stored data in particular treg described inside rhs_arg_params.
     */
    void compute_vector(size_t idx, std::size_t rhs_arg_idx,
            const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params) const;

private:
    /*
     * Determines if hint passed by user is valid (is inside range
     * <start_idx, end_idx>). If not it returns new treg idx value that will be
     * used as temporary treg in future computations.
     */
    int adjust_temp_treg_hint(
            int user_hint, int start_idx, int end_idx, int max_treg_idx) const;
    /*
     * Taking into account rhs_broadcasting_strategy and information from user
     * about tensor slice (rhs_arg_params) stored in Treg(treg_idx) calculates
     * address of rhs tensor slice needed for binary operation and returns
     * ptr to it.
     */
    Xbyak_aarch64::AdrNoOfs prepare_rhs_arg_addr(std::size_t treg_idx,
            std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
            const rhs_arg_dynamic_params_t &rhs_arg_params,
            const broadcasting_strategy_t rhs_broadcasting_strategy) const;
    /*
     * Loads data and applies particular binary operation.
     */
    void inject_binary(const dnnl_post_ops::entry_t &post_op, TReg dst,
            const Xbyak_aarch64::AdrNoOfs &rhs_addr, bool with_tail) const;
    /*
     * Helper functions responsible for preparing rhs tensor slice address.
     */
    void append_offset_from_operand(
            const std::map<int, Xbyak_aarch64::XReg> &treg_idx_to_elem_addr_off,
            int treg_idx, const Xbyak_aarch64::XReg &addr_reg,
            const Xbyak_aarch64::XReg &tmp_reg,
            std::size_t elem_size_bytes) const;
    void append_offset_under_mem_addr(
            const std::map<int, Xbyak_aarch64::AdrNoOfs>
                    &treg_idx_to_elem_addr_off,
            int treg_idx, const Xbyak_aarch64::XReg &addr_reg,
            const Xbyak_aarch64::XReg &tmp_reg,
            std::size_t elem_size_bytes) const;
    void append_value_offset(const std::map<int, int> &treg_idx_to_elem_val_off,
            int treg_idx, const Xbyak_aarch64::XReg &addr_reg,
            std::size_t elem_size_bytes) const;

    template <typename T>
    typename std::enable_if<std::is_same<T, Xbyak_aarch64::ZReg>::value
            || std::is_same<T, Xbyak_aarch64::AdrNoOfs>::value>::type
    execute_cmp_binary(const TReg &dst, const TReg &lhs, const T &rhs,
            const unsigned int cmp_predicate) const;
    template <typename T>
    typename std::enable_if<!(std::is_same<T, Xbyak_aarch64::ZReg>::value
            || std::is_same<T, Xbyak_aarch64::AdrNoOfs>::value)>::type
    execute_cmp_binary(const TReg &dst, const TReg &lhs, const T &rhs,
            const unsigned int cmp_predicate) const;
    void execute_binary(alg_kind_t binary_alg, const TReg &dst, const TReg &lhs,
            const TReg &rhs) const;
    void execute_binary(alg_kind_t binary_alg, const TReg &dst, const TReg &lhs,
            const Xbyak_aarch64::AdrNoOfs &rhs) const;
    /*
     * Used in scalar broadcast strategy, broadcasting single value of given
     * data type over entire vector TReg register.
     */
    void execute_broadcast(const dnnl_data_type_t &data_type,
            const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr,
            bool with_tail = false) const;
    void load_rhs(const dnnl_data_type_t &data_type, const TReg &tmp_reg,
            const Xbyak_aarch64::AdrNoOfs &rhs_addr,
            bool with_tail = false) const;
    void execute_broadcast_tail(const dnnl_data_type_t &data_type,
            const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr) const;
    void load_rhs_tail(const dnnl_data_type_t &data_type, const TReg &tmp_reg,
            const Xbyak_aarch64::AdrNoOfs &rhs_addr) const;
    void execute_broadcast_no_tail(const dnnl_data_type_t &data_type,
            const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr) const;
    void execute_broadcast_s8u8_no_tail(const data_type_t &data_type,
            const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr) const;
    void load_rhs_no_tail(const dnnl_data_type_t &data_type,
            const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr) const;
    void cvt_to_f32(const TReg &tmp_reg) const;
    /*
     * Returns pair consisting of flag indication preservation is needed for treg
     * index in second member that should be used as temporary treg inside inject
     * binary.
     */
    std::pair<bool, int> should_preserve_treg(int curr_idx, int treg_hint,
            int max_treg_idx, bool dt_helper_treg_needed) const;
    /*
     * Used in isa != avx512 where m32bcst is not supported, replaces ptr_b
     * with ptr.
     */
    Xbyak_aarch64::AdrNoOfs remove_bcast_bit(
            const Xbyak_aarch64::AdrNoOfs &rhs_addr) const;

    jit_generator *host_;
    const rhs_arg_static_params_t rhs_arg_static_params_;
    const Xbyak_aarch64::XReg param1_;
    const bcast_set_t supported_strategy_set_;
    static constexpr bool is_sve_512_
            = std::is_same<TReg, Xbyak_aarch64::ZReg>::value;
    /*
     * Instructions from SSE/AVX used to compute binary result like vaddps where
     * second operand is memory, require mem operand to be 16/32 byte explicitly
     * aligned. (Intel Manual chapter 2.4).
     * Rule is relaxed from AVX2 (Intel Manual chapter 14.9).
     * When using benchdnn zmalloc_protect doesn't guarantee that tensor memory
     * address is 64 byte aligned, which can cause segmentation fault.
     */
    static constexpr bool binary_op_with_unaligned_mem_operand_allowed_ = true;
};

} // namespace binary_injector
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
