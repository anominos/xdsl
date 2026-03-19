import typing as t
from collections import defaultdict
from dataclasses import dataclass

from ordered_set import OrderedSet

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_stack import OutOfRegisters
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.dialects.riscv.registers import RISCVRegisterType
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import PassFailedException


def _is_virtual_reg(reg: builtin.Attribute) -> t.TypeGuard[riscv.IntRegisterType]:
    return (
        isinstance(reg, riscv.IntRegisterType)
        and isinstance(reg.index, builtin.IntAttr)
        and reg.index.data < 0
    )


def _get_srcs_and_dsts_from_swaps(
    swaps: list[tuple[SSAValue, SSAValue]],
) -> tuple[list[SSAValue], list[RISCVRegisterType]]:
    srcs: list[SSAValue] = []
    dsts: list[RISCVRegisterType] = []
    for x, y in swaps:
        assert isinstance(x.type, RISCVRegisterType)
        assert isinstance(y.type, RISCVRegisterType)
        srcs.append(x)
        dsts.append(y.type)
        srcs.append(y)
        dsts.append(x.type)
    return srcs, dsts


def _get_next_virtual(func_op: riscv_func.FuncOp):
    """Generator that yields the next unused virtual register."""
    min_reg_data = 0
    for reg in RegisterAllocatableOperation.iter_all_used_registers(func_op.body):
        if _is_virtual_reg(reg) and isinstance(reg.index, builtin.IntAttr):
            min_reg_data = min(min_reg_data, reg.index.data)

    while True:
        min_reg_data -= 1
        print("Next vreg", riscv.IntRegisterType.infinite_register(~min_reg_data))
        yield riscv.IntRegisterType.infinite_register(~min_reg_data)


@dataclass(frozen=True)
class RISCVAllocateInfiniteRegistersPass(ModulePass):
    """
    Allocates infinite registers to physical registers in the module.
    """

    name = "riscv-allocate-infinite-registers"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in (i for i in op.walk() if isinstance(i, riscv_func.FuncOp)):
            register_stack = RiscvRegisterStack.get()

            # remove registers from stack that are already used in body
            for reg in RegisterAllocatableOperation.iter_all_used_registers(
                func_op.body
            ):
                register_stack.exclude_register(reg)

            phys_reg_by_inf_reg: dict[
                riscv.RISCVRegisterType, riscv.RISCVRegisterType
            ] = {}
            for inner_op in func_op.walk():
                for result in inner_op.results:
                    result_reg = result.type
                    if not isinstance(result_reg, riscv.RISCVRegisterType):
                        raise PassFailedException("Operand type not a register")

                    if (
                        isinstance(result_reg.index, builtin.IntAttr)
                        and result_reg.index.data < 0
                    ):
                        if result_reg in phys_reg_by_inf_reg:
                            # use previously allocated phys reg for this value
                            phys_reg = phys_reg_by_inf_reg[result_reg]
                        else:
                            # allocate a new phys reg
                            try:
                                phys_reg = register_stack.pop(type(result_reg))
                                phys_reg_by_inf_reg[result_reg] = phys_reg
                            except OutOfRegisters:
                                continue

                        Rewriter.replace_value_with_new_type(result, phys_reg)


@dataclass(frozen=True)
class ResolveVirtualRegisters(ModulePass):
    """
    Ensures virtual registers are only in parallel move operations.
    """

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in (i for i in op.walk() if isinstance(i, riscv_func.FuncOp)):
            func_defined_regs: dict[
                type[RISCVRegisterType], OrderedSet[RISCVRegisterType]
            ] = defaultdict(lambda: OrderedSet([]))
            value_by_reg: dict[RISCVRegisterType, SSAValue[builtin.Attribute]] = {}

            next_virtual_reg = _get_next_virtual(func_op)

            for inner_op in func_op.walk():
                # Create iterators of unused regs for current op to spill
                # To get next unused register, call next() with appropriate iterator
                inner_op_used_regs = set(inner_op.result_types).union(
                    inner_op.operand_types
                )
                op_unused_regs_iters = {
                    reg_type: (i for i in regs if i not in inner_op_used_regs)
                    for reg_type, regs in func_defined_regs.items()
                }

                # --- Spill virtual operands ---
                # if any uses are virtual, insert a parallel move op to load the registers
                virtuals = {
                    (operand, operand.type)
                    for operand in inner_op.operands
                    if _is_virtual_reg(operand.type)
                }
                swaps: list[tuple[SSAValue, SSAValue]] = []
                for _, virtual_reg in virtuals:
                    # swap virtual regs with unused physical regs
                    op_not_used_reg = next(op_unused_regs_iters[type(virtual_reg)])
                    swaps.append(
                        (value_by_reg[virtual_reg], value_by_reg[op_not_used_reg])
                    )
                if swaps:
                    # Load virtual registers
                    srcs, dsts = _get_srcs_and_dsts_from_swaps(swaps)  # flattened list
                    load_op = riscv.ParallelMovOp(
                        srcs,
                        dsts,
                        builtin.DenseArrayBase.from_list(builtin.i32, [32] * len(srcs)),
                        free_registers=builtin.ArrayAttr([]),
                    )
                    Rewriter.insert_op(load_op, InsertPoint.before(inner_op))
                    # Replace virtual regs with the physical regs they are loaded into
                    for i, value in enumerate(inner_op.operands):
                        if _is_virtual_reg(value.type):
                            idx = srcs.index(value)  # slow
                            inner_op.operands[i] = load_op.results[idx]

                    # Unload loaded registers back after the op
                    src_types = tuple(i.type for i in srcs)
                    src_types = t.cast(tuple[RISCVRegisterType], src_types)
                    unload_op = riscv.ParallelMovOp(
                        load_op.results,
                        src_types,
                        builtin.DenseArrayBase.from_list(builtin.i32, [32] * len(srcs)),
                        free_registers=builtin.ArrayAttr([]),
                    )
                    Rewriter.insert_op(unload_op, InsertPoint.after(inner_op))
                    # replace all later uses loaded values with the unloaded ones
                    for old_value, new_value in zip(
                        srcs, unload_op.results, strict=True
                    ):
                        old_value.replace_uses_with_if(
                            new_value,
                            lambda use: unload_op.is_before_in_block(use.operation),
                        )

                # --- Spill virtual results ---
                # We need to spill space to put results into,
                # Then spill the result and load back to previous state
                virtual_results = tuple(
                    i for i in inner_op.results if _is_virtual_reg(i.type)
                )
                spill_regs = tuple(
                    next(op_unused_regs_iters[type(vreg.type)])
                    for vreg in virtual_results
                )
                # Create space for results by spilling
                new_virtual_regs = tuple(
                    next(next_virtual_reg) for _ in virtual_results
                )
                if virtual_results:
                    # Spill space to put results in
                    spill_op = riscv.ParallelMovOp(
                        tuple(value_by_reg[i] for i in spill_regs),
                        new_virtual_regs,
                        builtin.DenseArrayBase.from_list(
                            builtin.i32, [32] * len(virtual_results)
                        ),
                        free_registers=builtin.ArrayAttr([]),
                    )
                    Rewriter.insert_op(spill_op, InsertPoint.before(inner_op))
                    # change results types to use newly freed regs
                    virtual_results = tuple(
                        Rewriter.replace_value_with_new_type(result, new_reg)
                        for result, new_reg in zip(
                            virtual_results, spill_regs, strict=True
                        )
                    )
                    # Spill this result
                    virtual_result_regs = tuple(
                        next(next_virtual_reg) for _ in virtual_results
                    )
                    spill_new_op = riscv.ParallelMovOp(
                        virtual_results,
                        virtual_result_regs,
                        builtin.DenseArrayBase.from_list(
                            builtin.i32, [32] * len(virtual_results)
                        ),
                        free_registers=builtin.ArrayAttr([]),
                    )
                    Rewriter.insert_op(spill_new_op, InsertPoint.after(inner_op))
                    # Reload old values
                    load_op = riscv.ParallelMovOp(
                        spill_op.results,
                        spill_regs,
                        builtin.DenseArrayBase.from_list(
                            builtin.i32, [32] * len(virtual_results)
                        ),
                        free_registers=builtin.ArrayAttr([]),
                    )
                    Rewriter.insert_op(load_op, InsertPoint.after(spill_new_op))
                    # Replace uses with virtual_results_regs
                    for result, new_result in zip(
                        virtual_results, spill_new_op.results, strict=True
                    ):
                        result.replace_uses_with_if(
                            new_result,
                            lambda x: x.operation != inner_op
                            and x.operation != spill_new_op,
                        )

                # Add results to func_defined_regs
                for result_type in inner_op.result_types:
                    assert isinstance(result_type, RISCVRegisterType)
                    func_defined_regs[type(result_type)].add(result_type)

                # update value by reg map
                for result in inner_op.results:
                    assert isinstance(result.type, RISCVRegisterType)
                    value_by_reg[result.type] = result
