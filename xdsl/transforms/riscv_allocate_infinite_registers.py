import typing as t
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ordered_set import OrderedSet

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_stack import OutOfRegisters
from xdsl.backend.register_type import RegisterType
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
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
            used_regs_by_type: defaultdict[
                type[RegisterType], OrderedSet[builtin.Attribute]
            ] = defaultdict(lambda: OrderedSet([]))
            value_by_reg: dict[RegisterType, SSAValue] = {}
            for inner_op in func_op.walk():
                for result in inner_op.results:
                    result_reg = result.type
                    assert isinstance(result_reg, RegisterType)
                    used_regs_by_type[type(result_reg)].add(result_reg)
                    value_by_reg[result_reg] = result

                # if any uses are virtual, insert a parallel move op to load the registers
                virtuals = {
                    operand_reg
                    for operand_reg in inner_op.operand_types
                    if _is_virtual_reg(operand_reg)
                }
                if len(virtuals) > 0:
                    # load all operands
                    # load non-virtual operands as well to ensure they are not spilled
                    op_used_regs = set(
                        operand_type for operand_type in inner_op.operand_types
                    )
                    used_reg_iterators = {
                        k: iter(v - op_used_regs) for k, v in used_regs_by_type.items()
                    }
                    srcs: list[SSAValue] = []
                    dsts: list[riscv.RISCVRegisterType] = []
                    for virtual_reg in virtuals:
                        # swap virtual regs with unused physical regs
                        reg = next(used_reg_iterators[type(virtual_reg)])
                        assert isinstance(reg, riscv.RISCVRegisterType)
                        srcs.append(value_by_reg[virtual_reg])
                        dsts.append(reg)
                        srcs.append(value_by_reg[reg])
                        dsts.append(virtual_reg)

                        load_op = riscv.ParallelMovOp(
                            srcs,
                            dsts,
                            builtin.DenseArrayBase.from_list(
                                builtin.i32, [32] * len(srcs)
                            ),
                            free_registers=builtin.ArrayAttr([]),
                        )
                        Rewriter.insert_op(load_op, InsertPoint.before(inner_op))
                        for old_value, new_value in zip(
                            srcs, load_op.results, strict=True
                        ):
                            old_value.replace_uses_with_if(
                                new_value,
                                lambda use: load_op.is_before_in_block(use.operation),
                            )
                            assert isinstance(new_value.type, RegisterType)
                            value_by_reg[new_value.type] = new_value
