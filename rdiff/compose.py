"""合并两份相邻的版本 1 补丁：A->B 与 B->C 合成 A->C。

只读取两份补丁本身，绝不打开任何文件正文（A/B/C 均不可得）。
合并是**结构验证**：链中间摘要一致、后补丁引用落在前补丁声明的目标
长度内、合并结果满足格式上限。内容摘要的真实性仍由最终 apply 使用
冻结基线核验，合并不越俎代庖，apply 也不因补丁来自合并而省略核验。

映射规则：后补丁的 COPY 引用的是中间版本 B 的区间，而 B 本身由前补丁
的指令序列描述（COPY 引用 A / LITERAL 自带字节）。逐段把 B 区间透过前
补丁翻译为只引用最初基线 A 的 COPY 与字面量。

输出规范与原格式兼容：相邻 LITERAL 合并为一条，COPY 每条最多
BLOCK_SIZE 字节，全部 COPY 只引用最初基线 A。
"""

from . import BLOCK_SIZE
from .patch import OP_COPY, OP_LITERAL, Patch, op_copy, op_literal

MAX_U32 = 0xFFFFFFFF


class ComposeError(Exception):
    """两份补丁无法安全合并（链摘要断裂、越界引用、结果超限等）。"""


def compose(first: Patch, second: Patch) -> Patch:
    """把 A->B 的 first 与 B->C 的 second 合成为 A->C 的补丁。

    任何不一致都抛 ComposeError，调用方保证失败时不触碰已有输出。
    """
    # 链的中间摘要必须一致：first 声称的 B 摘要 == second 声称的基线摘要
    if first.target_digest != second.base_digest:
        raise ComposeError("链摘要断裂：前补丁目标摘要与后补丁基线摘要不一致")

    mid_length = first.target_length  # B 的长度（仅据前补丁声明，不读正文）
    segments = []
    for op in second.ops:
        if op.kind == OP_LITERAL:
            # C 中相对 B 的新增字节，原样进入合并结果
            segments.append(op_literal(op.data))
        elif op.kind == OP_COPY:
            # 引用中间版本 B，必须落在其声明长度内
            offset, length = op.a, op.data
            if offset > mid_length or length > mid_length - offset:
                raise ComposeError("后补丁 COPY 越界引用中间版本")
            segments.extend(_map_through_first(first.ops, offset, length))
        else:
            raise ComposeError(f"后补丁含未知指令: {op.kind}")

    ops = _normalize(segments)
    _check_limits(ops)
    return Patch(
        second.target_length, first.base_digest, second.target_digest, ops
    )


def _map_through_first(first_ops, start: int, length: int) -> list:
    """把 B 的 [start, start+length) 逐段翻译为引用 A 的指令片段。"""
    end = start + length
    out = []
    b_pos = 0
    covered = 0
    for op in first_ops:
        op_len = op.length
        lo = max(start, b_pos)
        hi = min(end, b_pos + op_len)
        if lo < hi:
            if op.kind == OP_COPY:
                # B 的这段来自 A[op.a ...]，平移交集得到 A 中的区间
                out.append(op_copy(op.a + (lo - b_pos), hi - lo))
            elif op.kind == OP_LITERAL:
                # B 的这段是前补丁自带的字节，切片取出
                out.append(op_literal(op.data[lo - b_pos : hi - b_pos]))
            else:
                raise ComposeError(f"前补丁含未知指令: {op.kind}")
            covered += hi - lo
        b_pos += op_len
        if b_pos >= end:
            break
    if covered != length:
        # parse 已保证指令总长 == target_length，此处属于防御
        raise ComposeError("前补丁指令序列无法覆盖后补丁引用区间")
    return out


def _normalize(segments) -> list:
    """相邻字面量合并；基线中相邻的 COPY 合并；COPY 每条最多 BLOCK_SIZE。"""
    ops = []
    for seg in segments:
        if seg.kind == OP_LITERAL:
            if ops and ops[-1].kind == OP_LITERAL:
                ops[-1] = op_literal(ops[-1].data + seg.data)
            else:
                ops.append(seg)
            continue
        offset, length = seg.a, seg.data
        if (
            ops
            and ops[-1].kind == OP_COPY
            and ops[-1].a + ops[-1].data == offset
            and ops[-1].data + length <= BLOCK_SIZE
        ):
            ops[-1] = op_copy(ops[-1].a, ops[-1].data + length)
            continue
        while length > BLOCK_SIZE:  # 前补丁手工 COPY 可能超过块长，切块
            ops.append(op_copy(offset, BLOCK_SIZE))
            offset += BLOCK_SIZE
            length -= BLOCK_SIZE
        ops.append(op_copy(offset, length))
    return ops


def _check_limits(ops) -> None:
    """合并结果必须能用版本 1 格式表达；任何超限都整次拒绝。"""
    for op in ops:
        if op.kind == OP_COPY:
            if not 1 <= op.data <= BLOCK_SIZE:
                raise ComposeError("合并结果 COPY 长度超限")
            if not 0 <= op.a <= MAX_U32:
                raise ComposeError("合并结果 COPY 偏移超出格式上限")
        elif op.kind == OP_LITERAL:
            if not op.data:
                raise ComposeError("合并结果出现空 LITERAL")
        else:
            raise ComposeError(f"合并结果含未知指令: {op.kind}")
