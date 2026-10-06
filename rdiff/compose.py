"""合并两份版本 1 补丁（A→B 与 B→C ⇒ A→C），只读补丁、不读任何正文。

合并是**纯结构**操作，只消费两份补丁自身的字节与指令：
  * 不接收 A/B/C 正文，因此无法（也不试图）验证内容摘要；交付补丁的
    target_digest 只是原样转述后份补丁的声明，其真实性由最终 apply 持
    冻结基线做内容核验——结构检查与内容核验各自独立，不可互相冒充。
  * 补丁链必须对得上：first.target_digest == second.base_digest。
  * 后份补丁的每个 COPY 区间必须完全落在中间版 B 内（越界整次拒绝）。
  * 后份 COPY 引用的 B 区间按前份指令映射回最初基线 A：
      - 落在前份 COPY 段上 → 生成只引用 A 的 COPY（每条至多 256 字节，
        超出自动切分），即使该区间跨越多个前份段；
      - 落在前份 LITERAL 段上（即 B 相对 A 的新增字节）→ 直接转写字节；
      - 相邻 LITERAL（包括“后份 COPY 映射出的字面量”与“后份 LITERAL”
        接壤）合并为一条。
  * 合并结果长度必须等于后份目标长度，且 A/B/C 均不超过 MAX_FILE_SIZE。
任何结构问题抛 ComposeError；调用方须在全部计算完成后再原子落盘，
失败时保留既有输出（见 apply.atomic_write 与 cli.cmd_compose）。
"""

from bisect import bisect_right

from . import MAX_FILE_SIZE
from .patch import OP_COPY, OP_LITERAL, Patch, op_copy, op_literal

# 交付补丁中单条 COPY 的上限（与 delta 产出及补丁格式约定一致）
COPY_CHUNK = 256


class ComposeError(ValueError):
    """两份补丁无法结构合并：链断链、越界引用、损坏指令或结果超限。"""


def _checked_segments(patch, label):
    """逐条校验指令并建立“目标位置 -> 指令段”索引。

    返回 (指令总长, segments)；segments 元素为 (start, end, op)，
    段在目标轴上首尾相接、不留空洞。
    """
    segments = []
    pos = 0
    for op in patch.ops:
        if op.kind == OP_COPY:
            length = op.data
            if not isinstance(length, int) or not (1 <= length <= 0xFFFF):
                raise ComposeError(f"{label}存在长度非法的 COPY")
            if not isinstance(op.a, int) or not (0 <= op.a <= 0xFFFFFFFF):
                raise ComposeError(f"{label}存在偏移非法的 COPY")
        elif op.kind == OP_LITERAL:
            length = len(op.data)
            if length == 0:
                raise ComposeError(f"{label}存在空 LITERAL")
        else:
            raise ComposeError(f"{label}存在未知指令类型: {op.kind!r}")
        segments.append((pos, pos + length, op))
        pos += length
    if pos != patch.target_length:
        raise ComposeError(f"{label}指令总长与目标长度不符")
    return pos, segments


def compose(first, second):
    """把 first(A→B) 与 second(B→C) 结构合并为 A→C；只读两个补丁对象。"""
    # 1) 链关系：中间版摘要必须一致，否则两份补丁没有对应关系
    if first.target_digest != second.base_digest:
        raise ComposeError(
            "补丁链中间摘要不一致：前份目标与后份基线不是同一份中间版"
        )

    # 2) 结构校验 + 建立 B 轴上的段索引
    mid_len, first_segments = _checked_segments(first, "前补丁")
    final_len, _ = _checked_segments(second, "后补丁")
    if mid_len > MAX_FILE_SIZE:
        raise ComposeError(f"中间版超过 {MAX_FILE_SIZE} 字节上限")
    if final_len > MAX_FILE_SIZE:
        raise ComposeError(f"合并结果超过 {MAX_FILE_SIZE} 字节上限")

    out_ops = []
    emitted = 0

    def emit_literal(payload: bytes) -> None:
        nonlocal emitted
        if out_ops and out_ops[-1].kind == OP_LITERAL:
            out_ops[-1] = op_literal(out_ops[-1].data + payload)
        else:
            out_ops.append(op_literal(payload))
        emitted += len(payload)

    def emit_copy(offset: int, length: int) -> None:
        # 单条 COPY 至多 256 字节；整段复用也切成多条
        nonlocal emitted
        while length > 0:
            take = min(length, COPY_CHUNK)
            out_ops.append(op_copy(offset, take))
            offset += take
            length -= take
            emitted += take

    starts = [s for s, _, _ in first_segments]

    # 3) 逐条翻译后份指令
    for op in second.ops:
        if op.kind == OP_LITERAL:
            # 新增字节无需借助正文：载荷就在补丁里
            emit_literal(bytes(op.data))
            continue

        o, l = op.a, op.data
        # 后份 COPY 只能引用中间版 B，越界（含 offset 恰好等于 B 长度）整次拒绝
        if o > mid_len or l > mid_len - o:
            raise ComposeError("后补丁 COPY 越界引用中间版")

        # 该 B 区间可横跨多个前份段（搬移块边界、插入字节等）
        idx = bisect_right(starts, o) - 1
        cur, end = o, o + l
        while cur < end:
            seg_start, seg_end, fop = first_segments[idx]
            sub = cur - seg_start
            take = min(seg_end - cur, end - cur)
            if fop.kind == OP_LITERAL:
                # B 在这段是新增字节：映射后仍是字面量
                emit_literal(bytes(fop.data[sub : sub + take]))
            else:
                # B 在这段复用 A：映射回最初基线偏移，长度超限再切分
                emit_copy(fop.a + sub, take)
            cur += take
            idx += 1

    # 4) 总长兜底（正常情况下由两段结构恒等推出，仅作防御）
    if emitted != final_len:
        raise ComposeError("合并结果长度与后补丁目标长度不符")

    return Patch(
        target_length=final_len,
        base_digest=first.base_digest,
        target_digest=second.target_digest,
        ops=out_ops,
    )
