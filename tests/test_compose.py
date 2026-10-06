"""补丁合并（compose）专项测试。

两类验证严格分开：
  * 结构检查：parse_patch 与 compose 在没有任何正文的条件下验证链关系、
    越界、指令合法性与上限（compose 从不接收 A/B/C 正文）；
  * 内容核验：把交付补丁交给 build_target 持最初基线 A 逐字节重建，
    验证内容、长度与 SHA-256 全部等于最后目标 C。

直接运行：
    python3 -m unittest discover -s tests -t . -v
"""

import glob
import hashlib
import os
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rdiff import BLOCK_SIZE, MAX_FILE_SIZE
from rdiff.apply import build_target, ApplyError
from rdiff.compose import compose, ComposeError
from rdiff.delta import make_delta
from rdiff.patch import (
    OP_COPY,
    OP_LITERAL,
    Patch,
    parse_patch,
    serialize_patch,
    op_copy,
    op_literal,
    Op,
)
from rdiff.signature import make_signature
from rdiff.wire import FormatError


def block(seed: int, length: int = BLOCK_SIZE) -> bytes:
    return bytes((seed * 31 + i * 7 + (i * i) // 5) & 0xFF for i in range(length))


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "rdiff", *args], cwd=ROOT, capture_output=True, text=True
    )


class Workdir:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="rdiff-compose-test-")

    def path(self, name):
        return os.path.join(self.tmp, name)

    def write(self, name, data):
        with open(self.path(name), "wb") as f:
            f.write(data)

    def read(self, name):
        with open(self.path(name), "rb") as f:
            return f.read()

    def exists(self, name):
        return os.path.exists(self.path(name))

    def leftovers(self):
        return glob.glob(os.path.join(self.tmp, ".rdiff-compose-*"))


def real_patch(base: bytes, new: bytes) -> Patch:
    return make_delta(make_signature(base), new)


def compose_roundtrip(testcase, a, b, c):
    """真实 delta 生成两份补丁 → 合并 → 结构检查 → 对 A 逐字节核验得 C。"""
    p1 = real_patch(a, b)
    p2 = real_patch(b, c)
    combined = compose(p1, p2)
    assert_combined_well_formed(testcase, combined, len(c))
    # 交付物直接适用于最初基线：不提供 B、C 正文
    out = build_target(a, serialize_patch(combined))
    testcase.assertEqual(len(out), len(c))
    testcase.assertEqual(hashlib.sha256(out).digest(), hashlib.sha256(c).digest())
    testcase.assertEqual(out, c)
    # 头字段：最初基线 + 最后目标
    testcase.assertEqual(combined.base_digest, hashlib.sha256(a).digest())
    testcase.assertEqual(combined.target_digest, hashlib.sha256(c).digest())
    return combined


def assert_combined_well_formed(testcase, patch, expected_len):
    """结构断言（无正文参与）：相邻字面量已合并、COPY ≤256、总长一致。"""
    prev = None
    total = 0
    for op in patch.ops:
        if op.kind == OP_LITERAL:
            testcase.assertIsNot(prev, OP_LITERAL, "相邻 LITERAL 未合并")
            testcase.assertTrue(len(op.data) > 0)
            total += len(op.data)
        else:
            testcase.assertEqual(op.kind, OP_COPY)
            testcase.assertTrue(1 <= op.data <= 256)
            total += op.data
        prev = op.kind
    testcase.assertEqual(total, expected_len)
    testcase.assertEqual(patch.target_length, expected_len)
    # 必须仍是可解析的版本 1 二进制（END 后无垃圾、总长校验通过）
    parse_patch(serialize_patch(patch))


class ComposeRealDeltaTest(unittest.TestCase):
    def test_block_move_insert_two_steps(self):
        # A→B：块搬移 + 插入；B→C：再搬移 + 插入 + 删除，后份大量区间
        # 同时覆盖 B 的搬移块与新增字节
        a = block(1) + block(2) + block(3) + block(4)
        b = a[768:] + b"INSERTED" + a[256:768] + a[:256] + b"tail"
        c = b"X" * 10 + b[300:900] + b"NEW" + b[900:]
        compose_roundtrip(self, a, b, c)

    def test_short_tail_blocks(self):
        # 手工构造补丁精确表达“短尾作为 B 的签名短块被后份 COPY 命中”，
        # 避免随机/确定性块之间的意外弱命中干扰指令结构断言；
        # 内容仍由 build_target 对最初基线逐字节复验。
        rng = random.Random(99)
        for tail_len in (1, 88, 255):
            x = bytes(rng.randrange(256) for _ in range(256))
            y = bytes(rng.randrange(256) for _ in range(256))
            tail = bytes(rng.randrange(256) for _ in range(tail_len))
            a = x + tail
            b = y + tail
            c = b"Q" + tail + y
            first = Patch(
                len(b), hashlib.sha256(a).digest(), hashlib.sha256(b).digest(),
                [op_literal(y), op_copy(256, tail_len)],
            )
            second = Patch(
                len(c), hashlib.sha256(b).digest(), hashlib.sha256(c).digest(),
                [op_literal(b"Q"), op_copy(256, tail_len), op_literal(y)],
            )
            combined = compose(first, second)
            self.assertEqual(
                combined.ops,
                [op_literal(b"Q"), op_copy(256, tail_len), op_literal(y)],
            )
            assert_combined_well_formed(self, combined, len(c))
            self.assertEqual(build_target(a, serialize_patch(combined)), c)

    def test_baseline_shorter_than_block(self):
        a = bytes(range(200))
        b = b"HEAD" + a + b"TAIL"
        c = b[3:130] + b"!!" + b[130:]
        compose_roundtrip(self, a, b, c)

    def test_empty_files_all_combinations(self):
        blob = block(7) + block(8)
        cases = [
            (b"", b"", b""),
            (blob, b"", b""),
            (b"", blob, b""),
            (b"", b"", blob),
            (blob, b"", blob),
            (b"", blob, blob + b"x"),
        ]
        for a, b, c in cases:
            with self.subTest(la=len(a), lb=len(b), lc=len(c)):
                compose_roundtrip(self, a, b, c)

    def test_32kib_final_size(self):
        a = bytes((i * 131 + 17) & 0xFF for i in range(32768))
        b = b"q" * 137 + a[:-137]        # 32768
        c = b"z" * 200 + b[200:]         # 仍为 32768（头部替换、总长不变）
        self.assertEqual(len(b), 32768)
        self.assertEqual(len(c), 32768)
        compose_roundtrip(self, a, b, c)


class ComposeStructureTest(unittest.TestCase):
    def _patch(self, base, target, ops, target_digest=None):
        return Patch(
            target_length=len(target),
            base_digest=hashlib.sha256(base).digest(),
            target_digest=target_digest or hashlib.sha256(target).digest(),
            ops=ops,
        )

    def test_second_copy_spans_copy_literal_copy(self):
        x, y, z = block(1), block(2), block(3)
        a = x + y + z
        # B = z + "!!" + y：前份依次为 COPY(512)、LITERAL、COPY(256)
        b = z + b"!!" + y
        first = self._patch(
            a, b, [op_copy(512, 256), op_literal(b"!!"), op_copy(256, 256)]
        )
        # 后份第一个 COPY 横跨：z 尾 6 字节 + 新增 "!!" + y 头 12 字节
        c = b[250:270] + b[0:10]
        second = self._patch(b, c, [op_copy(250, 20), op_copy(0, 10)])
        combined = compose(first, second)
        self.assertEqual(
            combined.ops,
            [
                op_copy(762, 6),   # z 的尾部映射回基线 512+250
                op_literal(b"!!"),  # 新增字节直接转述
                op_copy(256, 12),  # y 的头映射回基线 256（COPY 取其前 12 字节）
                op_copy(512, 10),  # z 的开头
            ],
        )
        assert_combined_well_formed(self, combined, len(c))
        self.assertEqual(build_target(a, serialize_patch(combined)), c)

    def test_large_copy_chunked_at_256(self):
        a = block(1) + block(2)  # 真实 delta：COPY(0,256), COPY(256,256)
        first = real_patch(a, a)
        self.assertEqual(first.ops, [op_copy(0, 256), op_copy(256, 256)])
        # 手工后份：300 字节 COPY，跨越前份两条 COPY
        c = a[:300]
        second = self._patch(a, c, [op_copy(0, 300)])
        combined = compose(first, second)
        self.assertEqual(combined.ops, [op_copy(0, 256), op_copy(256, 44)])
        self.assertEqual(build_target(a, serialize_patch(combined)), c)

    def test_adjacent_literals_from_three_sources_merge(self):
        a = b"a" * 10
        # B = "MID" + a*10：新增字节在前
        first = self._patch(a, b"MID" + a, [op_literal(b"MID"), op_copy(0, 10)])
        b = b"MID" + a
        # 后份：LITERAL -> COPY 指向前份新增字节 -> LITERAL，三者接壤
        c = b"Z" + b"MID" + b"Q"
        second = self._patch(
            b, c, [op_literal(b"Z"), op_copy(0, 3), op_literal(b"Q")]
        )
        combined = compose(first, second)
        self.assertEqual(combined.ops, [op_literal(b"ZMIDQ")])
        self.assertEqual(build_target(a, serialize_patch(combined)), c)

    def test_partial_overlap_into_literal_segment(self):
        a = b"a" * 5
        first = self._patch(
            a, b"abcdef" + a, [op_literal(b"abcdef"), op_copy(0, 5)]
        )
        b = b"abcdef" + a
        # COPY 只取前份 LITERAL 段中间 3 字节
        second = self._patch(b, b"cde", [op_copy(2, 3)])
        combined = compose(first, second)
        self.assertEqual(combined.ops, [op_literal(b"cde")])
        self.assertEqual(build_target(a, serialize_patch(combined)), b"cde")

    def test_unrelated_chain_rejected(self):
        a, b, other = block(1), block(2), block(3)
        first = real_patch(a, b)
        second = real_patch(other, block(4))
        with self.assertRaises(ComposeError):
            compose(first, second)

    def test_forged_intermediate_digest_rejected(self):
        first = real_patch(block(1), block(2))
        second = real_patch(block(2), block(3))
        forged = Patch(
            second.target_length, b"\x00" * 32, second.target_digest, second.ops
        )
        with self.assertRaises(ComposeError):
            compose(first, forged)

    def test_second_copy_out_of_bounds_rejected(self):
        a = block(1)
        first = real_patch(a, a)  # B 长度 256
        # offset 恰好 == B 长度 / 越过尾 / offset 本身越界
        for ops in (
            [op_copy(256, 1)],
            [op_copy(250, 10)],
            [op_copy(10_000, 1)],
        ):
            with self.subTest(ops=ops):
                bad = Patch(1, hashlib.sha256(a).digest(),
                            hashlib.sha256(b"x").digest(), ops)
                with self.assertRaises(ComposeError):
                    compose(first, bad)

    def test_oversize_intermediate_and_result_rejected(self):
        first_bad = Patch(
            MAX_FILE_SIZE + 1, b"\x00" * 32, b"\x01" * 32,
            [op_literal(b"m" * (MAX_FILE_SIZE + 1))],
        )
        second_empty = Patch(0, b"\x01" * 32, b"\x01" * 32, [])
        with self.assertRaises(ComposeError):
            compose(first_bad, second_empty)

        first_empty = Patch(0, b"\x00" * 32, b"\x00" * 32, [])
        second_big = Patch(
            MAX_FILE_SIZE + 1, b"\x00" * 32, b"\x02" * 32,
            [op_literal(b"n" * (MAX_FILE_SIZE + 1))],
        )
        with self.assertRaises(ComposeError):
            compose(first_empty, second_big)

    def test_corrupt_instructions_rejected(self):
        good_header_first = dict(
            target_length=4, base_digest=b"a" * 32, target_digest=b"b" * 32
        )
        good_header_second = dict(
            target_length=1, base_digest=b"b" * 32, target_digest=b"c" * 32
        )
        for bad_ops_first, bad_ops_second in (
            ([Op(OP_COPY, 0, 0)], [op_literal(b"x")]),       # COPY 长度 0
            ([Op(OP_LITERAL, 0, b"")], [op_literal(b"x")]),  # 空 LITERAL
            ([Op(99, 0, b"")], [op_literal(b"x")]),          # 未知指令
            ([op_literal(b"abcd")], [Op(OP_COPY, 0, 0)]),    # 后份损坏 COPY
        ):
            with self.subTest():
                with self.assertRaises(ComposeError):
                    compose(
                        Patch(ops=bad_ops_first, **good_header_first),
                        Patch(ops=bad_ops_second, **good_header_second),
                    )

    def test_structural_pass_but_content_check_independent(self):
        # 结构全部合法，但后份伪造目标摘要：compose（无正文）必须放行，
        # 最终 apply 持基线做内容核验时必须拒绝——两者不可互相冒充。
        a = b"a" * 10
        b = b"b" * 10
        c = b"c" * 10
        first = real_patch(a, b)
        second = Patch(
            len(c),
            hashlib.sha256(b).digest(),
            hashlib.sha256(b"forged digest").digest(),
            [op_literal(c)],
        )
        combined = compose(first, second)  # 不抛异常
        self.assertEqual(
            combined.target_digest, hashlib.sha256(b"forged digest").digest()
        )
        with self.assertRaises(ApplyError):
            build_target(a, serialize_patch(combined))


class ComposeCliTest(unittest.TestCase):
    def setUp(self):
        self.wd = Workdir()

    def test_cli_end_to_end_then_apply(self):
        a = block(1) + block(2) + block(3)
        b = a[512:] + b"INS" + a[:512]
        c = b"P" + b[10:700] + b"Q" + b[700:]
        self.wd.write("a", a)
        self.wd.write("b", b)
        self.wd.write("c", c)
        cmds = [
            ("signature", "a", "sig_a"),
            ("signature", "b", "sig_b"),
            ("delta", "sig_a", "b", "p1"),
            ("delta", "sig_b", "c", "p2"),
            ("compose", "p1", "p2", "combined"),
            ("apply", "a", "combined", "out"),
        ]
        for cmd in cmds:
            r = self._run(cmd)
            self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertEqual(self.wd.read("out"), c)
        self.assertEqual(len(self.wd.read("out")), len(c))
        self.assertEqual(
            hashlib.sha256(self.wd.read("out")).digest(), hashlib.sha256(c).digest()
        )

    def _run(self, cmd):
        args = [cmd[0]] + [
            a if a in ("signature", "delta", "compose", "apply") else self.wd.path(a)
            for a in cmd[1:]
        ]
        return run_cli(*args)

    def test_failure_preserves_existing_output(self):
        sentinel = b"ORIGINAL COMBINED PATCH - must survive"
        self.wd.write("combined", sentinel)
        a, b, c = block(1), block(2), block(3)
        self.wd.write("p1", serialize_patch(real_patch(a, b)))
        # 链不匹配
        self.wd.write("p2_badchain", serialize_patch(real_patch(b"zz", c)))
        # 损坏字节（截断）
        self.wd.write("p2_trunc", serialize_patch(real_patch(b, c))[:40])
        for bad in ("p2_badchain", "p2_trunc"):
            r = run_cli(
                "compose", self.wd.path("p1"), self.wd.path(bad),
                self.wd.path("combined"),
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(self.wd.read("combined"), sentinel)
            self.assertEqual(self.wd.leftovers(), [])

    def test_failure_does_not_create_new_output(self):
        a, b = block(1), block(2)
        self.wd.write("p1", serialize_patch(real_patch(a, b)))
        self.wd.write("p2", serialize_patch(real_patch(b"qq", block(3))))
        r = run_cli(
            "compose", self.wd.path("p1"), self.wd.path("p2"),
            self.wd.path("never"),
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(self.wd.exists("never"))
        self.assertEqual(self.wd.leftovers(), [])


class ComposeFuzzTest(unittest.TestCase):
    def test_random_chains_byte_exact(self):
        rng = random.Random(20261006)
        for seed in range(40):
            rng.seed(1000 + seed)
            a = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 4000)))

            def mutate(buf):
                n = len(buf)
                chunks, i = [], 0
                while i < n:
                    l = min(rng.randrange(1, 320), n - i)
                    chunks.append((i, l))
                    i += l
                rng.shuffle(chunks)
                chunks = [ch for ch in chunks if rng.random() > 0.15]
                out = bytearray()
                for off, l in chunks:
                    out += buf[off : off + l]
                for _ in range(rng.randrange(5)):
                    p = rng.randrange(len(out) + 1)
                    out[p:p] = bytes(
                        rng.randrange(256)
                        for _ in range(rng.randrange(1, 25))
                    )
                return bytes(out)

            b, c = mutate(a), mutate(mutate(a))
            with self.subTest(seed=seed, la=len(a), lb=len(b), lc=len(c)):
                compose_roundtrip(self, a, b, c)


if __name__ == "__main__":
    unittest.main()
