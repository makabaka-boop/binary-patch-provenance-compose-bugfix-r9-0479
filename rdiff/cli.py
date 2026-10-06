"""命令行入口：python -m rdiff {signature,delta,apply} ..."""

import argparse
import sys

from . import MAX_FILE_SIZE
from .apply import ApplyError, apply_patch, atomic_write
from .delta import make_delta
from .compose import ComposeError, compose
from .patch import serialize_patch, parse_patch  # noqa: F401  (parse 供测试/校验)
from .signature import make_signature, parse_signature, serialize_sig
from .wire import FormatError


class InputError(Exception):
    """输入文件缺失、超限或格式非法。"""


def read_limited(path: str, limit: int = MAX_FILE_SIZE, what: str = "文件") -> bytes:
    with open(path, "rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise InputError(f"{what}超过 {limit} 字节上限: {path}")
    return data


def cmd_signature(args) -> None:
    baseline = read_limited(args.baseline, what="基线")
    sig = make_signature(baseline)
    with open(args.signature, "wb") as f:
        f.write(serialize_sig(sig))


def cmd_delta(args) -> None:
    sig_data = read_limited(args.signature, limit=MAX_FILE_SIZE * 2, what="签名")
    new = read_limited(args.new, what="新文件")
    try:
        sig = parse_signature(sig_data)
    except FormatError as exc:
        raise InputError(f"签名非法: {exc}") from exc
    patch = make_delta(sig, new)
    with open(args.patch, "wb") as f:
        f.write(serialize_patch(patch))


def cmd_apply(args) -> None:
    # max_size 仅用于限制基线读取；补丁自身由解析器校验
    read_limited(args.baseline, what="基线")
    apply_patch(args.baseline, args.patch, args.output, MAX_FILE_SIZE)


def cmd_compose(args):
    first = parse_patch(read_limited(args.first, MAX_FILE_SIZE * 4, "前补丁"))
    second = parse_patch(read_limited(args.second, MAX_FILE_SIZE * 4, "后补丁"))
    # 全部结构合并在内存中完成；任何拒绝都发生在触碰输出之前，
    # 再以同目录候选 + os.replace 原子落盘，失败保留既有输出。
    combined = serialize_patch(compose(first, second))
    atomic_write(args.output, combined, prefix=".rdiff-compose-")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rdiff", description="滚动校验二进制差分工具")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("signature", help="由基线生成签名")
    s.add_argument("baseline")
    s.add_argument("signature")
    s.set_defaults(func=cmd_signature)

    d = sub.add_parser("delta", help="由签名与新文件生成补丁（不读基线正文）")
    d.add_argument("signature")
    d.add_argument("new")
    d.add_argument("patch")
    d.set_defaults(func=cmd_delta)

    a = sub.add_parser("apply", help="由基线与补丁重建目标（不读新文件）")
    a.add_argument("baseline")
    a.add_argument("patch")
    a.add_argument("output")
    a.set_defaults(func=cmd_apply)
    c = sub.add_parser("compose", help="合并相邻的两份补丁；不读取基线或中间正文")
    c.add_argument("first")
    c.add_argument("second")
    c.add_argument("output")
    c.set_defaults(func=cmd_compose)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (
        InputError,
        ApplyError,
        ComposeError,
        FormatError,
        FileNotFoundError,
        OSError,
    ) as exc:
        print(f"rdiff: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
