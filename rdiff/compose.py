"""Compose two version-1 patches without opening any file body."""

from .patch import Patch, OP_COPY, OP_LITERAL


def compose(first, second):
    return Patch(
        second.target_length, first.base_digest, second.target_digest, list(second.ops)
    )
