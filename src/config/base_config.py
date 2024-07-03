# coding: utf-8

"""
pretty printing class
"""

from __future__ import annotations
import os.path as osp
from typing import Tuple


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __repr__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)
