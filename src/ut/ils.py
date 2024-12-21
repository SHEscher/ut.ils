"""
Collection of utility functions.

Author: Simon M. Hofmann | <[firstname].[lastname][ät]pm.me> | 2023
"""

# %% Imports
from __future__ import annotations

import difflib
import fileinput
import gc
import gzip
import math
import os
import pickle  # noqa: S403
import platform
import re
import subprocess  # noqa: S404
import sys
import warnings
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Generator, Sequence

import numpy as np
import psutil
import requests

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


# %% Paths < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class DisplayablePath:
    """
    Build DisplayablePath class for tree().

    With honourable mention to 'abstrus':
    https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python

    Note:
    ----
        * This uses recursion. It will raise a RecursionError on really deep folder trees
        * The tree is lazily evaluated. It should behave well on really wide folder trees.
          Immediate children of a given folder are not lazily evaluated, though.

    """

    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path: str | Path, parent_path: str | Path | DisplayablePath, is_last: bool) -> None:
        """Initialise DisplayablePath object."""
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def display_name(self) -> str:
        """Display path name."""
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(
        cls,
        root: str | Path,
        parent: str | Path | DisplayablePath | None = None,
        is_last: bool = False,
        criteria: bool | None = None,
    ) -> list[DisplayablePath]:
        """Display the file tree starting with given root."""
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted([path for path in root.iterdir() if criteria(path)], key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path, parent=displayable_root, is_last=is_last, criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path: str | Path) -> bool:  # noqa: ARG003
        return True

    def displayable(self) -> str:
        """Provide paths which can be displayed."""
        if self.parent is None:
            return self.display_name

        _filename_prefix = self.display_filename_prefix_last if self.is_last else self.display_filename_prefix_middle

        parts = [f"{_filename_prefix!s} {self.display_name!s}"]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle if parent.is_last else self.display_parent_prefix_last)
            parent = parent.parent

        return "".join(reversed(parts))


def tree(directory: str | Path) -> None:
    """
    Generate tree of given directory.

    Use this function the same way as the shell command `tree`.

    This leads to outputs such as:

        directory/
        ├── _static/
        │   ├── embedded/
        │   │   ├── deep_file
        │   │   └── very/
        │   │       └── deep/
        │   │           └── folder/
        │   │               └── very_deep_file
        │   └── less_deep_file
        ├── about.rst
        ├── conf.py
        └── index.rst

    """
    paths = DisplayablePath.make_tree(Path(directory))
    for path in paths:
        print(path.displayable())


def find(
    fname: str,
    folder: str = ".",
    typ: str = "file",
    exclusive: bool = True,
    fullname: bool = True,
    abs_path: bool = False,
    verbose: bool = False,
) -> str | list[str] | None:
    """
    Find file(s) or folder(s) in given folder.

    :param fname: full filename OR consecutive part of it
    :param folder: root folder to search
    :param typ: 'file' or folder 'dir'
    :param exclusive: only return path when only one file was found
    :param fullname: True: consider only files which exactly match the given fname
    :param abs_path: False: return relative path(s); True: return absolute path(s)
    :param verbose: Report findings

    :return: path to file OR list of paths, OR None
    """
    ctn_found = 0
    findings = []
    for root, dirs, files in os.walk(folder):
        search_in = files if typ.lower() == "file" else dirs
        for f in search_in:
            if (fname == f) if fullname else (fname in f):
                ffile = str(Path(root, f))  # found file

                if abs_path:
                    ffile = str(Path(ffile).resolve())

                findings.append(ffile)
                ctn_found += 1

    if exclusive and len(findings) > 1:
        if verbose:
            cprint(string=f"\nFound several {typ}s for given fname='{fname}', please specify:", col="y")
            print("", *findings, sep="\n\t>> ")
        return None

    if not exclusive and len(findings) > 1:
        if verbose:
            cprint(string=f"\nFound several {typ}s for given fname='{fname}', return list of {typ} paths", col="y")
        return findings

    if len(findings) == 0:
        if verbose:
            cprint(string=f"\nDid not find any {typ} for given fname='{fname}', return None", col="y")
        return None

    if verbose:
        cprint(string=f"\nFound this {typ}: '{findings[0]}'", col="y")
    return findings[0]


# %% Timer < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def chop_microseconds(delta: timedelta) -> timedelta:
    """Chop microseconds from given time delta."""
    return delta - timedelta(microseconds=delta.microseconds)


def function_timed(dry_funct: Callable[..., Any] | None = None, ms: bool | None = None) -> Callable[..., Any]:
    """
    Time the processing duration of wrapped function.

    Way to use:

    The following returns the duration without micro-seconds:

    @function_timed
    def abc():
        return 2+2

    The following returns micro-seconds as well:

    @function_timed(ms=True)
    def abcd():
        return 2+2

    :param dry_funct: parameter can be ignored. Results in output without micro-seconds
    :param ms: if micro-seconds should be printed, set to True
    :return:
    """

    def _function_timed(funct: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(funct)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap function to be timed."""
            start_timer = datetime.now()

            # whether to suppress wrapper: use functimer=False in main funct
            w = kwargs.pop("functimer", True)

            output = funct(*args, **kwargs)

            duration = datetime.now() - start_timer

            if w:
                if ms:
                    print(f"\nProcessing time of {funct.__name__}: {duration} [h:m:s:ms]")

                else:
                    print(f"\nProcessing time of {funct.__name__}: {chop_microseconds(duration)} [h:m:s]")

            return output

        return wrapper

    if dry_funct:
        return _function_timed(dry_funct)

    return _function_timed


def loop_timer(
    start_time: datetime, loop_length: int, loop_idx: int, loop_name: str | None = None, add_daytime: bool = False
) -> None:
    """
    Estimate the remaining time to run through given loop.

    Function must be placed at the end of the loop inside.
    Before the loop, take the start time by start_time=datetime.now()
    Provide position within in the loop via enumerate()
    In the form:
        '
        start = datetime.now()
        for idx, ... in enumerate(iterable):
            ... operations ...
            loop_timer(start_time=start, loop_length=len(iterable), loop_idx=idx)
        '
    :param start_time: time at the start of the loop
    :param loop_length: total length of iterable
    :param loop_idx: position within the loop
    :param loop_name: name the loop for print-out
    :param add_daytime: add leading day time to print-out
    """
    _idx = loop_idx
    ll = loop_length

    duration = datetime.now() - start_time
    rest_duration = chop_microseconds(duration / (_idx + 1) * (ll - _idx - 1))

    loop_name = "" if loop_name is None else " of " + loop_name

    now_time = f"{datetime.now().replace(microsecond=0)} | " if add_daytime else ""
    string = (
        f"{now_time}Estimated time to loop over rest{loop_name}: {rest_duration} [hh:mm:ss]\t "
        f"[ {'*' * int((_idx + 1) / ll * 30)}{'.' * (30 - int((_idx + 1) / ll * 30))} ] "
        f"{(_idx + 1) / ll * 100:.2f} %"
    )

    print(string, "\r" if (_idx + 1) != ll else "\n", end="")

    if (_idx + 1) == ll:
        cprint(
            string=f"{now_time}Total duration of loop{loop_name.split(' of')[-1]}: "
            f"{chop_microseconds(duration)} [hh:mm:ss]\n",
            col="b",
        )


def average_time(list_of_timestamps: list, in_timedelta: bool = True) -> float | timedelta:
    """
    Compute average time in a list of time-stamps.

    Necessary for Python 2.

    In Python3 do: np.mean([timedelta(0, 20), ..., timedelta(0, 32)])

    :param list_of_timestamps: list of time-stamps
    :param in_timedelta: whether to return in timedelta-format.
    :return: average time
    """
    mean_time = sum(list_of_timestamps, timedelta()).total_seconds() / len(list_of_timestamps)
    if in_timedelta:
        mean_time = timedelta(seconds=mean_time)
    return mean_time


def try_funct(funct: Callable[..., Any]) -> Callable[..., Any]:
    """
    Try wrapped function, if exception: tell user, but continue.

    Usage:
        @try_funct
        def abc(a, b, c):
            return a+b+c

        # this runs normally
        abc(1, 2, 3)

        # this catches the exception, and continues nonetheless
        abc(1, "no int", 3)
    """

    @wraps(funct)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap function to be tried."""
        try:
            return funct(*args, **kwargs)  # == function()

        except Exception:  # noqa: BLE001
            cprint(string=f"Function {funct.__name__} couldn't be successfully executed!", col="r")

    return wrapper


# %% Normalizer & numerics o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def normalize(
    array: npt.ArrayLike,
    lower_bound: float,
    upper_bound: float,
    global_min: float | None = None,
    global_max: float | None = None,
) -> npt.ArrayLike:
    """
    Min-Max-Scaling: Normalizes the input-array to lower and upper bound.

    :param array: To be transformed array
    :param lower_bound: lower-bound a
    :param upper_bound: upper-bound b
    :param global_min: if the array is part of a larger tensor, normalize w.r.t. global min and ...
    :param global_max: ... global max (i.e., tensor min/max)
    :return: normalized array
    """
    if not lower_bound < upper_bound:
        msg = "lower_bound must be < upper_bound"
        raise AssertionError(msg)

    array = np.array(array)
    a, b = lower_bound, upper_bound

    if global_min is not None:
        if global_min > np.nanmin(array):
            msg = "global_min must be <= np.nanmin(array)"
            raise AssertionError(msg)
        mini = global_min
    else:
        mini = np.nanmin(array)

    if global_max is not None:
        if global_max < np.nanmax(array):
            msg = "global_max must be >= np.nanmax(array)"
            raise AssertionError(msg)
        maxi = global_max
    else:
        maxi = np.nanmax(array)

    return (b - a) * ((array - mini) / (maxi - mini)) + a


def denormalize(
    array: npt.ArrayLike, denorm_minmax: tuple[float, float], norm_minmax: tuple[float, float]
) -> npt.ArrayLike:
    """
    Undo normalization of given array back to previous scaling.

    :param array: array to be denormalized
    :param denorm_minmax: tuple of (min, max) of the denormalized (target) vector
    :param norm_minmax: tuple of (min, max) of the normalized vector
    :return: denormalized value
    """
    array = np.array(array)

    dn_min, dn_max = denorm_minmax
    n_min, n_max = norm_minmax

    if not n_min < n_max:
        msg = "norm_minmax must be tuple (min, max), where min < max"
        raise AssertionError(msg)
    if not dn_min < dn_max:
        msg = "denorm_minmax must be tuple (min, max), where min < max"
        raise AssertionError(msg)

    de_normed_array = (array - n_min) / (n_max - n_min) * (dn_max - dn_min) + dn_min

    return np.array(de_normed_array)


def z_score(array: npt.ArrayLike) -> npt.ArrayLike:
    """
    Create z-score of the given array.

    :return: z-score array
    """
    sub_mean = np.nanmean(array)
    sub_std = np.nanstd(array)
    z_array = (array - sub_mean) / sub_std

    return np.array(z_array)


def get_factors(n: int) -> list[int]:
    """Get factors of given integer."""
    return [i for i in range(1, n + 1) if n % i == 0]


def oom(number: float) -> float:
    """Return order of magnitude of given number."""
    # TODO: what about 10**-x  # noqa: FIX002
    return math.floor(math.log(number, 10))


# %% Sorter  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def natural_sort(list_to_sort: list[str] | tuple[str] | Sequence[str] | Generator[str]) -> list[str]:
    """
    Sort a list naturally.

    For instance:

        ["Head34", "Head100", "Head8"] -> ["Head8", "Head34", "Head100"]

    Source: https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort

    :param list_to_sort: List to sort.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()  # noqa: E731
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa: E731
    return sorted(list_to_sort, key=alphanum_key)


def sort_row_by_row(mat: npt.NDArray[Any], mat_idx: npt.NDArray[int]) -> npt.NDArray[Any]:
    """
    Sort matrix with an index-matrix row by row.

    mat         mat_idx         sorted mat
    [[1,2,3],   [[1,2,0],  ==>  [[2,3,1],
     [4,5,6]]    [2,0,1]]  ==>   [6,4,5]]

    :param mat: matrix to be sorted by rows of mat_idx
    :param mat_idx: matrix with corresponding indices
    :return: sorted matrix
    """
    mat_idx = mat_idx.astype(int)
    if mat.shape != mat_idx.shape:
        msg = "Matrices must have the same shape!"
        raise AssertionError(msg)
    n_rows = mat.shape[0]

    sorted_mat = np.zeros(shape=mat.shape)

    for row in range(n_rows):
        sorted_mat[row, :] = mat[row, :][mat_idx[row, :]]

    return sorted_mat


def inverse_sort_row_by_row(mat: npt.NDArray[Any], mat_idx: npt.NDArray[int]) -> npt.NDArray[Any]:
    """
    Inverse-sort matrix with an index-matrix row by row.

    mat         mat_idx         sorted mat
    [[2,3,1],   [[1,2,0],  ==>  [[1,2,3],
     [6,4,5]]    [2,0,1]]  ==>   [4,5,6]]

    :param mat: matrix to be sorted by rows of mat_idx
    :param mat_idx: matrix with corresponding inverse indices indicating the original order
    :return: sorted matrix
    """
    mat_idx = mat_idx.astype(int)
    if mat.shape != mat_idx.shape:
        msg = "Matrices must have the same shape!"
        raise AssertionError(msg)
    n_rows = mat.shape[0]

    sorted_mat = np.zeros(shape=mat.shape)

    for row in range(n_rows):
        sorted_mat[row, :] = inverse_indexing(arr=mat[row, :], idx=mat_idx[row, :])

    return sorted_mat


def inverse_indexing(arr: npt.NDArray[Any], idx: list[int] | npt.NDArray[np.int_]) -> npt.NDArray[Any]:
    """
    Inverse indexing of the given array.

    For instance, inverse array [16., 2., 4.] to its origin [2., 4., 16.].
    For this we need the index-vector [2, 0, 1].

    (Note, this is different from using: [16., 2., 4.][[1, 2, 0]].)

    :param arr: altered array
    :param idx: former indexing vector
    :return: recovered array
    """
    inv_arr = np.repeat(np.nan, len(arr))
    for i, ix in enumerate(idx):
        inv_arr[ix] = arr[i]
    return inv_arr


def split_in_n_bins(
    a: list[Any] | (tuple[Any] | npt.NDArray[Any]), n: int, attribute_remainder: bool = True
) -> list[Any]:
    """Split in three bins and attribute the remainder equally: [1,2,3,4,5,6,7,8] => [1,2,7], [3,4,8], [5,6]."""
    size = len(a) // n
    split = np.split(a, np.arange(size, len(a), size))

    if attribute_remainder and (len(split) != n):
        att_i = 0
        remainder = list(split.pop(-1))
        while len(remainder) > 0:
            split[att_i] = np.append(split[att_i], remainder.pop(0))
            att_i += 1  # can't overflow
    elif len(split) != n:
        cprint(
            string=f"{len(split[-1])} remainder were put in extra bin. Return {len(split)} bins instead of {n}.",
            col="y",
        )

    return split


def dims_to_rectangularize(arr_length: int) -> tuple[int, int]:
    """Compute dimensions to rectangularize a 1-D array to a 2-D array."""
    if arr_length % 2 != 0:
        msg = "arr_length must be even!"
        raise ValueError(msg)

    factors = get_factors(n=arr_length)
    dim_i = factors[len(factors) // 2]
    dim_j = arr_length // dim_i

    return dim_i, dim_j


def rectangularize_1d_array(arr: npt.NDArray[Any], wide: bool = False) -> npt.NDArray[Any]:
    """
    Rectangularize a 1-D array to a 2-D array.

    :param arr: 1-D array
    :param wide: whether to return a wide array (i.e., more columns than rows)
    :return: 2-D array
    """
    arr = arr.squeeze()
    if np.ndim(arr) != 1:
        msg = "Array must be 1-D!"
        raise ValueError(msg)

    dim_i, dim_j = dims_to_rectangularize(arr_length=len(arr))
    return np.reshape(arr, newshape=(dim_i, dim_j)[:: -1 if wide else 1])


def get_n_cols_and_rows(n_plots: int, square: bool = True, verbose: bool = False) -> tuple[int, int]:
    """Define figure grid-size: with rpl x cpl cells."""
    factors = get_factors(n_plots)
    if len(factors) <= 2 or square:  # prime or square  # noqa: PLR2004
        rpl = 1
        cpl = 1
        while (rpl * cpl) < n_plots:
            if rpl == cpl:
                rpl += 1
            else:
                cpl += 1
    else:
        rpl = factors[len(factors) // 2]
        cpl = n_plots // rpl

    ndiff = rpl * cpl - n_plots
    if ndiff > 0 and verbose:
        cprint(string=f"There will {ndiff} empty plot slots.", col="y")

    return rpl, cpl


def get_string_overlap(s1: str, s2: str) -> str:
    """
    Find the longest overlap between two strings, starting from the left.

    For instance:
        get_string_overlap("abcdef", "abcefg") -> "abc"

    :param s1: first string
    :param s2: second string
    :return overlap between two strings [str]
    """
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, _, size = s.find_longest_match(0, len(s1), 0, len(s2))  # _ = pos_b

    return s1[pos_a : pos_a + size]


# %% Color prints & I/O << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def ol(string: str, wide_bar: bool = True) -> str:
    """
    Create overline over given string or character.

    ol("x") -> "x̅"
    """
    bw = "\u0305" if wide_bar else "\u0304"  # 0305: wide; 0304: smaller
    return "".join([f"{char}{bw}" for char in string])


def ss(string_with_nr: str, sub: bool = True) -> str:
    """
    Translate the following chars '0123456789()' into subscript or superscript and vice versa.

    ss("H2O") -> "H₂O"
    ss("This is x2", sub=False) -> "This is x²"
    """
    # TODO: also for chars  # noqa: FIX002
    # https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts
    subs = str.maketrans("0123456789()₀₁₂₃₄₅₆₇₈₉₍₎", "₀₁₂₃₄₅₆₇₈₉₍₎0123456789()")
    sups = str.maketrans("0123456789()⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾0123456789()")

    return string_with_nr.translate(subs if sub else sups)


class Bcolors:
    r"""
    Use for color-print-commands in console.

    Usage:
    print(Bcolors.HEADER + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
    print(Bcolors.OKBLUE + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)

    For more:

        CSELECTED = '\33[7m'

        CBLACK  = '\33[30m'
        CRED    = '\33[31m'
        CGREEN  = '\33[32m'
        CYELLOW = '\33[33m'
        CBLUE   = '\33[34m'
        CVIOLET = '\33[35m'
        CBEIGE  = '\33[36m'
        CWHITE  = '\33[37m'

        CBLACKBG  = '\33[40m'
        CREDBG    = '\33[41m'
        CGREENBG  = '\33[42m'
        CYELLOWBG = '\33[43m'
        CBLUEBG   = '\33[44m'
        CVIOLETBG = '\33[45m'
        CBEIGEBG  = '\33[46m'
        CWHITEBG  = '\33[47m'

        CGREY    = '\33[90m'
        CBEIGE2  = '\33[96m'
        CWHITE2  = '\33[97m'

        CGREYBG    = '\33[100m'
        CREDBG2    = '\33[101m'
        CGREENBG2  = '\33[102m'

        CYELLOWBG2 = '\33[103m'
        CBLUEBG2   = '\33[104m'
        CVIOLETBG2 = '\33[105m'
        CBEIGEBG2  = '\33[106m'
        CWHITEBG2  = '\33[107m'

    # For preview type:
    for i in [1, 4, 7] + list(range(30, 38)) + list(range(40, 48)) + list(range(90, 98)) + list(
            range(100, 108)):  # range(107+1)
        print(i, '\33[{}m'.format(i) + "ABC & abc" + '\33[0m')
    """

    HEADERPINK = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    UNDERLINE = "\033[4m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"  # this is necessary in the end to reset to default print

    DICT: ClassVar = {"p": HEADERPINK, "b": OKBLUE, "g": OKGREEN, "y": WARNING, "r": FAIL, "ul": UNDERLINE, "bo": BOLD}


def _col_check(col: str) -> None:
    """Check whether the given color is valid."""
    if col.lower() not in {"p", "b", "g", "y", "r"}:
        msg = "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
        raise ValueError(msg)


def cprint(string: str, col: str | None = None, fm: str | None = None, ts: bool = False) -> None:
    """
    Colorize and format print-out. Add leading time-stamp (fs) if required.

    :param string: print message
    :param col: color:'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), OR 'r'(ed)
    :param fm: format: 'ul'(:underline) OR 'bo'(:bold)
    :param ts: add leading time-stamp
    """
    if col:
        col = col.lower()
        _col_check(col=col)
        col = Bcolors.DICT[col]

    if fm:
        fm = fm[0:2].lower()
        if fm not in {"ul", "bo"}:
            msg = "fm must be 'ul'(:underline), 'bo'(:bold)"
            raise ValueError(msg)
        fm = Bcolors.DICT[fm]

    if ts:
        pfx = ""  # collecting leading indent or new line
        while string.startswith(("\n", "\t")):
            pfx += string[:1]
            string = string[1:]
        string = f"{pfx}{datetime.now():%Y-%m-%d %H:%M:%S} | {string}"

    # print given string with given formatting
    print(f"{col if col else ''}{fm if fm else ''}{string}{Bcolors.ENDC}")


def cinput(string: str, col: str | None = None) -> str:
    """Colorize string for input()."""
    if col:
        col = col.lower()
        _col_check(col=col)
        col = Bcolors.DICT[col]

    # input(given string) with given formatting
    return input("{}".format(col if col else "") + string + Bcolors.ENDC)


def block_print() -> None:
    """Disable print outs."""
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115, PTH123, PLW1514


def enable_print() -> None:
    """Restore & enable print outs."""
    # Check if sys.stdout can be closed
    sys.stdout = sys.__stdout__


def suppress_print(func: Callable[..., Any]) -> Callable[..., Any]:
    """Suppresses print within given function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrap function, in which print-out is to be suppressed."""
        block_print()
        output = func(*args, **kwargs)
        enable_print()

        return output

    return wrapper


def true_false_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap print function with true-false-request."""

    @wraps(func)
    def wrapper(*args: str, **kwargs: str) -> bool:
        """Wrap function for true-false-request."""
        func(*args, **kwargs)

        tof = input("(T)rue or (F)alse: ").lower()
        if tof not in {"true", "false", "t", "f"}:
            msg = "Must be 'T', 't' or 'T/true', or 'F', 'f', 'F/false'"
            raise ValueError(msg)
        return tof in "true"

    return wrapper


@true_false_request
def ask_true_false(question: str, col: str = "b") -> None:
    """
    Ask user for input for given True-or-False question.

    :param question: str
    :param col: print-color of question
    :return: answer
    """
    cprint(question, col)


def check_executor(return_shell_bool: bool = False) -> bool | None:
    """
    Check from where the script that executes this function is run from.

    :param return_shell_bool: True: provide boolean about result
    :return: only print (None) OR bool
    """
    ppid = os.getppid()
    shell: bool = psutil.Process(ppid).name() in {"bash", "zsh"}  # TODO: extend for other shells  # noqa: FIX002
    print(
        "Current script{} is executed via: {}{}{}".format(
            f" {Bcolors.WARNING}{sys.argv[0]}{Bcolors.ENDC}" if shell else "",  # platform.sys
            Bcolors.OKBLUE,
            psutil.Process(ppid).name(),
            Bcolors.ENDC,
        )
    )

    return shell if return_shell_bool else None


def cln(factor: int = 1) -> None:
    """Clean the console of interactive shells that do not allow for this."""
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n" * factor)


# %% Text << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def replace_line_in_file(
    path_to_file: str | Path,
    pattern: str | re.Pattern,
    fill_with: str | None,
    whole_line: bool = True,
    verbose: bool = False,
) -> bool | None:
    """
    Replace line with the matching pattern, either the whole line, or only the pattern.

    Delete patterns or entire line with a pattern by setting new_line to None or "".

    :param path_to_file: the path to the text-like file
    :param pattern: the pattern to match per line in file, which follows the convention of re.Pattern
    :param fill_with: new line which replaces the old line; None OR "" will delete pattern/line
    :param whole_line: whether to replace whole line, or only matching pattern
    :param verbose: verbose or not
    :return: whether pattern found or not
    """
    if not Path(path_to_file).exists():
        if verbose:
            cprint(string=f"Couldn't find file '{path_to_file}'!", col="r")
        return None

    fill_with = "" if fill_with is None else fill_with
    if verbose and len(fill_with) == 0:
        if whole_line:
            cprint(string="Given patterns will be deleted from file ...", col="y")
        else:
            cprint(string="Lines with given pattern will be deleted ...", col="y")

    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    found = False

    with fileinput.input(path_to_file, inplace=True) as file:
        for old_line in file:
            match = pattern.search(old_line)
            if match is not None:
                found = True
                if whole_line:
                    if len(fill_with) > 0:  # in the case of == 0: the line will be deleted
                        fill_with = fill_with if fill_with.endswith("\n") else fill_with + "\n"
                    sys.stdout.write(fill_with)
                else:
                    sys.stdout.write(pattern.sub(fill_with, old_line))
            else:
                sys.stdout.write(old_line)

    return found


# %% OS >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def open_folder(path: str | Path) -> None:
    """Open specific folder or file in Finder."""
    if not isinstance(path, (str, Path)):
        msg = f"Path must be string or Path, not {type(path)}"
        raise TypeError(msg)
    if not Path(path).exists():
        msg = f"Path '{path}' doesn't exist."
        raise FileNotFoundError(msg)

    if platform.system() == "Windows":  # for Windows
        os.startfile(path)  # noqa: S606
    elif platform.system() == "Darwin":  # ≈ sys.platform = 'darwin' | for Mac
        subprocess.Popen(["/usr/bin/open", path])  # noqa: S603
    else:  # for 'Linux'
        subprocess.Popen(["/usr/bin/xdg-open", path])  # noqa: S603


def browse_files(initialdir: str | Path | None = None, filetypes: str | None = None) -> str:
    """
    Browse and choose a file from the finder.

    :param initialdir: Where to start the search (ARG MUST BE NAMED 'initialdir')
    :param filetypes: what type of file-ending (suffix, e.g., '*.jpg')
    :return: path to chosen file
    """
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    root = Tk()
    root.withdraw()

    kwargs = {}
    if initialdir:
        kwargs.update({"initialdir": str(initialdir)})
    if filetypes:
        kwargs.update({"filetypes": [(filetypes + " File", "*." + filetypes.lower())]})

    return askopenfilename(parent=root, title="Choose the file", **kwargs)


def delete_dir_and_files(parent_path: str | Path, force: bool = False, verbose: bool = True) -> None:
    """
    Delete the given folder and all subfolders and files.

    os.walk() returns three values on each iteration of the loop:
        i)    The name of the current folder: dir path
        ii)   A list of folders in the current folder: dir names
        iii)  A list of files in the current folder: files

    :param parent_path: path to parent folder
    :param force: True: don't ask to remove
    :param verbose: True: list files will be deleted
    :return: None
    """
    # Print the effected files and subfolders
    if Path(parent_path).exists():
        if not (not verbose and force):
            print(f"\nFollowing (sub-)folders and files of parent folder '{parent_path}' would be deleted:")
            for file in Path(parent_path).glob("**/*"):
                cprint(string=f"{file}", col="b")

        # Double check: Ask whether to delete
        delete = True if force else ask_true_false("Do you want to delete this tree and corresponding files?", col="r")

        if delete:
            # Delete all folders and files in the tree
            for dir_path, _dirn_ames, files in os.walk(parent_path, topdown=False):  # start from bottom
                for file_name in files:
                    if verbose:
                        cprint(string=f"Remove file: {file_name}", col="r")  # f style (for Python > 3.5)
                    Path(dir_path, file_name).unlink()
                if verbose:
                    cprint(string=f"Remove folder: {dir_path}", col="r")
                Path(dir_path).rmdir()  # os.rmdir(dir_path)
        else:
            cprint(string="Tree and files won't be deleted!", col="b")

    else:
        cprint(string=f"Given folder '{parent_path}' doesn't exist.", col="r")


def get_folder_size(parent_dir: str | Path) -> int:
    """Return size of the given parent directory with all subdirectories in bytes."""
    total_size = 0
    for dir_path, _dir_names, file_names in os.walk(str(parent_dir)):
        for f in file_names:
            fp = Path(dir_path, f)
            # skip if it is a symbolic link
            if not fp.is_symlink():
                total_size += fp.stat().st_size  # os.path.getsize(fp)
    return total_size


def bytes2megabytes(n_bytes: int) -> float:
    """Convert bytes to megabytes (MB) in binary (not decimal)."""
    return n_bytes / 10**6


def bytes_to_rep_string(size_bytes: int) -> str:
    """Convert the number of bytes into representative string."""
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 10**3)))
    p = math.pow(10**3, i)
    size_ = round(size_bytes / p, 2)

    return f"{size_} {size_name[i]}"


def check_storage_size(obj: Any, verbose: bool = True) -> int:
    """
    Return storage of variable in the appropriate unit.

    :param obj: any object in workspace
    :param verbose: verbose or not
    :return: object size in bytes
    """
    if isinstance(obj, np.ndarray):
        size_bytes = obj.nbytes
        message = ""
    else:
        size_bytes = sys.getsizeof(obj)
        message = "Only trustworthy for pure python objects, otherwise returns size of view object."

    if verbose:
        print(f"Size of given object: {bytes_to_rep_string(size_bytes=size_bytes)} {message}")

    return size_bytes


def compute_array_size(
    shape: tuple[int, ...] | list[int, ...], dtype: np.dtype | int | float, verbose: bool = False
) -> int:
    """
    Compute the theoretical size of a NumPy array with the given shape and data type.

    :param shape: tuple, shape of the array (e.g., (n_samples, x, y, z))
    :param dtype: data type of the array elements (e.g., np.float32, np.int64, np.uint8, int, float)
    :param verbose: bool, print the size of the array in readable format
    :return: size of the array in bytes
    """
    # Get the size of each element in bytes
    element_size = np.dtype(dtype).itemsize
    # Compute the total number of elements
    num_elements = np.prod(shape)
    # Compute the total size in bytes
    total_size_in_bytes = num_elements * element_size
    if verbose:
        print(f"Size of {dtype.__name__}-array of shape {shape}: {bytes_to_rep_string(total_size_in_bytes)}")
    return total_size_in_bytes


# %% Save objects externally & load them o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@function_timed
def save_obj(obj: Any, name: str, folder: str, hp: bool = True, as_zip: bool = False, save_as: str = "pkl"):
    """
    Save object as pickle or numpy file.

    :param obj: object to be saved
    :param name: name of pickle/numpy file
    :param folder: target folder
    :param hp: True:
    :param as_zip: True: zip file
    :param save_as: default is pickle, can be "npy" for numpy arrays
    """
    # Remove suffix here, if there is e.g. "*.gz.pkl":
    if name.endswith(".gz"):
        name = name[:-3]
        as_zip = True
    if name.endswith((".pkl", ".npy", ".npz")):
        save_as = "pkl" if name.endswith(".pkl") else "npy"
        name = name[:-4]

    p2save = Path(folder) / name

    # Create parent folder if not available
    parent_dir = p2save.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    save_as = save_as.lower()
    if save_as == "pkl":
        open_it, suffix = (gzip.open, ".pkl.gz") if as_zip else (open, ".pkl")
        with open_it(f"{p2save}{suffix}", "wb") as f:  # do not use .with_suffix() since it cuts, e.g., "file.name"
            if hp:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            else:
                p = pickle.Pickler(f)
                p.fast = True
                p.dump(obj)
    elif save_as == "npy":
        if not isinstance(obj, np.ndarray):
            msg = f"Given object is not a numpy array, but '{type(obj)}'!"
            raise TypeError(msg)
        if as_zip:
            np.savez_compressed(file=p2save, arr=obj)
        else:
            np.save(arr=obj, file=p2save)
    else:
        msg = f"Format save_as='{save_as}' unknown!"
        raise ValueError(msg)


@function_timed
def load_obj(name: str, folder: str | PosixPath) -> Any:
    """
    Load a pickle or numpy object into workspace.

    :param name: name of the dataset
    :param folder: target folder
    :return: the loaded object
    """
    # Check whether also a zipped version is available: "*.pkl.gz"
    possible_fm = [".pkl", ".pkl.gz", ".npy", ".npz"]

    def _raise_name_issue() -> None:
        _msg = (
            f"'{folder}' contains too many files which could fit name='{name}'.\nSpecify full name including suffix!"
        )
        raise ValueError(_msg)

    # Check all files in the folder which find the name + *suffix
    found_files = [str(pa) for pa in Path(folder).glob(name + "*")]

    n_max = 2
    if not any(name.endswith(fm) for fm in possible_fm):
        # No file-format found, check folder for files
        if len(found_files) == 0:
            msg = f"In '{folder}' no file with given name='{name}' was found!"
            raise FileNotFoundError(msg)
        if len(found_files) == n_max:  # == 2
            # There can be a zipped & unzipped version, take the unzipped version if applicable
            file_name_overlap = get_string_overlap(found_files[0], found_files[1])
            if file_name_overlap.endswith(".pkl"):  # .pkl and .pkl.gz found
                name = Path(file_name_overlap).name
            elif file_name_overlap.endswith(".np"):  # .npy and .npz found
                name = Path(file_name_overlap).name + "y"  # .npy
            else:  # if the two found files are not of the same file-type
                _raise_name_issue()
        elif len(found_files) > n_max:
            _raise_name_issue()
        else:  # len(found_files) == 1
            name = Path(found_files[0]).name  # un-list

    path_to_file = Path(folder, name)

    # Load and return
    if path_to_file.name.endswith((".pkl", ".pkl.gz")):  # pickle case
        open_it = gzip.open if path_to_file.name.endswith(".gz") else open
        with open_it(path_to_file, "rb") as f:
            return pickle.load(f)
    else:  # numpy case
        file = np.load(path_to_file)
        if isinstance(file, np.lib.npyio.NpzFile):  # name.endswith(".npz"):
            # This asserts that object was saved this way: np.savez_compressed(file=..., arr=obj), as
            # in save_obj() or with np.savez(file=..., obj)
            # If numpy zip (.npz)
            arr_keys = [k for k in file if "arr" in k]
            if len(arr_keys) > 1:
                cprint(
                    string=f"'{path_to_file}' as several array keys! The returned file is of type: {type(file)}",
                    col="y",
                )
            else:
                file = file[arr_keys[0]]
        return file


def memory_in_use() -> int:
    """Check how much memory in currently in use."""
    bytes_in_use = psutil.virtual_memory().used
    print(f"{bytes_to_rep_string(bytes_in_use)} ({psutil.virtual_memory().percent} %) of memory are used.")
    return bytes_in_use


def free_memory(variable: tuple | list | dict | Any | None = None, verbose: bool = False) -> None:
    """
    Free memory from given variable.

    This functions frees memory of (unsigned) objects.
    If a variable (i.e., object) is given, it deletes it from namespace and memory.
    Source: https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python

    Alternatively, use subprocesses like this:
        import concurrent.futures

        def df_processing_func(data):
            ...
            return df

        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            df = executor.map(df_processing_func, [data])[0]

    """
    if verbose:
        print("Before cleaning memory ...")
        memory_in_use()

    if variable is not None:
        if isinstance(variable, (tuple, list)):
            for var in variable:
                del var
        if isinstance(variable, dict):
            for key in list(variable.keys()):
                del variable[key]
        del variable
    gc.collect()

    if verbose:
        print("After cleaning memory ...")
        memory_in_use()


# %% Send messages  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def send_to_mattermost(
    text: str, incoming_webhook: str, username: str, channel: str = "", icon_url: str = ""
) -> requests.Response:
    """
    Send a message to a Mattermost channel.

    For more info check: https://mattermost.com/blog/mattermost-integrations-incoming-webhooks/

    :param text: Text to send to channel (can have Markdown syntax)
    :param incoming_webhook: incoming webhook must be generated via Mattermost settings.
    :param username: Username what should be displayed, sending this message.
    :param channel: webhooks are usually bound to a channel (default).
                    However, another channel can be given if the webhooks allow sending messages there, too.
    :param icon_url: Set an icon for the sender.
    :return: response message
    """
    headers = {}  # {'Content-Type': 'application/json'}
    values = f'{{"text": "{text}",  "channel": "{channel}", "username": "{username}", "icon_url": "{icon_url}"}}'
    return requests.post(incoming_webhook, headers=headers, data=values, timeout=10)


# %% Compute tests  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def deprecated(dry_funct: Callable[..., Any] | None = None, message: str | None = None) -> Callable[..., Any]:
    """
    Mark functions as deprecated with a function decorator.

    It will result in a warning being emitted when the function is used.
    """

    def _deprecated(funct: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(funct)
        def wrapper(*args, **kwargs):
            """Wrap function."""
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(
                message=f"Call to deprecated function {funct.__name__}."
                + (("\n\n" + message + "\n") if message else ""),
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return funct(*args, **kwargs)

        return wrapper

    if dry_funct:
        return _deprecated(dry_funct)

    return _deprecated


def end() -> None:
    """
    Fire *end* firework.

    Can be placed at the end of a script to indicate that the script has finished.

    :return: None
    """
    cprint("\n" + "*<o>*" * 9 + "  END  " + "*<o>*" * 9 + "\n", col="p", fm="bo")


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
