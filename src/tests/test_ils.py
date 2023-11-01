# !/usr/bin/env python3
"""Tests for ils.py in the `gUt` package."""

# %% Import
from datetime import datetime
from time import sleep

import numpy as np
import pytest
from gut.ils import (
    ask_true_false,
    block_print,
    browse_files,  # noqa: F401
    bytes2megabytes,
    bytes_to_rep_string,
    check_executor,
    check_storage_size,  # noqa: F401
    cinput,
    cln,
    cprint,
    delete_dir_and_files,  # noqa: F401
    denormalize,
    enable_print,
    end,
    find,
    free_memory,  # noqa: F401
    function_timed,
    get_factors,
    get_folder_size,  # noqa: F401
    get_n_cols_and_rows,
    get_string_overlap,
    interpolate_nan,
    inverse_indexing,
    inverse_sort_row_by_row,
    load_obj,  # noqa: F401
    loop_timer,
    memory_in_use,
    normalize,
    ol,
    oom,
    open_folder,  # noqa: F401
    replace_line_in_file,  # noqa: F401
    run_gpu_test,  # noqa: F401
    save_obj,  # noqa: F401
    send_to_mattermost,  # noqa: F401
    sort_row_by_row,
    split_in_n_bins,
    ss,
    suppress_print,
    tree,
    try_funct,
    z_score,
)


# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
def foo():
    """Test function."""
    print("test")


# %% Test ils.py << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_tree(capsys):
    """Test tree() function."""
    assert tree(".") is None
    out, err = capsys.readouterr()
    assert "├── README.md" in out
    assert "├── gut/" in out
    assert "└── ils.py" in out


def test_find():
    """Test find() function."""
    results = find(
        fname="setup.py", folder=".", typ="file", exclusive=True, fullname=True, abs_path=True, verbose=True
    )
    assert isinstance(results, str)
    assert results.startswith("/"), f"results should be an absolute path, but is {results}"
    assert results.endswith("setup.py")


def test_function_timed(capsys):
    """Test function_timed() function."""

    @function_timed(ms=True)
    def test_function():
        sleep(0.25)

    test_function()
    out, err = capsys.readouterr()
    assert "Processing time of test_function: 0:00:00." in out


def test_loop_timer(capsys):
    """Test loop_timer() function."""
    start_time = datetime.now()
    for i in range(5):
        sleep(0.2)
        loop_timer(start_time=start_time, loop_length=5, loop_idx=i, loop_name="Test loop", add_daytime=False)
    out, err = capsys.readouterr()
    assert "Total duration of loop Test loop" in out


def test_try_funct(capsys):
    """Test try_funct() function."""

    @try_funct
    def test_function():
        msg = "test"
        raise Exception(msg)

    test_function()
    out, err = capsys.readouterr()
    assert "Function test_function couldn't be successfully executed!" in out


def test_denormalize():
    """Test denormalize() function."""
    org_arr = [1, 2, 3, 4, 5]
    normed_arr = normalize(array=org_arr, lower_bound=-1, upper_bound=1, global_min=None, global_max=None)
    denorm_arr = denormalize(array=normed_arr, denorm_minmax=(1, 5), norm_minmax=(-1, 1))
    assert all(org_arr == denorm_arr)


def test_z_score():
    """Test z_score() function."""
    z_arr = z_score(array=[-3, -2, -1, 0, 1, 2, 3])
    assert all(z_arr == [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])


def test_get_factors():
    """Test get_factors() function."""
    facts = get_factors(9)
    assert np.all(np.array(facts) == np.array([1, 3, 9]))
    facts = get_factors(7)
    assert np.all(np.array(facts) == np.array([1, 7]))


def test_oom():
    """Test oom() function."""
    assert oom(105) == 2
    assert oom(10**5 + 44) == 5
    assert oom(10**7 + 10**6) == 7
    assert oom(10**0) == 0
    with pytest.raises(ValueError):
        oom(0)


def test_inverse_sort_row_by_row():
    """Test inverse_sort_row_by_row() function."""
    ipt_mat = np.array([[2, 3, 1], [6, 4, 5]])
    idx_mat = np.array([[1, 2, 0], [2, 0, 1]])
    target_mat = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.all(inverse_sort_row_by_row(mat=ipt_mat, mat_idx=idx_mat) == target_mat)


def test_sort_row_by_row():
    """Test sort_row_by_row() function."""
    ipt_mat = np.array([[2, 3, 1], [6, 4, 5]])
    idx_mat = np.array([[2, 0, 1], [1, 2, 0]])
    target_mat = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.all(sort_row_by_row(mat=ipt_mat, mat_idx=idx_mat) == target_mat)


def test_inverse_indexing():
    """Test inverse_indexing() function."""
    org_arr = np.array([1, 2, 3])  # index: [0, 1, 2]
    re_ordered_arr = np.array([3, 1, 2])
    inv_idx_arr = np.array([2, 0, 1])
    assert np.all(inverse_indexing(arr=re_ordered_arr, idx=inv_idx_arr) == org_arr)


def test_interpolate_nan():
    """Test interpolate_nan() function."""
    arr_with_nan = np.array([1, 2, np.nan, 4, 5])
    assert np.all(interpolate_nan(arr_with_nan=arr_with_nan) == np.array([1, 2, 3, 4, 5]))


def test_split_in_n_bins(capsys):
    """Test split_in_n_bins() function."""
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    exp1 = np.array([1, 2, 7])
    exp2 = np.array([3, 4, 8])
    exp3 = np.array([5, 6])
    r1, r2, r3 = split_in_n_bins(a=a, n=3, attribute_remainder=True)
    assert np.all(exp1 == r1)
    assert np.all(exp2 == r2)
    assert np.all(exp3 == r3)

    exp1 = np.array([1, 2])
    exp2 = np.array([3, 4])
    exp3 = np.array([5, 6])
    exp4 = np.array([7, 8])
    r1, r2, r3, r4 = split_in_n_bins(a=a, n=3, attribute_remainder=False)
    out, err = capsys.readouterr()
    assert "2 remainder were put in extra bin. Return 4 bins instead of 3." in out
    assert np.all(exp1 == r1)
    assert np.all(exp2 == r2)
    assert np.all(exp3 == r3)
    assert np.all(exp4 == r4)


def test_get_n_cols_and_rows(capsys):
    """Test get_n_cols_and_rows() function."""
    assert get_n_cols_and_rows(n_plots=9, square=True, verbose=False) == (3, 3)
    assert get_n_cols_and_rows(n_plots=10, square=True, verbose=True) == (4, 3)
    out, err = capsys.readouterr()
    assert "There will 2 empty plot slots." in out
    assert get_n_cols_and_rows(n_plots=10, square=False, verbose=False) == (5, 2)


def test_get_string_overlap():
    """Test get_string_overlap() function."""
    assert get_string_overlap("abcde", "abcef") == "abc"
    assert get_string_overlap("aBCde", "aBCef") == "aBC"


def test_ol():
    """Test ol() function."""
    ol_str = ol(string="x", wide_bar=True)
    assert not ol_str.isascii()
    assert ol_str.startswith("x")
    assert ol_str.isprintable()
    ol_str = ol(string="y", wide_bar=False)
    assert not ol_str.isascii()
    assert ol_str.startswith("y")
    assert ol_str.isprintable()


def test_ss():
    """Test ss() function."""
    ss_str = ss(string_with_nr="N23", sub=True)
    assert not ss_str.isascii()
    assert ss_str.isprintable()
    ss_str = ss(string_with_nr="N23", sub=False)
    assert not ss_str.isascii()
    assert ss_str.isprintable()


def test_cprint(capsys):
    """Test cprint() function."""
    string = "Hello test!"
    cprint(string=string, col="g")
    out, err = capsys.readouterr()
    assert "Hello test!" in out

    cprint(string=string, fm="bo")
    out, err = capsys.readouterr()
    assert "Hello test!" in out

    cprint(string=string, ts=True)
    out, err = capsys.readouterr()
    assert str(datetime.now())[:4] in out


def test_cinput(monkeypatch):
    """Test cinput() function."""
    # This simulates the user entering "Mark" in the terminal:
    monkeypatch.setattr("builtins.input", lambda _: "Some input")
    ipt = cinput(string="Give me some input: ", col="y")
    assert ipt == "Some input"


def test_block_print(capsys):
    """Test block_print() function."""
    foo()
    out, err = capsys.readouterr()
    assert out == "test\n"

    block_print()
    foo()
    out, err = capsys.readouterr()
    assert not out


def test_enable_print():
    """Test enable_print() function."""
    assert enable_print() is None


def test_suppress_print(capsys):
    """Test suppress_print() function."""

    @suppress_print
    def test_function():
        print("test")

    test_function()
    out, err = capsys.readouterr()
    assert not out
    enable_print()


def test_ask_true_false(monkeypatch):
    """Test ask_true_false() function."""
    for monkey_answer in ["T", "t", "True", "true"]:
        monkeypatch.setattr("builtins.input", lambda _: monkey_answer)
        answer = ask_true_false(question="Who this you?")
        assert answer
    for monkey_answer in ["F", "f", "False", "false"]:
        monkeypatch.setattr("builtins.input", lambda _: monkey_answer)
        answer = ask_true_false(question="Who this you?")
        assert not answer


def test_check_executor(capsys):
    """Test check_executor() function."""
    is_shell = check_executor(return_shell_bool=True)
    out, err = capsys.readouterr()
    assert is_shell
    assert "Current script" in out


def test_cln(capsys):
    """Test cln() function."""
    cln()
    out, err = capsys.readouterr()
    assert "\n\n\n\n\n\n\n\n\n\n\n\n\n\n" in out


def test_replace_line_in_file():
    """Test replace_line_in_file() function."""
    # TODO: replace_line_in_file()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_open_folder():
    """Test open_folder() function."""
    # TODO: open_folder()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_browse_files():
    """Test browse_files() function."""
    # TODO: browse_files()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_delete_dir_and_files():
    """Test delete_dir_and_files() function."""
    # TODO: delete_dir_and_files()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_get_folder_size():
    """Test get_folder_size() function."""
    # TODO: get_folder_size()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_bytes2megabytes():
    """Test bytes2megabytes() function."""
    assert bytes2megabytes(n_bytes=int(1e7 + 5e6)) == 15.0  # 1e7 == 10**7,
    assert bytes2megabytes(n_bytes=10**6) == 1.0


def test_bytes_to_rep_string():
    """Test bytes_to_rep_string() function."""
    assert bytes_to_rep_string(size_bytes=int(1e7 + 5e6)) == "15.0 MB"
    assert bytes_to_rep_string(size_bytes=int(5e11)) == "500.0 GB"
    assert bytes_to_rep_string(size_bytes=int(1e13 + 5e11)) == "10.5 TB"


def test_check_storage_size():
    """Test check_storage_size() function."""
    # TODO: check_storage_size()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_save_obj():
    """Test save_obj() function."""
    # TODO: save_obj()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_load_obj():
    """Test load_obj() function."""
    # TODO: load_obj()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_memory_in_use(capsys):
    """Test memory_in_use() function."""
    memory_in_use()
    out, err = capsys.readouterr()
    assert "%) of memory are used." in out


def test_free_memory(capsys):  # noqa: ARG001
    """Test free_memory() function."""
    # TODO: free_memory(verbose=True)  # noqa: FIX002
    # out, err = capsys.readouterr()  # noqa: ERA001
    # assert "%) of memory are used." in out  # noqa: ERA001
    pytest.fail("Not implemented yet!")


def test_send_to_mattermost():
    """Test send_to_mattermost() function."""
    # TODO: send_to_mattermost()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_run_gpu_test():
    """Test run_gpu_test() function."""
    # TODO: update function run_gpu_test()  # noqa: FIX002
    pytest.fail("Not implemented yet!")


def test_end(capsys):
    """Test end() function."""
    end()
    out, err = capsys.readouterr()
    assert "*<o>*" in out
    assert "END" in out


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
