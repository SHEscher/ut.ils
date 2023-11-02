# !/usr/bin/env python3
"""
Tests for ils.py in the `ut.ils` package.

Run me via the shell:

    pytest . --cov; coverage html; open src/tests/coverage_html_report/index.html

"""
# %% Import
import re
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import numpy as np
import pytest
from dotenv import dotenv_values
from ut.ils import (
    _col_check,
    ask_true_false,
    average_time,
    block_print,
    browse_files,
    bytes2megabytes,
    bytes_to_rep_string,
    check_executor,
    check_storage_size,
    cinput,
    cln,
    cprint,
    delete_dir_and_files,
    denormalize,
    deprecated,
    dims_to_rectangularize,
    enable_print,
    end,
    find,
    free_memory,
    function_timed,
    get_factors,
    get_folder_size,
    get_n_cols_and_rows,
    get_string_overlap,
    inverse_indexing,
    inverse_sort_row_by_row,
    load_obj,
    loop_timer,
    memory_in_use,
    normalize,
    ol,
    oom,
    open_folder,
    rectangularize_1d_array,
    replace_line_in_file,
    save_obj,
    send_to_mattermost,
    sort_row_by_row,
    split_in_n_bins,
    ss,
    suppress_print,
    tree,
    try_funct,
    z_score,
)

# %% Global vars, paths & funcs < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
TEMP_DIR = Path("./.test_cache/")
TEMP_FILE = TEMP_DIR / "test_file.txt"


def foo():
    """Test function."""
    print("test")


@pytest.fixture()
def temp_dir_and_file():
    """Create temp dir and file."""
    TEMP_DIR.mkdir(exist_ok=True, mode=0o777, parents=True)
    TEMP_FILE.write_text("Hello there.\nIt's a beautiful day.\n")
    yield TEMP_DIR, TEMP_FILE

    # Tear down
    # Remove all files in temp dir
    for file in TEMP_DIR.glob("*"):
        file.unlink()
    # Remove temp dir
    if TEMP_DIR.is_dir():
        TEMP_DIR.rmdir()


# %% Test ils.py << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_tree(capsys):
    """Test tree() function."""
    assert tree(".") is None
    out, err = capsys.readouterr()
    assert "├── README.md" in out
    assert "├── ut/" in out
    assert "└── ils.py" in out


def test_find(capsys, temp_dir_and_file):
    """Test find() function."""
    results = find(
        fname="setup.py", folder=".", typ="file", exclusive=True, fullname=True, abs_path=True, verbose=True
    )
    assert isinstance(results, str)
    assert results.startswith("/"), f"results should be an absolute path, but is {results}"
    assert results.endswith("setup.py")

    temp_dir, temp_file = temp_dir_and_file
    temp_file2 = temp_dir / Path(temp_file.stem).with_suffix(".py")
    temp_file2.touch()

    results = find(
        fname=temp_file.stem,
        folder=".test_cache",
        typ="file",
        exclusive=True,
        fullname=False,
        abs_path=True,
        verbose=True,
    )
    out, err = capsys.readouterr()
    assert results is None
    assert "Found several files for given fname='" in out

    results = find(
        fname=temp_file.stem,
        folder=".test_cache",
        typ="file",
        exclusive=False,
        fullname=False,
        abs_path=True,
        verbose=True,
    )
    out, err = capsys.readouterr()
    assert isinstance(results, list)
    assert len(results) == 2
    assert "return list of file paths" in out

    results = find(
        fname="NO_FILE", folder=".test_cache", typ="file", exclusive=False, fullname=False, abs_path=True, verbose=True
    )
    out, err = capsys.readouterr()
    assert results is None
    assert "Did not find any file for given fname='NO_FILE', return None" in out

    temp_file2.unlink()


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


def test_average_time():
    """Test average_time() function."""
    start_time = datetime.now()
    avg_t = average_time(
        list_of_timestamps=[datetime.now() - start_time, datetime.now() - start_time], in_timedelta=False
    )
    assert isinstance(avg_t, float)
    assert avg_t < 1

    avg_t = average_time(
        list_of_timestamps=[datetime.now() - start_time, datetime.now() - start_time], in_timedelta=True
    )

    assert isinstance(avg_t, timedelta)
    assert avg_t < timedelta(seconds=2)


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
    """Test normalize() and denormalize() function."""
    org_arr = np.array([1, 2, 3, 4, 5])

    normed_arr = normalize(array=org_arr, lower_bound=-1, upper_bound=1, global_min=0, global_max=6)
    denorm_arr = denormalize(array=normed_arr, denorm_minmax=(0, 6), norm_minmax=(-1, 1))
    np.testing.assert_array_almost_equal(org_arr, denorm_arr)

    normed_arr = normalize(array=org_arr, lower_bound=-1, upper_bound=1, global_min=None, global_max=None)
    denorm_arr = denormalize(array=normed_arr, denorm_minmax=(1, 5), norm_minmax=(-1, 1))
    np.testing.assert_array_almost_equal(org_arr, denorm_arr)

    # Test Errors
    # normalize
    with pytest.raises(AssertionError, match="lower_bound must be < upper_bound"):
        normalize(array=org_arr, lower_bound=99, upper_bound=-50, global_min=None, global_max=None)

    with pytest.raises(AssertionError, match=re.escape("global_min must be <= np.nanmin(array)")):
        normalize(array=org_arr, lower_bound=-1, upper_bound=1, global_min=5, global_max=None)

    with pytest.raises(AssertionError, match=re.escape("global_max must be >= np.nanmax(array)")):
        normalize(array=org_arr, lower_bound=-1, upper_bound=1, global_min=None, global_max=4)

    # denormalize
    with pytest.raises(AssertionError, match=re.escape("norm_minmax must be tuple (min, max), where min < max")):
        denormalize(array=normed_arr, denorm_minmax=(1, 5), norm_minmax=(5, 1))

    with pytest.raises(AssertionError, match=re.escape("denorm_minmax must be tuple (min, max), where min < max")):
        denormalize(array=normed_arr, denorm_minmax=(5, 1), norm_minmax=(-1, 1))


def test_z_score():
    """Test z_score() function."""
    z_arr = z_score(array=[-3, -2, -1, 0, 1, 2, 3])
    assert np.all(z_arr == [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])


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
    with pytest.raises(ValueError, match="math domain error"):
        oom(0)


def test_inverse_sort_row_by_row():
    """Test inverse_sort_row_by_row() function."""
    ipt_mat = np.array([[2, 3, 1], [6, 4, 5]])
    idx_mat = np.array([[1, 2, 0], [2, 0, 1]])
    target_mat = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.all(inverse_sort_row_by_row(mat=ipt_mat, mat_idx=idx_mat) == target_mat)

    with pytest.raises(AssertionError, match="Matrices must have the same shape!"):
        inverse_sort_row_by_row(mat=ipt_mat, mat_idx=np.array([[1, 2, 0]]))


def test_sort_row_by_row():
    """Test sort_row_by_row() function."""
    ipt_mat = np.array([[2, 3, 1], [6, 4, 5]])
    idx_mat = np.array([[2, 0, 1], [1, 2, 0]])
    target_mat = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.all(sort_row_by_row(mat=ipt_mat, mat_idx=idx_mat) == target_mat)

    with pytest.raises(AssertionError, match="Matrices must have the same shape!"):
        sort_row_by_row(mat=ipt_mat, mat_idx=np.array([[1, 2, 0]]))


def test_inverse_indexing():
    """Test inverse_indexing() function."""
    org_arr = np.array([1, 2, 3])  # index: [0, 1, 2]
    re_ordered_arr = np.array([3, 1, 2])
    inv_idx_arr = np.array([2, 0, 1])
    assert np.all(inverse_indexing(arr=re_ordered_arr, idx=inv_idx_arr) == org_arr)


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


def test_dims_to_rectangularize():
    """Test dims_to_rectangularize() function."""
    assert dims_to_rectangularize(arr_length=4) == (2, 2)
    assert dims_to_rectangularize(arr_length=10) == (5, 2)
    assert dims_to_rectangularize(arr_length=16) == (4, 4)

    with pytest.raises(ValueError, match="arr_length must be even!"):
        dims_to_rectangularize(arr_length=9)

    with pytest.raises(IndexError, match="list index out of range"):
        dims_to_rectangularize(arr_length=-4)


def test_rectangularize_1d_array():
    """Test rectangularize_1d_array() function."""
    arr = np.zeros(shape=(4,))
    assert rectangularize_1d_array(arr=arr, wide=False).shape == (2, 2)
    arr = np.zeros(shape=(10,))
    assert rectangularize_1d_array(arr=arr, wide=False).shape == (5, 2)
    assert rectangularize_1d_array(arr=arr, wide=True).shape == (2, 5)
    arr = np.zeros(shape=(1, 16))
    assert rectangularize_1d_array(arr=arr, wide=False).shape == (4, 4)

    arr = np.zeros(shape=(4, 4))
    with pytest.raises(ValueError, match="Array must be 1-D!"):
        rectangularize_1d_array(arr=arr, wide=False)


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


def test_col_check():
    """Test _col_check() function."""
    for col in ("p", "b", "g", "y", "r"):
        assert _col_check(col=col) is None

    with pytest.raises(ValueError, match="col must be"):
        _col_check(col="x")


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

    cprint(string="\n" + string + "\n\tBye!", ts=True)
    out, err = capsys.readouterr()
    assert str(datetime.now())[:4] in out

    with pytest.raises(ValueError, match="col must be"):
        cprint(string=string, col="x")

    with pytest.raises(ValueError, match="fm must be"):
        cprint(string=string, fm="x")


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
        monkeypatch.setattr("builtins.input", lambda _: monkey_answer)  # noqa: B023
        answer = ask_true_false(question="Is this you?")
        assert answer
    for monkey_answer in ["F", "f", "False", "false"]:
        monkeypatch.setattr("builtins.input", lambda _: monkey_answer)  # noqa: B023
        answer = ask_true_false(question="Is this you?")
        assert not answer

    # Wrong answer
    monkeypatch.setattr("builtins.input", lambda _: "X")
    with pytest.raises(ValueError, match=re.escape("Must be 'T', 't' or 'T/true', or 'F', 'f', 'F/false'")):
        ask_true_false(question="Is this you?")


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


def test_replace_line_in_file(capsys, temp_dir_and_file):
    """Test replace_line_in_file() function."""
    temp_dir, temp_file = temp_dir_and_file

    # Replace part of a line: "It's a beautiful day." -> "It's a sunny morning".
    found = replace_line_in_file(
        path_to_file=temp_file,
        pattern="beautiful day",
        fill_with="sunny morning",
        whole_line=False,
        verbose=True,
    )
    # Assert
    assert found
    assert "Hello there.\nIt's a sunny morning.\n" in temp_file.read_text()

    # Replace a whole line that contains the word "Hello"
    found = replace_line_in_file(
        path_to_file=temp_file,
        pattern="Hello",
        fill_with="Good morning!",
        whole_line=True,
        verbose=False,
    )
    assert found
    assert "Good morning!\nIt's a sunny morning.\n" in temp_file.read_text()

    # Delete lines that contain the given pattern
    found = replace_line_in_file(
        path_to_file=temp_file,
        pattern="morning",  # ... which contains 'morning', here both
        fill_with=None,  # -> delete the pattern, i.e., replace it with ''
        whole_line=True,  # -> delete whole line with that pattern
        verbose=True,
    )
    out, err = capsys.readouterr()
    assert found
    assert not temp_file.read_text()  # should have deleted both lines
    assert "Given patterns will be deleted from file ..." in out

    # Fill file again
    temp_file.write_text("Good morning!\nIt's a sunny morning.\n")
    # Delete patterns
    found = replace_line_in_file(
        path_to_file=temp_file,
        pattern="morning",  # ... which contains 'morning', here both
        fill_with=None,  # -> delete pattern
        whole_line=False,  # -> but keep the rest of the line
        verbose=True,
    )
    out, err = capsys.readouterr()
    assert found
    assert "Lines with given pattern will be deleted ..." in out
    assert "Good !\nIt's a sunny .\n" in temp_file.read_text()

    # No existing file
    found = replace_line_in_file(
        path_to_file="TO/NO/WHERE",
        pattern="morning",
        fill_with=None,
        whole_line=True,
        verbose=True,
    )
    out, err = capsys.readouterr()
    assert not found
    assert "Couldn't find file 'TO/NO/WHERE'!" in out


# @pytest.mark.skip(reason="only partially testable")
def test_open_folder():
    """Test open_folder() function."""
    assert open_folder(path=Path.home()) is None

    with pytest.raises(TypeError, match="must be string or Path, not"):
        open_folder(path=1313)

    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        open_folder(path="INTO/NO/WHERE")


# @pytest.mark.skip(reason="only partially testable")
def test_browse_files():
    """Test browse_files() function."""
    file = browse_files(initialdir=Path(__file__).parent, filetypes="py")
    assert isinstance(file, str)
    assert "test_ils.py" in file


def test_delete_dir_and_files(capsys, monkeypatch, temp_dir_and_file):
    """Test delete_dir_and_files() function part 1."""
    temp_dir, temp_file = temp_dir_and_file
    monkeypatch.setattr("builtins.input", lambda _: "F")
    delete_dir_and_files(parent_path=temp_dir, force=False, verbose=True)
    out, err = capsys.readouterr()
    assert "Tree and files won't be deleted!" in out

    monkeypatch.setattr("builtins.input", lambda _: "T")
    delete_dir_and_files(parent_path=temp_dir, force=False, verbose=True)
    out, err = capsys.readouterr()
    assert "Following (sub-)folders and files of parent folder" in out
    assert "Remove file:" in out
    assert "Remove folder:" in out

    # Check force
    temp_dir, temp_file = temp_dir_and_file
    delete_dir_and_files(parent_path=temp_dir, force=True, verbose=False)
    assert not temp_dir.exists()
    assert not temp_file.exists()

    # Delete non-existing folder
    delete_dir_and_files(parent_path="SOME/WHERE", force=True, verbose=False)
    out, err = capsys.readouterr()
    assert "doesn't exist." in out


def test_get_folder_size(temp_dir_and_file):
    """Test get_folder_size() function."""
    temp_dir, temp_file = temp_dir_and_file
    with temp_file.open("wb") as out:
        out.truncate(1024**3)
    size_bytes = get_folder_size(parent_dir=temp_dir)
    assert abs(size_bytes - 1024**3) < 100
    assert "GB" in bytes_to_rep_string(size_bytes)


def test_bytes2megabytes():
    """Test bytes2megabytes() function."""
    assert bytes2megabytes(n_bytes=int(1e7 + 5e6)) == 15.0  # 1e7 == 10**7,
    assert bytes2megabytes(n_bytes=10**6) == 1.0


def test_bytes_to_rep_string():
    """Test bytes_to_rep_string() function."""
    assert bytes_to_rep_string(size_bytes=int(1e7 + 5e6)) == "15.0 MB"
    assert bytes_to_rep_string(size_bytes=int(5e11)) == "500.0 GB"
    assert bytes_to_rep_string(size_bytes=int(1e13 + 5e11)) == "10.5 TB"


def test_check_storage_size(capsys):
    """Test check_storage_size() function."""
    arr = np.ones(shape=(2**8, 2**8, 2**8))  # 256
    size_bytes = check_storage_size(obj=arr, verbose=False)
    assert "MB" in bytes_to_rep_string(size_bytes)
    size_bytes = check_storage_size(obj=arr, verbose=True)
    out, err = capsys.readouterr()
    assert "MB" in out
    assert size_bytes == 2**27  # 8 bytes per position [size_bytes / ((2**8)**3)]
    assert size_bytes == arr.nbytes

    size_bytes = check_storage_size(obj=[1, 2, 3], verbose=True)
    out, err = capsys.readouterr()
    assert "Only trustworthy for pure python objects, otherwise returns size of view object" in out
    assert size_bytes < 100


def test_save_obj(temp_dir_and_file):
    """Test save_obj() function."""
    temp_dir, _ = temp_dir_and_file
    file_name = "test_save_file"
    arr = np.ones(shape=(2**8, 2**8, 2**8))

    # save as pkl
    save_obj(obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=False, save_as="pkl")
    assert (temp_dir / file_name).with_suffix(".pkl").exists()
    (temp_dir / file_name).with_suffix(".pkl").unlink()

    save_obj(obj=arr, name=file_name, folder=temp_dir, hp=False, as_zip=True, save_as="pkl")

    assert (temp_dir / file_name).with_suffix(".pkl.gz").exists()
    (temp_dir / file_name).with_suffix(".pkl.gz").unlink()

    save_obj(obj=arr, name=file_name + ".pkl.gz", folder=temp_dir, hp=False, as_zip=True, save_as="pkl")

    assert (temp_dir / file_name).with_suffix(".pkl.gz").exists()
    (temp_dir / file_name).with_suffix(".pkl.gz").unlink()

    # save as npy
    save_obj(obj=arr, name=file_name + ".npy", folder=temp_dir, hp=True, as_zip=False, save_as="npy")

    assert (temp_dir / file_name).with_suffix(".npy").exists()
    (temp_dir / file_name).with_suffix(".npy").unlink()

    save_obj(obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=True, save_as="npy")

    assert (temp_dir / file_name).with_suffix(".npz").exists()
    (temp_dir / file_name).with_suffix(".npz").unlink()

    with pytest.raises(ValueError, match="Format save_as='not_supported' unknown!"):
        save_obj(obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=True, save_as="not_supported")

    with pytest.raises(TypeError, match="Given object is not a numpy array"):
        save_obj(obj=[1, 2, 3], name=file_name, folder=temp_dir, hp=False, as_zip=False, save_as="npy")


def test_load_obj(capsys, temp_dir_and_file):
    """Test load_obj() function."""
    temp_dir, _ = temp_dir_and_file
    file_name = "test_save_file"
    arr = np.random.randint(low=0, high=255, size=(2**8, 2**8, 2**8), dtype=np.uint8)  # noqa: NPY002

    save_obj(obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=False, save_as="npy")

    arr_loaded = load_obj(name=file_name, folder=temp_dir)
    assert np.all(arr_loaded == arr)

    # Test case when several files are found
    save_obj(obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=True, save_as="npy")  # -> creates a .npz file

    # Here we have npy and npz file with the same name
    arr_loaded = load_obj(name=file_name, folder=temp_dir)
    assert np.all(arr_loaded == arr)

    # Now we have a third file
    (temp_dir / file_name).with_suffix(".npx").touch()  # create additional empty file

    with pytest.raises(ValueError, match="Specify full name including suffix!"):
        load_obj(name=file_name, folder=temp_dir)

    (temp_dir / file_name).with_suffix(".npz").unlink()
    (temp_dir / file_name).with_suffix(".npx").unlink()

    # Test with 2 files but different suffixes
    save_obj(obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=False, save_as="pkl")  # -> creates a .pkl file

    with pytest.raises(ValueError, match="Specify full name including suffix!"):
        load_obj(name=file_name, folder=temp_dir)

    (temp_dir / file_name).with_suffix(".npy").unlink()

    # Now we test 2 pickle files
    save_obj(
        obj=arr, name=file_name, folder=temp_dir, hp=True, as_zip=True, save_as="pkl"
    )  # -> creates a .pkl.gz file

    arr_loaded = load_obj(name=file_name, folder=temp_dir)
    assert np.all(arr_loaded == arr)

    (temp_dir / file_name).with_suffix(".pkl").unlink()
    (temp_dir / file_name).with_suffix(".pkl.gz").unlink()

    # Test when no file is found
    with pytest.raises(FileNotFoundError, match="no file with given name"):
        load_obj(name="NO_FILE", folder=temp_dir)

    # Check npz file which was not saved via save_obj()
    np.savez(temp_dir / file_name, arr)  # -> creates a .npz file
    arr_loaded = load_obj(name=file_name, folder=temp_dir)
    assert np.all(arr_loaded == arr)

    np.savez_compressed(temp_dir / file_name, arr, arr)  # save multiple arrays
    arr_loaded = load_obj(name=file_name, folder=temp_dir)
    out, err = capsys.readouterr()
    assert "as several array keys" in out
    for al_key in arr_loaded:
        assert np.all(arr_loaded[al_key] == arr)

    (temp_dir / file_name).with_suffix(".npz").unlink()


def test_memory_in_use(capsys):
    """Test memory_in_use() function."""
    memory_in_use()
    out, err = capsys.readouterr()
    assert "%) of memory are used." in out


def test_free_memory(capsys):
    """Test free_memory() function."""
    free_memory(variable=None, verbose=False)
    arr = np.ones(shape=(2**10, 2**10, 2**10))
    start_in_use = memory_in_use()
    out, err = capsys.readouterr()
    assert "%) of memory are used." in out

    del arr
    free_memory(variable=None, verbose=True)
    end_in_use = memory_in_use()
    assert end_in_use < start_in_use
    out, err = capsys.readouterr()
    assert "Before cleaning memory" in out
    assert "After cleaning memory ..." in out
    with pytest.raises(NameError, match="cannot access local variable 'arr' where it is not associated with a value"):
        print(arr)  # noqa: F821

    a_dict = {"a": 1, "b": 2}
    assert free_memory(variable=a_dict, verbose=False) is None
    a_list = [1, 2, 3]
    assert free_memory(variable=a_list, verbose=False) is None


def test_send_to_mattermost():
    """Test send_to_mattermost() function."""
    response = send_to_mattermost(
        text=f"Hello from {__file__}!",
        incoming_webhook=dotenv_values(".env")["minerva_webhook_in"],
        username="pytest",
        channel="",
        icon_url="https://a.fsdn.com/allura/s/pytest/"
        "icon?02ce67553d632b0adbd3b9046f4b39f3c3b5294ab102b26770f31385b378fd0b?&w=128",
    )
    assert response.text == "ok"


def test_deprecated():
    """Test deprecated() function."""

    @deprecated
    def abc():
        """Test function."""
        return 1

    with pytest.warns(expected_warning=DeprecationWarning, match="Call to deprecated function abc."):
        result = abc()
        assert result == 1

    @deprecated(message="This function is deprecated!")
    def defg():
        """Test function."""
        return 1

    with pytest.warns(expected_warning=DeprecationWarning, match="This function is deprecated!"):
        result = defg()
        assert result == 1


def test_end(capsys):
    """Test end() function."""
    end()
    out, err = capsys.readouterr()
    assert "*<o>*" in out
    assert "END" in out


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
