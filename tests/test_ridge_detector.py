from pathlib import Path

import pytest

from ridge_detector import RidgeDetector


@pytest.fixture
def img0_path():
    return Path("data/images/img0.jpg")


# At some point, the data should be compared with expected values, but for now, we can
# at least check if we run without errors.
def test_ridge_detector(img0_path):
    detector = RidgeDetector(
        line_widths=6,
        low_contrast=50,
        high_contrast=200,
        min_len=1,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=True,
        correct_pos=True,
    )
    detector.detect_lines(img0_path)


def test_image_contours(img0_path):
    result = RidgeDetector(
        line_widths=6,
        low_contrast=50,
        high_contrast=200,
        min_len=1,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=True,
        correct_pos=True,
    ).detect_lines(img0_path)
    assert result.get_image_contours(show_width=True) is not None
    assert result.get_image_contours(show_width=False) is not None
    assert result.get_binary_contours() is not None
    assert result.get_binary_widths() is not None


def test_to_dataframe(img0_path):
    detector = RidgeDetector(
        line_widths=6,
        low_contrast=50,
        high_contrast=200,
        min_len=1,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=True,
        correct_pos=True,
    )
    result = detector.detect_lines("data/images/img0.jpg")
    df = result.to_dataframe()
    expected_columns = {
        "contour_id",
        "position",
        "x",
        "y",
        "length",
        "line_width",
        "angle_of_normal",
        "class",
    }
    assert not df.empty
    missing_columns = expected_columns - set(df.columns)
    extra_columns = set(df.columns) - expected_columns
    assert not missing_columns, f"Missing columns: {missing_columns}"
    assert not extra_columns, f"Extra columns: {extra_columns}"
