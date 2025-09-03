from pathlib import Path

import pytest

from ridge_detector import RidgeDetector
from ridge_detector.detector import InvalidLineWidthError, RidgeDetectorConfig


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


class TestParseLineWidths:
    def test_parse_empty(self):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths("")

    def test_parse_single(self):
        assert RidgeDetectorConfig.parse_line_widths("5") == [5]

    def test_parse_commas(self):
        assert RidgeDetectorConfig.parse_line_widths("1,2,3") == [1, 2, 3]

    def test_range(self):
        assert RidgeDetectorConfig.parse_line_widths("1:3") == [1, 2, 3]

    def test_range_delta(self):
        assert RidgeDetectorConfig.parse_line_widths("1:5:2") == [1, 3, 5]

    def test_range_stop_only(self):
        assert RidgeDetectorConfig.parse_line_widths(":3") == [1, 2, 3]

    def test_range_start_only(self):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths("1:")

    @pytest.mark.parametrize("width_str", ["1::", ":2:", "1:2:"])
    def test_blank_range(self, width_str):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths(width_str)

    def test_too_many_colons_range(self):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths("1:10:2:3")

    def test_commas_and_range(self):
        assert RidgeDetectorConfig.parse_line_widths("2:4,1") == [2, 3, 4, 1]
        assert RidgeDetectorConfig.parse_line_widths("1:3,5") == [1, 2, 3, 5]

    @pytest.mark.parametrize("width_str", ["1.5", "1.5,2.5", "1.5:3.5"])
    def test_decimals(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths(width_str)

    @pytest.mark.parametrize("width_str", ["a", "1,a", "a,1", "1:3,a", ",", "1,,"])
    def test_non_number_commas(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths(width_str)

    @pytest.mark.parametrize("width_str", ["a:3", "1:a", "1:3:a"])
    def test_non_number_range(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths(width_str)

    @pytest.mark.parametrize("width_str", ["0", "-1", "1,0", "1,-2", "1:0", "1:-3"])
    def test_negative_or_zero(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            RidgeDetectorConfig.parse_line_widths(width_str)
