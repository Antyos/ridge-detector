import pytest

from ridge_detector.gui import (
    InvalidLineWidthError,
    parse_line_widths,
)


class TestParseLineWidths:
    def test_parse_empty(self):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths("")

    def test_parse_single(self):
        assert parse_line_widths("5") == [5]

    def test_parse_commas(self):
        assert parse_line_widths("1,2,3") == [1, 2, 3]

    def test_range(self):
        assert parse_line_widths("1:3") == [1, 2, 3]

    def test_range_delta(self):
        assert parse_line_widths("1:5:2") == [1, 3, 5]

    def test_range_stop_only(self):
        assert parse_line_widths(":3") == [1, 2, 3]

    def test_range_start_only(self):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths("1:")

    @pytest.mark.parametrize("width_str", ["1::", ":2:", "1:2:"])
    def test_blank_range(self, width_str):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths(width_str)

    def test_too_many_colons_range(self):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths("1:10:2:3")

    def test_commas_and_range(self):
        assert parse_line_widths("2:4,1") == [2, 3, 4, 1]
        assert parse_line_widths("1:3,5") == [1, 2, 3, 5]

    @pytest.mark.parametrize("width_str", ["1.5", "1.5,2.5", "1.5:3.5"])
    def test_decimals(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths(width_str)

    @pytest.mark.parametrize("width_str", ["a", "1,a", "a,1", "1:3,a", ",", "1,,"])
    def test_non_number_commas(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths(width_str)

    @pytest.mark.parametrize("width_str", ["a:3", "1:a", "1:3:a"])
    def test_non_number_range(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths(width_str)

    @pytest.mark.parametrize("width_str", ["0", "-1", "1,0", "1,-2", "1:0", "1:-3"])
    def test_negative_or_zero(self, width_str: str):
        with pytest.raises(InvalidLineWidthError):
            parse_line_widths(width_str)
