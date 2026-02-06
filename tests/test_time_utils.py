"""Tests for coastal_calibration.time_utils module."""

from __future__ import annotations

import pytest

from coastal_calibration.time_utils import advance_time, format_forcing_date, parse_date_components


class TestAdvanceTime:
    def test_advance_24_hours(self):
        assert advance_time("2024010100", 24) == "2024010200"

    def test_advance_negative_hours(self):
        assert advance_time("2024010100", -48) == "2023123000"

    def test_leap_year_feb_28(self):
        assert advance_time("2024022800", 24) == "2024022900"

    def test_non_leap_year_feb_28(self):
        assert advance_time("2023022800", 24) == "2023030100"

    def test_advance_zero_hours(self):
        assert advance_time("2024010112", 0) == "2024010112"

    def test_year_boundary_forward(self):
        assert advance_time("2024123123", 1) == "2025010100"

    def test_year_boundary_backward(self):
        assert advance_time("2025010100", -1) == "2024123123"

    def test_month_boundary(self):
        assert advance_time("2024013123", 1) == "2024020100"

    def test_large_advance(self):
        # 2024 is a leap year (366 days = 8784 hours)
        result = advance_time("2024010100", 8784)
        assert result == "2025010100"

    def test_with_extra_characters_in_date(self):
        """The function takes first 10 chars, so extra chars are ignored."""
        assert advance_time("202401010099", 24) == "2024010200"


class TestParseDateComponents:
    def test_basic_parsing(self):
        result = parse_date_components("2024061112")
        assert result["year"] == "2024"
        assert result["month"] == "06"
        assert result["day"] == "11"
        assert result["hour"] == "12"
        assert result["pdy"] == "20240611"
        assert result["cyc"] == "12"

    def test_midnight(self):
        result = parse_date_components("2024010100")
        assert result["hour"] == "00"
        assert result["cyc"] == "00"

    def test_end_of_year(self):
        result = parse_date_components("2024123123")
        assert result["year"] == "2024"
        assert result["month"] == "12"
        assert result["day"] == "31"
        assert result["hour"] == "23"


class TestFormatForcingDate:
    def test_basic(self):
        assert format_forcing_date("2024010112") == "202401011200"

    def test_midnight(self):
        assert format_forcing_date("2024060100") == "202406010000"

    def test_extra_characters(self):
        assert format_forcing_date("202401011200") == "202401011200"
