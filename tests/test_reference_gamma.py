"""Comprehensive tests for data.reference.gamma module."""

from __future__ import annotations

from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
import requests

from data.reference import get_updown_asset_ids, get_updown_asset_ids_with_slug
from data.reference.gamma import (
    _extract_asset_ids,
    _get_btc_slug,
    _parse_positive_resolution_value,
    _resolution_to_seconds,
)

# ============================================================================
# RESOLUTION PARSING TESTS
# ============================================================================


class TestResolutionParsing:
    """Tests for resolution parsing helper functions."""

    def test_parse_positive_resolution_value_valid(self):
        """Test parsing valid resolution values."""
        assert _parse_positive_resolution_value("5m") == 5
        assert _parse_positive_resolution_value("1h") == 1
        assert _parse_positive_resolution_value("15m") == 15
        assert _parse_positive_resolution_value("24h") == 24
        assert _parse_positive_resolution_value("7d") == 7

    def test_parse_positive_resolution_value_invalid_format(self):
        """Test parsing invalid resolution formats."""
        with pytest.raises(ValueError, match=r"Invalid resolution format.*missing numeric value"):
            _parse_positive_resolution_value("m")

        with pytest.raises(ValueError, match=r"Invalid resolution format.*non-numeric value"):
            _parse_positive_resolution_value("abcd")

        with pytest.raises(ValueError, match=r"Invalid resolution format.*non-numeric value"):
            _parse_positive_resolution_value("5x5m")

    def test_parse_positive_resolution_value_non_positive(self):
        """Test parsing non-positive resolution values."""
        with pytest.raises(ValueError, match="Resolution must be a positive integer"):
            _parse_positive_resolution_value("0m")

        # Negative numbers are caught by the non-numeric check
        with pytest.raises(ValueError, match=r"Invalid resolution format.*non-numeric value"):
            _parse_positive_resolution_value("-5m")

    def test_resolution_to_seconds_minutes(self):
        """Test converting minute resolutions to seconds."""
        assert _resolution_to_seconds("1m") == 60
        assert _resolution_to_seconds("5m") == 300
        assert _resolution_to_seconds("15m") == 900
        assert _resolution_to_seconds("30m") == 1800

    def test_resolution_to_seconds_hours(self):
        """Test converting hour resolutions to seconds."""
        assert _resolution_to_seconds("1h") == 3600
        assert _resolution_to_seconds("4h") == 14400
        assert _resolution_to_seconds("12h") == 43200
        assert _resolution_to_seconds("24h") == 86400

    def test_resolution_to_seconds_days(self):
        """Test converting day resolutions to seconds."""
        assert _resolution_to_seconds("1d") == 86400
        assert _resolution_to_seconds("7d") == 604800
        assert _resolution_to_seconds("30d") == 2592000

    def test_resolution_to_seconds_invalid_unit(self):
        """Test invalid resolution units."""
        with pytest.raises(ValueError, match="Invalid resolution format"):
            _resolution_to_seconds("5x")

        with pytest.raises(ValueError, match="Invalid resolution format"):
            _resolution_to_seconds("5s")

        with pytest.raises(ValueError, match="Invalid resolution format"):
            _resolution_to_seconds("5")


# ============================================================================
# SLUG GENERATION TESTS
# ============================================================================


class TestSlugGeneration:
    """Tests for BTC slug generation."""

    def test_get_btc_slug_5min(self):
        """Test BTC slug generation with 5-minute resolution."""
        # 1234567890 seconds = 2009-02-13 23:31:30 UTC
        # Rounded down to 5-minute boundary: 1234567800 (23:30:00)
        utctime = 1234567890
        slug = _get_btc_slug(utctime, "5m")
        assert slug == "btc-updown-5m-1234567800"

    def test_get_btc_slug_1hour(self):
        """Test BTC slug generation with 1-hour resolution."""
        # 1234567890 seconds = 2009-02-13 23:31:30 UTC
        # Rounded down to 1-hour boundary: 1234566000 (23:00:00)
        utctime = 1234567890
        slug = _get_btc_slug(utctime, "1h")
        assert slug == "btc-updown-1h-1234566000"

    def test_get_btc_slug_1day(self):
        """Test BTC slug generation with 1-day resolution."""
        # 1234567890 seconds = 2009-02-13 23:31:30 UTC
        # Rounded down to 1-day boundary: 1234483200 (00:00:00 on 2009-02-13)
        utctime = 1234567890
        slug = _get_btc_slug(utctime, "1d")
        assert slug == "btc-updown-1d-1234483200"

    def test_get_btc_slug_exact_boundary(self):
        """Test slug generation when time is exactly on boundary."""
        # Time already on 5-minute boundary
        utctime = 1234567800  # Exactly 23:30:00
        slug = _get_btc_slug(utctime, "5m")
        assert slug == "btc-updown-5m-1234567800"


# ============================================================================
# ASSET ID EXTRACTION TESTS
# ============================================================================


class TestAssetIdExtraction:
    """Tests for extracting asset IDs from API responses."""

    def test_extract_asset_ids_list_format(self):
        """Test extracting asset IDs when clobTokenIds is a list."""
        response_data = [
            {
                "clobTokenIds": ["token1", "token2", "token3"]
            }
        ]
        asset_ids = _extract_asset_ids(response_data)
        assert asset_ids == ["token1", "token2", "token3"]

    def test_extract_asset_ids_string_format(self):
        """Test extracting asset IDs when clobTokenIds is a JSON string."""
        response_data = [
            {
                "clobTokenIds": '["token1", "token2", "token3"]'
            }
        ]
        asset_ids = _extract_asset_ids(response_data)
        assert asset_ids == ["token1", "token2", "token3"]

    def test_extract_asset_ids_empty_response(self):
        """Test extracting asset IDs from empty response."""
        assert _extract_asset_ids([]) == []

    def test_extract_asset_ids_missing_field(self):
        """Test extracting asset IDs when clobTokenIds field is missing."""
        response_data = [{"some_other_field": "value"}]
        asset_ids = _extract_asset_ids(response_data)
        assert asset_ids == []

    def test_extract_asset_ids_invalid_type(self):
        """Test extracting asset IDs when clobTokenIds is invalid type."""
        response_data = [{"clobTokenIds": 12345}]  # Not a list or string
        asset_ids = _extract_asset_ids(response_data)
        assert asset_ids == []

    def test_extract_asset_ids_empty_string(self):
        """Test extracting asset IDs from empty JSON string."""
        response_data = [{"clobTokenIds": "[]"}]
        asset_ids = _extract_asset_ids(response_data)
        assert asset_ids == []


# ============================================================================
# API INTEGRATION TESTS
# ============================================================================


class TestGetUpdownAssetIds:
    """Tests for get_updown_asset_ids function."""

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_success(self, mock_get):
        """Test successful retrieval of asset IDs."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = [
            {"clobTokenIds": ["asset1", "asset2"]}
        ]
        mock_get.return_value = mock_response

        result = get_updown_asset_ids(utctime=1234567890, resolution="5m")

        assert result == ["asset1", "asset2"]
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "btc-updown-5m-1234567800" in call_args[0][0]

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_non_ok_response(self, mock_get):
        """Test handling of non-OK HTTP responses."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.NOT_FOUND
        mock_response.text = "Market not found"
        mock_get.return_value = mock_response

        result = get_updown_asset_ids(utctime=1234567890, resolution="5m")

        assert result == []

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_request_exception(self, mock_get):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.RequestException("Connection error")

        result = get_updown_asset_ids(utctime=1234567890, resolution="5m")

        assert result == []

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_timeout(self, mock_get):
        """Test handling of request timeout."""
        mock_get.side_effect = requests.Timeout("Request timeout")

        result = get_updown_asset_ids(utctime=1234567890, resolution="5m")

        assert result == []

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_empty_response(self, mock_get):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        result = get_updown_asset_ids(utctime=1234567890, resolution="5m")

        assert result == []


class TestGetUpdownAssetIdsWithSlug:
    """Tests for get_updown_asset_ids_with_slug function."""

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_with_slug_success(self, mock_get):
        """Test successful retrieval of slug and asset IDs."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = [
            {"clobTokenIds": ["asset1", "asset2", "asset3"]}
        ]
        mock_get.return_value = mock_response

        slug, asset_ids = get_updown_asset_ids_with_slug(
            utctime=1234567890, resolution="1h"
        )

        assert slug == "btc-updown-1h-1234566000"
        assert asset_ids == ["asset1", "asset2", "asset3"]

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_with_slug_request_error(self, mock_get):
        """Test handling of request errors with slug function."""
        mock_get.side_effect = requests.RequestException("Network error")

        slug, asset_ids = get_updown_asset_ids_with_slug(
            utctime=1234567890, resolution="5m"
        )

        assert slug == ""
        assert asset_ids == []

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_with_slug_non_ok_response(self, mock_get):
        """Test non-OK response with slug function."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_response.text = "Invalid request"
        mock_get.return_value = mock_response

        slug, asset_ids = get_updown_asset_ids_with_slug(
            utctime=1234567890, resolution="1d"
        )

        assert slug == ""
        assert asset_ids == []

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_with_slug_different_resolutions(self, mock_get):
        """Test slug generation with different resolutions."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = [{"clobTokenIds": ["id1"]}]
        mock_get.return_value = mock_response

        # Test 5-minute resolution
        slug_5m, _ = get_updown_asset_ids_with_slug(utctime=1234567890, resolution="5m")
        assert slug_5m == "btc-updown-5m-1234567800"

        # Test 1-hour resolution
        slug_1h, _ = get_updown_asset_ids_with_slug(utctime=1234567890, resolution="1h")
        assert slug_1h == "btc-updown-1h-1234566000"

        # Test 1-day resolution
        slug_1d, _ = get_updown_asset_ids_with_slug(utctime=1234567890, resolution="1d")
        assert slug_1d == "btc-updown-1d-1234483200"

    @patch("data.reference.gamma.requests.get")
    def test_get_updown_asset_ids_with_slug_timeout_param(self, mock_get):
        """Test that timeout parameter is passed to requests.get."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.OK
        mock_response.json.return_value = [{"clobTokenIds": ["id1"]}]
        mock_get.return_value = mock_response

        get_updown_asset_ids_with_slug(utctime=1234567890, resolution="5m")

        # Verify timeout=10 was passed
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs.get("timeout") == 10


# ============================================================================
# INTEGRATION TESTS WITH REAL API
# ============================================================================


class TestGammaAPIIntegration:
    """Integration tests using real Polymarket Gamma API.

    These tests make actual HTTP requests to the Polymarket API
    and verify the integration works end-to-end.
    """

    @pytest.mark.integration
    def test_get_updown_asset_ids_real_api_current_5m_market(self):
        """Test fetching current 5-minute market from real API."""
        from datetime import UTC, datetime  # noqa: PLC0415 - test-local

        # Use current time to get an active market
        utctime = int(datetime.now(tz=UTC).timestamp())

        asset_ids = get_updown_asset_ids(utctime=utctime, resolution="5m")

        # Should get a list of asset IDs (usually 2 for up/down markets)
        assert isinstance(asset_ids, list)
        if asset_ids:  # May be empty if no market exists at this time
            assert len(asset_ids) > 0
            # Asset IDs should be hex strings
            for asset_id in asset_ids:
                assert isinstance(asset_id, str)
                assert len(asset_id) > 0

    @pytest.mark.integration
    def test_get_updown_asset_ids_with_slug_real_api_current_1h_market(self):
        """Test fetching current 1-hour market slug and IDs from real API."""
        from datetime import UTC, datetime  # noqa: PLC0415 - test-local

        utctime = int(datetime.now(tz=UTC).timestamp())

        slug, asset_ids = get_updown_asset_ids_with_slug(utctime=utctime, resolution="1h")

        # Slug should follow expected format
        assert isinstance(slug, str)
        if asset_ids:  # May be empty if no market exists
            assert slug.startswith("btc-updown-1h-")
            assert len(asset_ids) > 0
            # Verify slug contains a valid timestamp
            timestamp_part = slug.split("-")[-1]
            assert timestamp_part.isdigit()

    @pytest.mark.integration
    def test_get_updown_asset_ids_real_api_different_resolutions(self):
        """Test fetching markets with different resolutions from real API."""
        from datetime import UTC, datetime  # noqa: PLC0415 - test-local

        utctime = int(datetime.now(tz=UTC).timestamp())

        # Test multiple resolutions
        for resolution in ["5m", "1h", "1d"]:
            asset_ids = get_updown_asset_ids(utctime=utctime, resolution=resolution)

            assert isinstance(asset_ids, list)
            # We expect some resolutions to have active markets
            if asset_ids:
                assert all(isinstance(aid, str) for aid in asset_ids)

    @pytest.mark.integration
    def test_get_updown_asset_ids_real_api_nonexistent_market(self):
        """Test fetching a market that doesn't exist returns empty list."""
        # Use a very old timestamp where no markets existed
        utctime = 1000000000  # September 2001

        asset_ids = get_updown_asset_ids(utctime=utctime, resolution="5m")

        # Should return empty list for non-existent market
        assert asset_ids == []

    @pytest.mark.integration
    def test_get_updown_asset_ids_with_slug_real_api_validate_structure(self):
        """Test that real API response structure is correctly parsed."""
        from datetime import UTC, datetime  # noqa: PLC0415 - test-local

        utctime = int(datetime.now(tz=UTC).timestamp())

        slug, asset_ids = get_updown_asset_ids_with_slug(utctime=utctime, resolution="5m")

        # Slug and asset_ids should be consistent
        if asset_ids:
            # If we got asset IDs, slug should also be populated
            assert slug != ""
            assert slug.startswith("btc-updown")
        else:
            # If no asset IDs, slug might be populated or empty
            # (depends on whether market exists but has no tokens)
            pass
