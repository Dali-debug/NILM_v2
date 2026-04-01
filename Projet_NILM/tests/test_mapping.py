"""
test_mapping.py
---------------
Validation tests for appliance-column mappings in refit_metadata.py.

Covers:
- House 3 ground-truth column assignments
- House 9 ground-truth column assignments
- Canonical alias resolution for key appliances
- Cross-house lookup (same canonical name, different column per house)
"""

import sys
import os

# Allow importing from the parent Projet_NILM directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refit_metadata import (
    get_appliance_column,
    get_house_appliances,
    parse_house_number,
    HOUSE_APPLIANCES,
    APPLIANCE_ALIASES,
)


# ---------------------------------------------------------------------------
# House 3 mapping tests
# ---------------------------------------------------------------------------

class TestHouse3Mapping:
    """Ground truth for House 3 (verified by user):
    0-Aggregate, 1-Toaster, 2-Fridge-Freezer, 3-Freezer, 4-Tumble Dryer,
    5-Dishwasher, 6-Washing Machine, 7-Television, 8-Microwave, 9-Kettle
    """

    def test_kettle_is_appliance9(self):
        assert get_appliance_column(3, "kettle") == "Appliance9", \
            "Kettle must map to Appliance9 in House 3"

    def test_microwave_is_appliance8(self):
        assert get_appliance_column(3, "microwave") == "Appliance8", \
            "Microwave must map to Appliance8 in House 3"

    def test_fridge_is_appliance2(self):
        assert get_appliance_column(3, "fridge") == "Appliance2", \
            "Fridge-Freezer must map to Appliance2 in House 3"

    def test_tv_is_appliance7(self):
        assert get_appliance_column(3, "tv") == "Appliance7", \
            "Television must map to Appliance7 in House 3"

    def test_dishwasher_is_appliance5(self):
        assert get_appliance_column(3, "dishwasher") == "Appliance5", \
            "Dishwasher must map to Appliance5 in House 3"

    def test_washing_machine_is_appliance6(self):
        assert get_appliance_column(3, "washing_machine") == "Appliance6", \
            "Washing Machine must map to Appliance6 in House 3"

    def test_tumble_dryer_is_appliance4(self):
        assert get_appliance_column(3, "tumble_dryer") == "Appliance4", \
            "Tumble Dryer must map to Appliance4 in House 3"

    def test_toaster_is_appliance1(self):
        assert get_appliance_column(3, "toaster") == "Appliance1", \
            "Toaster must map to Appliance1 in House 3"

    def test_freezer_is_appliance3(self):
        assert get_appliance_column(3, "freezer") == "Appliance3", \
            "Freezer must map to Appliance3 in House 3"

    def test_full_house_map_order(self):
        house_map = get_house_appliances(3)
        assert house_map["Appliance1"] == "Toaster"
        assert house_map["Appliance2"] == "Fridge-Freezer"
        assert house_map["Appliance3"] == "Freezer"
        assert house_map["Appliance4"] == "Tumble Dryer"
        assert house_map["Appliance5"] == "Dishwasher"
        assert house_map["Appliance6"] == "Washing Machine"
        assert house_map["Appliance7"] == "Television"
        assert house_map["Appliance8"] == "Microwave"
        assert house_map["Appliance9"] == "Kettle"


# ---------------------------------------------------------------------------
# House 9 mapping tests
# ---------------------------------------------------------------------------

class TestHouse9Mapping:
    """Ground truth for House 9 (verified by user):
    0-Aggregate, 1-Fridge-Freezer, 2-Washer Dryer, 3-Washing Machine,
    4-Dishwasher, 5-Television Site, 6-Microwave, 7-Kettle,
    8-Hi-Fi, 9-Electric Heater
    """

    def test_kettle_is_appliance7(self):
        assert get_appliance_column(9, "kettle") == "Appliance7", \
            "Kettle must map to Appliance7 in House 9"

    def test_microwave_is_appliance6(self):
        assert get_appliance_column(9, "microwave") == "Appliance6", \
            "Microwave must map to Appliance6 in House 9"

    def test_fridge_is_appliance1(self):
        assert get_appliance_column(9, "fridge") == "Appliance1", \
            "Fridge-Freezer must map to Appliance1 in House 9"

    def test_tv_is_appliance5(self):
        assert get_appliance_column(9, "tv") == "Appliance5", \
            "Television Site must map to Appliance5 in House 9"

    def test_dishwasher_is_appliance4(self):
        assert get_appliance_column(9, "dishwasher") == "Appliance4", \
            "Dishwasher must map to Appliance4 in House 9"

    def test_washing_machine_is_appliance3(self):
        assert get_appliance_column(9, "washing_machine") == "Appliance3", \
            "Washing Machine must map to Appliance3 in House 9"

    def test_washer_dryer_is_appliance2(self):
        assert get_appliance_column(9, "washer_dryer") == "Appliance2", \
            "Washer Dryer must map to Appliance2 in House 9"

    def test_hi_fi_is_appliance8(self):
        assert get_appliance_column(9, "hi_fi") == "Appliance8", \
            "Hi-Fi must map to Appliance8 in House 9"

    def test_electric_heater_is_appliance9(self):
        assert get_appliance_column(9, "electric_heater") == "Appliance9", \
            "Electric Heater must map to Appliance9 in House 9"

    def test_full_house_map_order(self):
        house_map = get_house_appliances(9)
        assert house_map["Appliance1"] == "Fridge-Freezer"
        assert house_map["Appliance2"] == "Washer Dryer"
        assert house_map["Appliance3"] == "Washing Machine"
        assert house_map["Appliance4"] == "Dishwasher"
        assert house_map["Appliance5"] == "Television Site"
        assert house_map["Appliance6"] == "Microwave"
        assert house_map["Appliance7"] == "Kettle"
        assert house_map["Appliance8"] == "Hi-Fi"
        assert house_map["Appliance9"] == "Electric Heater"


# ---------------------------------------------------------------------------
# Cross-house mapping tests
# ---------------------------------------------------------------------------

class TestCrossHouseMapping:
    """Verify that the same canonical appliance resolves to different columns
    in different houses (the fundamental requirement for cross-house eval)."""

    def test_kettle_different_columns(self):
        col3 = get_appliance_column(3, "kettle")
        col9 = get_appliance_column(9, "kettle")
        assert col3 != col9, (
            f"Kettle should be on different columns in House 3 ({col3}) "
            f"and House 9 ({col9})"
        )
        assert col3 == "Appliance9"
        assert col9 == "Appliance7"

    def test_microwave_different_columns(self):
        col3 = get_appliance_column(3, "microwave")
        col9 = get_appliance_column(9, "microwave")
        assert col3 != col9, (
            f"Microwave should be on different columns in House 3 ({col3}) "
            f"and House 9 ({col9})"
        )
        assert col3 == "Appliance8"
        assert col9 == "Appliance6"

    def test_fridge_different_columns(self):
        col3 = get_appliance_column(3, "fridge")
        col9 = get_appliance_column(9, "fridge")
        assert col3 != col9
        assert col3 == "Appliance2"
        assert col9 == "Appliance1"

    def test_tv_different_columns(self):
        col3 = get_appliance_column(3, "tv")
        col9 = get_appliance_column(9, "tv")
        assert col3 != col9
        assert col3 == "Appliance7"
        assert col9 == "Appliance5"


# ---------------------------------------------------------------------------
# Alias resolution tests
# ---------------------------------------------------------------------------

class TestAliasResolution:
    """Verify canonical alias normalization."""

    def test_fridge_freezer_label_resolves_to_fridge(self):
        # House 3 has "Fridge-Freezer" at Appliance2 → canonical "fridge" finds it
        assert get_appliance_column(3, "fridge") == "Appliance2"

    def test_television_site_resolves_to_tv(self):
        # House 9 has "Television Site" at Appliance5 → canonical "tv" finds it
        assert get_appliance_column(9, "tv") == "Appliance5"

    def test_unknown_appliance_returns_none(self):
        result = get_appliance_column(3, "nonexistent_device")
        assert result is None

    def test_invalid_house_raises(self):
        import pytest
        with pytest.raises(ValueError):
            get_appliance_column(99, "kettle")


# ---------------------------------------------------------------------------
# parse_house_number tests
# ---------------------------------------------------------------------------

class TestParseHouseNumber:
    def test_parse_house_3(self):
        assert parse_house_number("../Processed_Data_CSV/House_3.csv") == 3

    def test_parse_house_9(self):
        assert parse_house_number("../Processed_Data_CSV/House_9.csv") == 9

    def test_parse_house_windows_path(self):
        assert parse_house_number(r"C:\Data\House_9.csv") == 9

    def test_parse_no_number_raises(self):
        import pytest
        with pytest.raises(ValueError):
            parse_house_number("no_number_here.csv")


# ---------------------------------------------------------------------------
# Run with pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
