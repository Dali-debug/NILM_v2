"""
data
----
REFIT dataset metadata: appliance-to-column mappings and house configurations.

Public API
----------
    from data.refit_metadata import get_appliance_column, get_house_appliances
    from data.refit_metadata import parse_house_number, HOUSE_APPLIANCES
"""

from .refit_metadata import (
    get_appliance_column,
    get_house_appliances,
    parse_house_number,
    validate_metadata,
    HOUSE_APPLIANCES,
    APPLIANCE_ALIASES,
)

__all__ = [
    "get_appliance_column",
    "get_house_appliances",
    "parse_house_number",
    "validate_metadata",
    "HOUSE_APPLIANCES",
    "APPLIANCE_ALIASES",
]
