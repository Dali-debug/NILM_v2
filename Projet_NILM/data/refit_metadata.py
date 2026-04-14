"""
refit_metadata.py
-----------------
Appliance-column mappings for each REFIT house.

Data is loaded from ``refit_metadata.yaml`` (same directory) so the Python
source stays code-only.  The public API is unchanged:

    HOUSE_APPLIANCES   dict[int, list[str]]
    APPLIANCE_ALIASES  dict[str, list[str]]
    get_appliance_column(house_number, target) -> str | None
    get_house_appliances(house_number) -> dict[str, str]
    parse_house_number(filepath) -> int
    validate_metadata() -> None

Source: REFIT dataset documentation —
  https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
"""

from __future__ import annotations

import os
import re

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Load data from YAML
# ---------------------------------------------------------------------------

def _load_yaml() -> dict:
    """Load and return the raw YAML data from ``refit_metadata.yaml``."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load refit_metadata.yaml. "
            "Install it with: pip install PyYAML"
        ) from exc

    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "refit_metadata.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"Metadata file not found: {yaml_path!r}. "
            "Ensure refit_metadata.yaml is in the same directory as this module."
        )

    with open(yaml_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    return data


_data = _load_yaml()

# Public module-level dicts (same shape as before, integer keys for houses)
HOUSE_APPLIANCES: dict[int, list[str]] = {
    int(k): v for k, v in _data["house_appliances"].items()
}

APPLIANCE_ALIASES: dict[str, list[str]] = dict(_data["appliance_aliases"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_metadata() -> None:
    """Validate the loaded metadata for internal consistency.

    Checks performed
    ----------------
    - Every house has exactly 9 appliance entries.
    - No house has a None or empty appliance name.
    - All alias values are non-empty lists of strings.

    Raises
    ------
    ValueError
        On the first inconsistency found.
    """
    for house, appliances in HOUSE_APPLIANCES.items():
        if len(appliances) != 9:
            raise ValueError(
                f"House {house} has {len(appliances)} appliances; expected 9."
            )
        for idx, name in enumerate(appliances):
            if not name or not isinstance(name, str):
                raise ValueError(
                    f"House {house}, Appliance{idx+1}: invalid name {name!r}."
                )

    for canonical, aliases in APPLIANCE_ALIASES.items():
        if not aliases or not all(isinstance(a, str) for a in aliases):
            raise ValueError(
                f"Alias list for '{canonical}' is empty or contains non-strings."
            )

    logger.debug("Metadata validation passed (%d houses).", len(HOUSE_APPLIANCES))


# Run validation once at import time to catch data errors early.
validate_metadata()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_appliance_column(house_number: int, target: str) -> str | None:
    """Return the CSV column name for *target* appliance in *house_number*.

    Parameters
    ----------
    house_number : int
        REFIT house number (1-21, no 14).
    target : str
        Canonical appliance name, e.g. ``"kettle"``, ``"fridge"``, ``"tv"``.

    Returns
    -------
    str or None
        CSV column name (``"Appliance1"`` … ``"Appliance9"``) or ``None``
        if the appliance is not present in that house.

    Notes
    -----
    Matching is two-pass: exact (case-insensitive) match first, then
    substring match.  This ensures ``"freezer"`` resolves to ``Freezer``
    rather than ``Fridge-Freezer``.
    """
    if house_number not in HOUSE_APPLIANCES:
        raise ValueError(
            f"House {house_number} not found. "
            f"Available houses: {sorted(HOUSE_APPLIANCES.keys())}"
        )

    aliases = APPLIANCE_ALIASES.get(target.lower(), [target.lower()])
    labels = HOUSE_APPLIANCES[house_number]
    labels_lc = [label.lower() for label in labels]

    # First pass: exact match (case-insensitive).
    for idx, label_lc in enumerate(labels_lc):
        if any(alias == label_lc for alias in aliases):
            return f"Appliance{idx + 1}"

    # Second pass: substring match for multi-word / compound labels.
    for idx, label_lc in enumerate(labels_lc):
        if any(alias in label_lc for alias in aliases):
            return f"Appliance{idx + 1}"

    return None


def get_house_appliances(house_number: int) -> dict[str, str]:
    """Return a dict mapping column name → device label for *house_number*.

    Example
    -------
    >>> get_house_appliances(3)
    {'Appliance1': 'Toaster', 'Appliance2': 'Fridge-Freezer', ...}
    """
    if house_number not in HOUSE_APPLIANCES:
        raise ValueError(f"House {house_number} not in HOUSE_APPLIANCES.")
    return {
        f"Appliance{i + 1}": label
        for i, label in enumerate(HOUSE_APPLIANCES[house_number])
    }


def parse_house_number(filepath: str) -> int:
    """Extract the house number from a filepath like ``…/House_3.csv``.

    Parameters
    ----------
    filepath : str
        Path to a REFIT CSV file.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If no integer can be parsed from the filename.
    """
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    if not match:
        raise ValueError(
            f"Cannot parse house number from filename: {basename!r}"
        )
    return int(match.group(1))
