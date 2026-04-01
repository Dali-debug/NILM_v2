"""
refit_metadata.py
-----------------
Appliance-column mappings for each REFIT house.

For each house the REFIT CSV contains columns:
  Time, Unix, Aggregate, Appliance1, Appliance2, ..., Appliance9

This module maps the generic column names (Appliance1-9) to the actual
device names for every house in the dataset.

Source: REFIT dataset documentation —
  https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned
"""

# Canonical appliance names used for matching in the pipeline.
# When searching for a target appliance (e.g. "kettle"), any label that
# CONTAINS one of these keys (case-insensitive) will be selected.
APPLIANCE_ALIASES = {
    "kettle":    ["kettle"],
    "microwave": ["microwave"],
    "fridge":    ["fridge", "fridge-freezer", "fridge freezer", "fridgefreezer"],
    "tv":        ["television", "tv", "television site"],
    "washing_machine": ["washing machine"],
    "washer_dryer": ["washer dryer", "washer-dryer"],
    "dishwasher": ["dishwasher"],
    "tumble_dryer": ["tumble dryer", "tumbledryer"],
    "toaster":   ["toaster"],
    "freezer":   ["freezer"],
    "computer":  ["computer", "desktop", "laptop"],
    "hi_fi":     ["hi-fi", "hifi", "hi fi"],
    "electric_heater": ["electric heater", "electricheater"],
}

# Per-house appliance lists.
# Index 0 → Appliance1, Index 1 → Appliance2, …, Index 8 → Appliance9
HOUSE_APPLIANCES = {
    1:  [
        "Fridge",
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "Computer",
        "Television",
        "Microwave",
        "Kettle",
        "Toaster",
    ],
    2:  [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Freezer",
        "Computer",
        "Television",
        "Microwave",
        "Kettle",
    ],
    3:  [
        "Toaster",
        "Fridge-Freezer",
        "Freezer",
        "Tumble Dryer",
        "Dishwasher",
        "Washing Machine",
        "Television",
        "Microwave",
        "Kettle",
    ],
    4:  [
        "Fridge",
        "Freezer",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
    ],
    5:  [
        "Fridge-Freezer",
        "Freezer",
        "Washing Machine",
        "Washing Machine 2",
        "Dishwasher",
        "Tumble Dryer",
        "Television",
        "Kettle",
        "Microwave",
    ],
    6:  [
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
    7:  [
        "Fridge",
        "Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Television",
        "Microwave",
        "Kettle",
        "Toaster",
    ],
    8:  [
        "Fridge-Freezer",
        "Freezer",
        "Washing Machine",
        "Washing Machine 2",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Microwave",
        "Television",
    ],
    9:  [
        "Fridge-Freezer",
        "Washer Dryer",
        "Washing Machine",
        "Dishwasher",
        "Television Site",
        "Microwave",
        "Kettle",
        "Hi-Fi",
        "Electric Heater",
    ],
    10: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
    11: [
        "Fridge",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
    ],
    12: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
    13: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Freezer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
    ],
    15: [
        "Fridge",
        "Freezer",
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
    ],
    16: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
    17: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Unknown",
        "Unknown 2",
    ],
    18: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
    19: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Unknown",
        "Unknown 2",
    ],
    20: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
    21: [
        "Fridge-Freezer",
        "Washing Machine",
        "Dishwasher",
        "Tumble Dryer",
        "Kettle",
        "Television",
        "Microwave",
        "Toaster",
        "Computer",
    ],
}


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
        The CSV column name (``"Appliance1"`` … ``"Appliance9"``) that
        corresponds to the requested appliance, or ``None`` if not found.

    Notes
    -----
    Matching is two-pass: an exact (case-insensitive) match is attempted
    first, then a substring match.  This ensures that e.g. ``"freezer"``
    resolves to the ``Freezer`` label rather than ``Fridge-Freezer``.
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
    # Prevents "freezer" from matching "Fridge-Freezer" before "Freezer".
    for idx, label_lc in enumerate(labels_lc):
        if any(alias == label_lc for alias in aliases):
            return f"Appliance{idx + 1}"

    # Second pass: substring match for multi-word or compound labels.
    for idx, label_lc in enumerate(labels_lc):
        if any(alias in label_lc for alias in aliases):
            return f"Appliance{idx + 1}"

    return None


def get_house_appliances(house_number: int) -> dict:
    """Return a dict mapping column name → device label for *house_number*.

    Example
    -------
    >>> get_house_appliances(3)
    {'Appliance1': 'Toaster', 'Appliance2': 'Washing Machine', ...}
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
        The house number.

    Raises
    ------
    ValueError
        If the number cannot be parsed.
    """
    import re
    import os

    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)", basename)
    if not match:
        raise ValueError(f"Cannot parse house number from filename: {basename!r}")
    return int(match.group(1))
