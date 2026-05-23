"""
Projet_NILM
-----------
REFIT NILM pipeline package.

Public API
----------
    from Projet_NILM.pipeline.preprocessing import preprocess_house
    from Projet_NILM.pipeline.train_hmm     import run_training
    from Projet_NILM.pipeline.disaggregate  import run_disaggregation
    from Projet_NILM.data.refit_metadata    import get_appliance_column, parse_house_number
    from Projet_NILM.config                 import DEFAULT_APPLIANCES, POWER_ON_THRESHOLD
"""

from .config import DEFAULT_APPLIANCES, DEFAULT_N_STATES, POWER_ON_THRESHOLD
from .pipeline.preprocessing import preprocess_house, load_refit_csv, hampel_filter
from .pipeline.train_hmm import run_training, load_models, save_models, reconstruct_hmm
from .pipeline.disaggregate import run_disaggregation
from .data.refit_metadata import (
    get_appliance_column,
    get_house_appliances,
    parse_house_number,
    HOUSE_APPLIANCES,
    APPLIANCE_ALIASES,
)

__all__ = [
    # config
    "DEFAULT_APPLIANCES",
    "DEFAULT_N_STATES",
    "POWER_ON_THRESHOLD",
    # preprocessing
    "preprocess_house",
    "load_refit_csv",
    "hampel_filter",
    # training
    "run_training",
    "load_models",
    "save_models",
    "reconstruct_hmm",
    # disaggregation
    "run_disaggregation",
    # metadata
    "get_appliance_column",
    "get_house_appliances",
    "parse_house_number",
    "HOUSE_APPLIANCES",
    "APPLIANCE_ALIASES",
]
