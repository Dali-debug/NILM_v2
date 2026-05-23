"""
pipeline
--------
NILM processing pipeline: preprocessing, HMM training, disaggregation,
and real-time inference.

Public API
----------
    from pipeline.preprocessing import preprocess_house, load_refit_csv
    from pipeline.train_hmm    import run_training, load_models
    from pipeline.disaggregate import run_disaggregation, disaggregate_nilm
    from pipeline.realtime     import RealTimeNILM
"""

from .preprocessing import preprocess_house, load_refit_csv, hampel_filter, interpolate_missing
from .train_hmm     import run_training, load_models, save_models, reconstruct_hmm, train_appliance_hmm
from .disaggregate  import run_disaggregation, disaggregate_submetering, disaggregate_nilm
from .realtime      import RealTimeNILM

__all__ = [
    # preprocessing
    "preprocess_house",
    "load_refit_csv",
    "hampel_filter",
    "interpolate_missing",
    # training
    "run_training",
    "load_models",
    "save_models",
    "reconstruct_hmm",
    "train_appliance_hmm",
    # disaggregation
    "run_disaggregation",
    "disaggregate_submetering",
    "disaggregate_nilm",
    # real-time
    "RealTimeNILM",
]
