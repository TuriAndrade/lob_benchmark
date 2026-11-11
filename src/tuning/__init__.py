from .linear_model_tuning import main as linear_model_tuning_main
from .lightgbm_model_tuning import main as lightgbm_model_tuning_main
from .tcn_model_tuning import main as tcn_model_tuning_main

models = {
    "linear_model": linear_model_tuning_main,
    "lightgbm_model": lightgbm_model_tuning_main,
    "tcn_model": tcn_model_tuning_main,
}
