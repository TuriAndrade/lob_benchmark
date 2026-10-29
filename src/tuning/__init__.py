from .linear_model import main as linear_model_main
from .lightgbm_model import main as lightgbm_model_main
from .tcn_model import main as tcn_model_main

models = {
    "linear": linear_model_main,
    "lightgbm": lightgbm_model_main,
    "tcn": tcn_model_main,
}
