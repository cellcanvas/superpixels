from dataclasses import dataclass
import pandas as pd


@dataclass
class EvaluationResult:
    cohen_kappa: float
    confusion_matrix: pd.DataFrame


def evaluate_spp_features(
    df: pd.DataFrame,
) ->  EvaluationResult:
    pass