"""
This module implements the confusion matrix and Cohen's kappa score.

Cohen's kappa score: A statistic that measures the agreement between predicted and true classes, adjusted for chance.
Confusion matrix: Shows the performance of the classification model by displaying the true and predicted classes in a
matrix form.
"""

from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import pandas as pd


@dataclass
class EvaluationResult:
    cohen_kappa: float
    confusion_matrix: pd.DataFrame


def evaluate_spp_features(df: pd.DataFrame, ground_truth_label: str, predicted_label: str) -> EvaluationResult:
    """
    Evaluates superpixel features by calculating Cohen's kappa score and the
    confusion matrix based on ground truth and predicted labels.

    Example?
        df = pd.DataFrame({
        ...     'ground_truth': [1, 0, 1, 1, 0],
        ...     'predicted': [1, 0, 1, 0, 0]
        ... })
        result = evaluate_spp_features(df, 'ground_truth', 'predicted')
    """
    y_true = df[ground_truth_label]
    y_pred = df[predicted_label]

    kappa_score = cohen_kappa_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return EvaluationResult(kappa_score, pd.DataFrame(conf_matrix))
