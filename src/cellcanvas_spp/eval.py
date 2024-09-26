"""
This module implements the confusion matrix and Cohen's kappa score from the predicted values of the ground truth
through cross-validation:

Cross-Validation: Technique to split the data into "folds" and train the model on part of the data while testing it
on the rest
Cohen's kappa score: A statistic that measures the agreement between predicted and true classes, adjusted for chance.
Confusion matrix: Shows the performance of the classification model by displaying the true and predicted classes in a
matrix form.
"""

from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import pandas as pd


@dataclass
class EvaluationResult:
    cohen_kappa: float
    confusion_matrix: pd.DataFrame
    y_pred: pd.DataFrame


def evaluate_spp_features(df: pd.DataFrame,**kwargs) -> EvaluationResult:
    """
    Evaluates superpixel features by calculating Cohen's kappa score and the
    confusion matrix based on ground truth and predicted labels.
    """
    # estimator = SVC(**kwargs)
    estimator = RandomForestClassifier(**kwargs)
    x = df.drop(columns="ground_truth")  # feature set
    y_true = df["ground_truth"]  # labels

    # creating predicted labels based on SVC. Automatically defaults to 5 folds.
    y_pred = cross_val_predict(estimator, x, y_true)

    kappa_score = cohen_kappa_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # print("Kappa:", kappa_score)
    # print("Confusion:", conf_matrix)

    return EvaluationResult(kappa_score, pd.DataFrame(conf_matrix), pd.DataFrame(y_pred,columns=['y_pred']))
