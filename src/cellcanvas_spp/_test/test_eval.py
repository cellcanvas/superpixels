import pytest
import pandas as pd


from cellcanvas_spp.eval import evaluate_spp_features


@pytest.fixture
def mock_complete_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "area": [10, 1000, 5000],
            "intensity_mean": [-0.1, 5, -0.5],
            "ground_truth": [0, 7, 7],
        },
        index=pd.Index([5, 2024, 10], name="superpixel"),
    )


def test_eval_spp_features(
    mock_complete_dataframe: pd.DataFrame,
) -> None:

    result = evaluate_spp_features(mock_complete_dataframe)

    assert result.cohen_kappa == 1.0
    assert result.confusion_matrix.equals(
        pd.DataFrame(
            {
                0: [1, 0],
                7: [0, 2],
            },
            index=[0, 7],
        )
    )
