import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"

@pytest.fixture
def sample_file_paths():
    return {
        "base_file_path": "input/monthly_listening_v3.jsonl",
        "advanced_file_path": "input/merged_llp.jsonl"
    }

def test_ab_experiment_train_and_compare(sample_file_paths):
    response = requests.post(
        f"{BASE_URL}/ab_experiment/train_and_compare",
        json=sample_file_paths
    )
    assert response.status_code == 200
    data = response.json()
    assert "mae_comparison" in data
    assert "predicted_values_comparison" in data
    assert "feature_importances_comparison" in data

    assert "base_model_mae" in data["mae_comparison"]
    assert "advanced_model_mae" in data["mae_comparison"]
    assert isinstance(data["mae_comparison"]["base_model_mae"], float)
    assert isinstance(data["mae_comparison"]["advanced_model_mae"], float)

    assert "base_model" in data["predicted_values_comparison"]
    assert "advanced_model" in data["predicted_values_comparison"]
    assert len(data["predicted_values_comparison"]["base_model"]["actual"]) == len(data["predicted_values_comparison"]["base_model"]["predicted"])

    assert "base_model" in data["feature_importances_comparison"]
    assert "advanced_model" in data["feature_importances_comparison"]
    assert isinstance(data["feature_importances_comparison"]["base_model"], dict)
    assert isinstance(data["feature_importances_comparison"]["advanced_model"], dict)
