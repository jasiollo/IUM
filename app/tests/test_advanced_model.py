import pytest
import requests

BASE_URL = "http://127.0.0.1:8080"
@pytest.fixture
def sample_file_path():
    
    return "/home/yannosh/ium/ium/app/input/merged_llp.jsonl" 

def test_train_base_model(sample_file_path):
    response = requests.post(
        f"{BASE_URL}/advanced_model/train",
        json={"file_path": sample_file_path}
    )
    assert response.status_code == 200
    assert "Advanced model trained successfully" in response.json().get("message", "")

def test_get_mae():
    response = requests.get(f"{BASE_URL}/advanced_model/mae")
    assert response.status_code == 200
    assert "mae" in response.json()
    assert isinstance(response.json()["mae"], float)

def test_get_predicted_values():
    response = requests.get(f"{BASE_URL}/advanced_model/predict")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2  
    assert isinstance(data[0], list) and isinstance(data[1], list)

def test_get_feature_importances():
    response = requests.get(f"{BASE_URL}/advanced_model/importance_features")
    assert response.status_code == 200
    features = response.json()
    assert isinstance(features, dict)
    assert all(isinstance(k, str) and isinstance(v, float) for k, v in features.items())
