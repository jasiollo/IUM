import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gluonts.torch.model.deepar import DeepAREstimator
# from gluonts.torch import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions


def load_jsonl_data(file_path):
    users_data = []
    with open(file_path, 'r') as file:
        for line in file:
            users_data.append(json.loads(line))
    return users_data


def prepare_dataset(data, target_key="listening_time_s"):
    time_series = []
    for record in data:
        user_id = record["user_id"]
        static_features = [user_id]
        series = []

        for month_data in record["prev_3_months_data"]:
            series.append({
                "timestamp": f"{month_data['year']}-{month_data['month']:02d}",
                "target": month_data[target_key],
                "dynamic_features": [
                    month_data["likes"],
                    month_data["skips"],
                    month_data["avg_popularity"],
                    month_data["unique_artists"],
                    month_data["unique_tracks"]
                ],
                "static_features": static_features
            })

        series.append({
            "timestamp": f"{record['year']}-{record['month']:02d}",
            "target": record[target_key],
            "dynamic_features": [
                0,
                0,
                0,
                0,
                0
            ],
            "static_features": static_features
        })
        time_series.append(series)
    return time_series


def create_gluonts_dataset(time_series, start_index, end_index):
    dataset = []
    for series in time_series:
        timestamps = [entry["timestamp"] for entry in series]
        targets = [entry["target"] for entry in series]
        dynamic_features = np.array([entry["dynamic_features"] for entry in series]).T
        static_features = series[0]["static_features"]

        dataset.append({
            "start": timestamps[0],
            "target": targets,
            "feat_dynamic_real": dynamic_features.tolist(),
            "feat_static_cat": static_features
        })
    return ListDataset(dataset, freq="M")


def train_deepar(dataset, prediction_length=1, epochs=10):
    estimator = DeepAREstimator(
        freq="M",
        prediction_length=prediction_length,
    )
    predictor = estimator.train(dataset)
    return predictor


def evaluate_deepar(predictor, test_dataset):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_dataset,
        predictor=predictor,
        num_samples=100
    )

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it)
    return agg_metrics, item_metrics


if __name__ == "__main__":
    test_size = 0.2
    prediction_length = 1
    epochs = 10

    data = load_jsonl_data("./merged_llp.jsonl")
    time_series = prepare_dataset(data)
    num_users = len(time_series)
    train_index = int((1 - test_size) * num_users)
    train_dataset = create_gluonts_dataset(time_series, 0, train_index)
    test_dataset = create_gluonts_dataset(time_series, train_index, num_users)
    predictor = train_deepar(train_dataset, prediction_length, epochs)

    agg_metrics, item_metrics = evaluate_deepar(predictor, test_dataset)

    for key, value in agg_metrics.items():
        print(f"{key}: {value}")
