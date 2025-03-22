import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from merge_data import load_jsonl


class AdvancedModel:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.model = None
        self.feature_names = []
        self.mae = None
        self.predicted_values = None
        self.actual_values = None

    def set_file_path(self, new_file_path):
        self.file_path = new_file_path

    def load_data(self):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        return load_jsonl(self.file_path)

    def prepare_data_for_model(self, data):
        rows = []
        for record in data:
            base = {
                "user_id": record["user_id"],
                "year": record["year"],
                "month": record["month"],
                "listening_time_s": record["listening_time_s"]
            }
            for i, prev in enumerate(record["prev_3_months_data"]):
                base[f"prev_{i+1}_listening_time_s"] = prev["listening_time_s"]
                base[f"prev_{i+1}_likes"] = prev["likes"]
                base[f"prev_{i+1}_buy_premium"] = int(prev["buy_premium"])
                base[f"prev_{i+1}_skips"] = prev["skips"]
                base[f"prev_{i+1}_avg_popularity"] = prev["avg_popularity"]
                base[f"prev_{i+1}_unique_artists"] = prev["unique_artists"]
                base[f"prev_{i+1}_unique_tracks"] = prev["unique_tracks"]
            rows.append(base)
        return pd.DataFrame(rows)

    def train(self):
        data = self.load_data()
        df = self.prepare_data_for_model(data)
        
        features = [col for col in df.columns if col != "listening_time_s"]
        X = df[features]
        y = df["listening_time_s"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        self.mae = mean_absolute_error(y_test, y_pred)
        self.predicted_values = y_pred
        self.actual_values = y_test
        
        self.feature_names = features

    def get_mae(self):
        if self.mae is None:
            raise ValueError("Model is not trained yet.")
        return self.mae

    def get_predicted_values(self):
        if self.predicted_values is None or self.actual_values is None:
            raise ValueError("Model is not trained yet.")
        return self.actual_values.tolist(), self.predicted_values.tolist()

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
