import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify
from load_data import load_jsonl


class BaseModel:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.mae = None
        self.predicted_values = None
        self.actual_values = None

    def set_file_path(self, new_file_path):
        self.file_path = new_file_path

    def load_data(self):
        return load_jsonl(self.file_path)

    def train(self):
        data = self.load_data()
        print(data)
        X = data[['user_id', 'year', 'month']]
        y = data['listening_time_s']

        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        self.predicted_values = self.model.predict(X_test_scaled)
        self.actual_values = y_test
        self.mae = mean_absolute_error(y_test, self.predicted_values)
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
