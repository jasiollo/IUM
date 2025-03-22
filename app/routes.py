from flask import request, jsonify
from models.base_model import BaseModel
from models.advanced_model import AdvancedModel
from utils.ab_tests import ABExperiment

base_model = BaseModel()
advanced_model = AdvancedModel()
ab_experiment = ABExperiment(base_model, advanced_model)

def configure_routes(app):

    # BASE MODEL ROUTES
    @app.route('/base_model/train', methods=['POST'])
    def train_base_model():
        file_path = request.get_json(force=True).get("file_path")
        base_model.set_file_path(file_path)
        try:
            base_model.train()
            return jsonify({"message": "Base model trained successfully."})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/base_model/mae', methods=['GET'])
    def get_base_model_mae():
        try:
            mae = base_model.get_mae()
            return jsonify({"mae": mae})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/base_model/predict', methods=['GET'])
    def predict_base_model():
        try:
            values = base_model.get_predicted_values()
            return jsonify(values)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/base_model/importance_features', methods=['GET'])
    def get_base_model_importance_features():
        try:
            features = base_model.get_feature_importances()
            return jsonify(features)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # ADVANCED MODEL ROUTES
    @app.route('/advanced_model/train', methods=['POST'])
    def train_advanced_model():
        file_path = request.get_json(force=True).get("file_path")
        advanced_model.set_file_path(file_path)
        try:
            advanced_model.train()
            return jsonify({"message": "Advanced model trained successfully."})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/advanced_model/mae', methods=['GET'])
    def get_advanced_model_mae():
        try:
            mae = advanced_model.get_mae()
            return jsonify({"mae": mae})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/advanced_model/predict', methods=['GET'])
    def predict_advanced_model():
        try:
            values = advanced_model.get_predicted_values()
            return jsonify(values)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/advanced_model/importance_features', methods=['GET'])
    def get_advanced_model_importance_features():
        try:
            features = advanced_model.get_feature_importances()
            return jsonify(features)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/ab_experiment/train_and_compare', methods=['POST'])
    def ab_experiment_train_and_compare():
        try:
            data = request.json
            base_file_path = data.get("base_file_path")
            advanced_file_path = data.get("advanced_file_path")

            if not base_file_path or not advanced_file_path:
                return jsonify({"error": "File paths for both models are required."}), 400

            ab_experiment.train_models(base_file_path, advanced_file_path)
            comparison_results = ab_experiment.compare_models()

            return jsonify(comparison_results), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
