class ABExperiment:
    def __init__(self, base_model, advanced_model):
        self.base_model = base_model
        self.advanced_model = advanced_model

    def train_models(self, base_file_path, advanced_file_path):
        self.base_model.set_file_path(base_file_path)
        self.advanced_model.set_file_path(advanced_file_path)

        self.base_model.train()
        self.advanced_model.train()

    def compare_models(self):
        base_mae = self.base_model.get_mae()
        advanced_mae = self.advanced_model.get_mae()

        base_actual, base_predicted = self.base_model.get_predicted_values()
        advanced_actual, advanced_predicted = self.advanced_model.get_predicted_values()

        base_importances = self.base_model.get_feature_importances()
        advanced_importances = self.advanced_model.get_feature_importances()

        return {
            "mae_comparison": {
                "base_model_mae": base_mae,
                "advanced_model_mae": advanced_mae
            },
            "predicted_values_comparison": {
                "base_model": {
                    "actual": base_actual,
                    "predicted": base_predicted
                },
                "advanced_model": {
                    "actual": advanced_actual,
                    "predicted": advanced_predicted
                }
            },
            "feature_importances_comparison": {
                "base_model": base_importances,
                "advanced_model": advanced_importances
            }
        }
