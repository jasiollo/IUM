import requests
import matplotlib.pyplot as plt

BASE_URL = "http://127.0.0.1:8080"

def train_ab_experiment(base_file_path, advanced_file_path):
    try:
        response = requests.post(
            f"{BASE_URL}/ab_experiment/train_and_compare",
            json={"base_file_path": base_file_path, "advanced_file_path": advanced_file_path},
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.json().get('error', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        print("Error: Unable to connect to the server. Please ensure the server is running.")
        return None

def plot_mae_comparison(base_mae, advanced_mae):
    models = ['Base Model', 'Advanced Model']
    mae_values = [base_mae, advanced_mae]
    plt.figure(figsize=(6, 4))
    plt.bar(models, mae_values, color=['blue', 'green'])
    plt.title('MAE Comparison')
    plt.ylabel('Mean Absolute Error')
    plt.show()

def plot_predictions(actual, predicted, title):
    actual, predicted = actual[:100], predicted[:100]
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label=f'{title} Actual', alpha=0.7)
    plt.plot(predicted, label=f'{title} Predicted', linestyle="dashed", alpha=0.7)
    plt.legend()
    plt.title(f'{title} Predictions')
    plt.show()

def plot_feature_importances(base_importances, advanced_importances):
    plt.figure(figsize=(12, 6))
    base_features, base_values = zip(*base_importances.items())
    adv_features, adv_values = zip(*advanced_importances.items())

    plt.bar(base_features, base_values, alpha=0.7, label='Base Model')
    plt.bar(adv_features, adv_values, alpha=0.7, label='Advanced Model')
    plt.xticks(rotation=90)
    plt.title('Feature Importances Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base_dataset_path = "input/monthly_listening_v3.jsonl"
    advanced_dataset_path = "input/merged_llp.jsonl"

    print("Running A/B Experiment...")
    results = train_ab_experiment(base_dataset_path, advanced_dataset_path)

    if results:
        mae_comparison = results['mae_comparison']
        predicted_values_comparison = results['predicted_values_comparison']
        feature_importances_comparison = results['feature_importances_comparison']

        plot_mae_comparison(
            mae_comparison['base_model_mae'], 
            mae_comparison['advanced_model_mae']
        )
        plot_predictions(
            predicted_values_comparison['base_model']['actual'],
            predicted_values_comparison['base_model']['predicted'],
            'Base Model'
        )
        plot_predictions(
            predicted_values_comparison['advanced_model']['actual'],
            predicted_values_comparison['advanced_model']['predicted'],
            'Advanced Model'
        )
        plot_feature_importances(
            feature_importances_comparison['base_model'],
            feature_importances_comparison['advanced_model']
        )
