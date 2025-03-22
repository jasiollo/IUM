import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from merge_data import load_jsonl
import matplotlib.pyplot as plt


def prepare_data_for_model(data):
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


data = load_jsonl("./merged_llp.jsonl")
df = prepare_data_for_model(data)

features = [col for col in df.columns if col not in ["listening_time_s"]]

# "prev_1_listening_time_s", "prev_2_listening_time_s", "prev_3_listening_time_s",
# "prev_1_buy_premium", "prev_2_buy_premium", "prev_3_buy_premium", 
# "prev_1_likes", "prev_2_likes", "prev_3_likes",  "prev_1_skips", "prev_2_skips", "prev_3_skips", 
# , "prev_1_avg_popularity", "prev_2_avg_popularity", "prev_3_avg_popularity", "prev_1_unique_artists", "prev_2_unique_artists", "prev_3_unique_artists", "prev_1_unique_tracks", "prev_2_unique_tracks", "prev_3_unique_tracks"

X = df[features]
y = df["listening_time_s"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(y_test))

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(mae)

plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Rzeczywiste wartości')
plt.plot(y_pred, label='Przewidywane wartości', linestyle='dashed')
plt.legend()
plt.title('Porównanie rzeczywistych i przewidywanych wartości')
plt.show()

feature_importance = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print(importance_df)
