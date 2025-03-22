from app.load_data import load_jsonl


data = load_jsonl("./app/input/monthly_listening_v3.jsonl")
print(data.loc[:, "listening_time_s"].mean())
