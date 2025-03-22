import json
from collections import defaultdict
from datetime import datetime

def process_sessions(file_path):
    user_data = defaultdict(lambda: defaultdict(lambda: {'likes': 0, 'buy_premium': False, "skips": 0}))
    line_num = 0

    with open(file_path, 'r') as f:
        for line in f:
            line_num += 1
            if line_num % 10000 == 0:
                print(line_num)
            record = json.loads(line.strip())
            user_id = record['user_id']
            event_type = record['event_type']
            timestamp = record['timestamp']
            year = datetime.fromisoformat(timestamp).year
            month = datetime.fromisoformat(timestamp).month

            key = f"{year}-{month:02d}"

            if event_type == "Like":
                user_data[user_id][key]['likes'] += 1
            elif event_type == "BuyPremium":
                user_data[user_id][key]['buy_premium'] = True
            elif event_type == "Skip":
                user_data[user_id][key]["skips"] += 1

    return user_data


def save_to_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for user_id, months in data.items():
            for key, stats in months.items():
                year, month = key.split('-')
                output_record = {
                    "user_id": user_id,
                    "year": year,
                    "month": month,
                    "likes": stats["likes"],
                    "buy_premium": stats["buy_premium"],
                    "skips": stats["skips"]
                }
                f.write(json.dumps(output_record) + '\n')


input_path = "./data2/sessions.jsonl"
output_path = "likes_premium.jsonl"
data = process_sessions(input_path)
save_to_jsonl(data, output_path)
