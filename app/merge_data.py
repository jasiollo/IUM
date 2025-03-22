import json
from collections import defaultdict
from datetime import datetime, timedelta


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_previous_months(year, month, n=3):
    date = datetime(year, month, 1)
    previous_months = []
    for _ in range(n):
        date -= timedelta(days=27)
        previous_months.append((date.year, date.month))
    return previous_months


def add_historical_data(listening_data, likes_premium_data, track_statistics):
    likes_premium_index = defaultdict(lambda: {'likes': 0, 'buy_premium': False, 'skips': 0})
    for record in likes_premium_data:
        key = (record['user_id'], int(record['year']), int(record['month']))
        likes_premium_index[key] = {
            'likes': record['likes'],
            'buy_premium': record['buy_premium'],
            'skips': record['skips']
        }

    listening_index = defaultdict(lambda: 0)
    for record in listening_data:
        key = (record['user_id'], record['year'], record['month'])
        listening_index[key] = record['listening_time_s']

    track_stats_index = {(stat['user_id'], stat['year'], stat['month']): stat for stat in track_statistics}

    enriched_data = []
    for record in listening_data:
        user_id = record['user_id']
        year = record['year']
        month = record['month']

        prev_months = get_previous_months(year, month)
        prev_data = []

        for prev_year, prev_month in prev_months:
            key = (user_id, prev_year, prev_month)
            prev_data.append({
                'year': prev_year,
                'month': prev_month,
                'listening_time_s': listening_index[key],
                'likes': likes_premium_index[key]['likes'],
                'buy_premium': likes_premium_index[key]['buy_premium'],
                'skips': likes_premium_index[key]['skips'],
                'avg_popularity': track_stats_index.get(key, {}).get('avg_popularity', 0),
                'unique_artists': track_stats_index.get(key, {}).get('unique_artists', 0),
                'unique_tracks': track_stats_index.get(key, {}).get('unique_tracks', 0)
            })

        enriched_record = {
            **record,
            'prev_3_months_data': prev_data
        }
        enriched_data.append(enriched_record)

    return enriched_data


def calculate_track_statistics(sessions_path, tracks):
    track_info = {track["id"]: track for track in tracks}

    user_monthly_stats = defaultdict(lambda: defaultdict(lambda: {
        "total_popularity": 0,
        "track_count": 0,
        "artist_set": set(),
        "track_set": set()
    }))    

    with open(sessions_path, 'r') as f:
        line_num = 0
        for line in f:
            line_num += 1
            if line_num % 100000 == 0:
                print(line_num)
            session = json.loads(line.strip())
            user_id = session["user_id"]
            timestamp = datetime.fromisoformat(session["timestamp"])
            year, month = timestamp.year, timestamp.month
            track_id = session["track_id"]

            if track_id in track_info:
                track_data = track_info[track_id]
                user_month = user_monthly_stats[user_id][(year, month)]
                user_month["total_popularity"] += track_data['popularity']
                user_month["track_count"] += 1
                user_month["artist_set"].add(track_data['artist_id'])
                user_month["track_set"].add(track_id)

    stats = []
    for user_id, months in user_monthly_stats.items():
        for (year, month), data in months.items():
            stats.append({
                'user_id': user_id,
                'year': year,
                'month': month,
                'avg_popularity': data['total_popularity'] / data['track_count'] if data['track_count'] > 0 else 0,
                'unique_artists': len(data['artist_set']),
                'unique_tracks': len(data['track_set'])
            })

    return stats


def save_to_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    listening_file = "./monthly_listening_v3.jsonl"
    likes_premium_file = "./likes_premium.jsonl"
    output_file = "merged_llp.jsonl"
    sessions_path = "./data2/sessions.jsonl"
    tracks_path = "./data2/tracks.jsonl"

    listening_data = load_jsonl(listening_file)
    likes_premium_data = load_jsonl(likes_premium_file)
    tracks_data = load_jsonl(tracks_path)

    tracks_statistics = calculate_track_statistics(sessions_path, tracks_data)

    merged_data = add_historical_data(listening_data, likes_premium_data, tracks_statistics)

    save_to_jsonl(merged_data, output_file)
