import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from tqdm import tqdm

def load_sessions(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    columns = ["timestamp", "user_id", "track_id", "event_type", "session_id"]
    return pd.DataFrame(data, columns=columns)

def load_jsonl_line_by_line(file_path):
    with open(file_path, "r") as file:
        for line in file:
            json_line = json.loads(line.strip())
            df = pd.DataFrame([json_line])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            yield df

def load_old_sessions_by_line():
    columns = ["timestamp", "user_id", "track_id", "event_type", "session_id"]
    with open("data/sessions.jsonl", "r") as file:
        for line in file:
            json_line = json.loads(line.strip())
            df = pd.DataFrame([json_line])
            df.columns = columns
            df["timestamp"] = df["timestamp"].apply(clean_date)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            yield df

def clean_date(date_str):
    return re.sub(r" \+.*$", "", date_str)

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def calculate_listening_time(old=False):
    if old:
        load_line = load_old_sessions_by_line()
    else:
        load_line = load_jsonl_line_by_line("./data2/sessions.jsonl")

    tracks = load_jsonl("./data2/tracks.jsonl")

    prev_id = None
    prev_timestamp = None
    prev_session = None
    prev_event = None
    prev_month = None
    monthly_listening = {}

    x = 2

    for df in tqdm(load_line): 
        user_id = df["user_id"].iloc[0]
        if user_id is None:
            user_id = prev_id
        timestamp = df["timestamp"].iloc[0]
        session = df["session_id"].iloc[0]
        event = df["event_type"].iloc[0]
        track = df["track_id"].iloc[0]
        month = timestamp.month
        year = timestamp.year

        if prev_id == user_id and prev_session == session:
            if event in ["Skip", "Play"] and prev_event == "Play":
                time_diff = (timestamp - prev_timestamp).total_seconds()
                monthly_listening[(year, month, int(user_id))] = monthly_listening.get((year, month, user_id), 0) + time_diff
        elif prev_id != user_id or prev_month != month or prev_session != session:
            if prev_event == "Play" and prev_id is not None:
                track_row = tracks.loc[tracks["id"] == prev_track]
                if not track_row.empty:
                    duration = track_row["duration_ms"].iloc[0] / 1000
                    monthly_listening[(prev_timestamp.year, prev_timestamp.month, int(prev_id))] = monthly_listening.get((prev_timestamp.year, prev_timestamp.month, prev_id), 0) + duration

        if prev_id != user_id:
            x -= 1
            print(user_id)
            if x == 0:
                break
        prev_id = user_id
        prev_timestamp = timestamp
        prev_session = session
        if event in ["Play", "Skip"]:
            prev_event = event
            prev_track = track
        prev_month = month

    return monthly_listening

def save_monthly_listening_to_file(data, filename):
    with open(filename, 'w') as f:
        for key, value in data.items():
            year, month, user_id = key 
            line = {
                "year": year,
                "month": month,
                "user_id": user_id,
                "listening_time_s": value
            }
            f.write(json.dumps(line) + "\n")

# monthly = calculate_listening_time()
# save_monthly_listening_to_file(monthly, "monthly_listening_v2.jsonl")
