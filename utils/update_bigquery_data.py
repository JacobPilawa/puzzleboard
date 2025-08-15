import os
import time
import datetime
import pandas as pd
import joblib
from google.cloud import bigquery
from helpers import compute_speed_puzzle_rankings

# ---------------------------
# Load credentials from GitHub Actions secret
# ---------------------------
GCP_KEY_FILE = "gcp_key.json"
if "GCP_KEY" in os.environ:
    with open(GCP_KEY_FILE, "w") as f:
        f.write(os.environ["GCP_KEY"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY_FILE
else:
    raise RuntimeError("GCP_KEY environment variable not found. Set it in GitHub Actions secrets.")

# ---------------------------
# BigQuery client
# ---------------------------
client = bigquery.Client()

# List datasets and tables
datasets = list(client.list_datasets())
for dataset in datasets:
    print("Dataset ID:", dataset.dataset_id)

dataset_id = datasets[0].dataset_id  # assuming first dataset is correct
tables = list(client.list_tables(dataset_id))
for table in tables:
    print("Table ID:", table.table_id)

# Build queries
queries = [f"{dataset_id}.{table.table_id}" for table in tables]

# ---------------------------
# Fetch data from BigQuery
# ---------------------------
dfs = []
for sub in queries:
    time.sleep(1)  # avoid rate limiting
    query = f"SELECT * FROM {sub}"
    try:
        df = client.query(query).to_dataframe()
        dfs.append(df)
    except Exception as e:
        print(f"Error getting {sub}: {e}")
        dfs.append(None)

# Assign tables to variables
competition_entries = dfs[0]
competition_instances = dfs[3]
competitions = dfs[4]
competitors = dfs[5]
entry_competitors = dfs[6]
player_event_metrics = dfs[8]

# ---------------------------
# Processing
# ---------------------------
res = competition_entries.copy()
res = res.drop(columns=['competition_team_size', 'load_timestamp'])
res = res.merge(competitions, on='competition_id', how='left')
res = res.drop(columns=['last_updated_timestamp'])
res = res.merge(entry_competitors, on='entry_id', how='left')
res = res.drop(columns=['load_timestamp', 'role_in_entry'])
res = res.merge(competitors, on='competitor_id', how='left')
res = res.merge(player_event_metrics, on=['competitor_id', 'competition_id'], how='left')

# Calculate PPM
res['remaining_value'] = res['remaining_value'].fillna(0)
res['PPM'] = (res['pieces_value'].astype(float) - res['remaining_value'].astype(float)) / (res['completion_time_seconds'] / 60)

# Latest JPAR Out
date_strings = res['competition_date_x'].apply(str)
res['Date'] = pd.to_datetime(date_strings, errors='coerce')
res['Latest JPAR Out'] = res.sort_values('Date') \
    .groupby('competitor_id')['rating_out'] \
    .transform('last')

# Corrected time
res['time_penalty'] = (((res['pieces_value'].astype(float) - res['remaining_value'].astype(float)) /
                       res['completion_time_seconds'].astype(float)) ** (-1)) * res['remaining_value'].astype(float)
res['corrected_time'] = res['time_penalty'].astype(float) + res['completion_time_seconds'].astype(float)

# Total events
res['Total Events'] = res.groupby('competitor_id')['competitor_id'].transform('count')

# Format duration helper
def format_duration(seconds):
    if pd.isna(seconds):
        return None
    td = datetime.timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

res["12-Month Avg Completion Time"] = res['rolling_12mo_avg_completion_time'].apply(format_duration)

# Final dataframe
final_df = res[[
    "rank_in_competition_x",
    "display_name",
    "time_value",
    "remaining_value",
    "Date",
    "pieces_value",
    "source_tab_name",
    "competition_name",
    "completion_time_seconds",
    "rating_in",
    "rating_out",
    "event_strength_rating",
    "12-Month Avg Completion Time",
    "PPM",
    "Latest JPAR Out",
    "time_penalty",
    "corrected_time",
    "Total Events"
]].rename(columns={
    "rank_in_competition_x": "Rank",
    "display_name": "Name",
    "time_value": "Time",
    "remaining_value": "Remaining",
    "pieces_value": "Pieces",
    "source_tab_name": "Event",
    "competition_name": "Full_Event",
    "completion_time_seconds": "time_in_seconds",
    "rating_in": "PTR In",
    "rating_out": "PTR Out",
    "event_strength_rating": "Avg PTR In (Event)",
})

# ---------------------------
# Rankings & Save Files
# ---------------------------
styled_df, results = compute_speed_puzzle_rankings(
    final_df,
    min_puzzles=3,
    min_event_attempts=10,
    weighted=False
)

# # Save to data folder with timestamp
# today_str = datetime.datetime.utcnow().strftime("%y%m%d")
# scrape_file = f"data/{today_str}_scrape.pkl"
# rankings_file = f"data/{today_str}_standard_rankings_output.pkl"
#
# final_df.to_pickle(scrape_file)
# joblib.dump(results, rankings_file)

# save data to folder without timestamp
scrape_file = f"data/most_recent_scrape.pkl"
rankings_file = f"data/most_recent_standard_rankings_output.pkl"

final_df.to_pickle(scrape_file)
joblib.dump(results, rankings_file)

print(f"Saved scrape to {scrape_file}")
print(f"Saved rankings to {rankings_file}")
