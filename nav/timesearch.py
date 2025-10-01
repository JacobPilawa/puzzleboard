import streamlit as st
from utils.helpers import get_ranking_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_ranking_table, load_data, get_bottom_string, get_standard_ranking_table

#### GET DATA
df = load_data()
styled_table, results = get_standard_ranking_table()
bottom_string = get_bottom_string()

# ---------- Helper function to convert seconds to HH:MM:SS ----------
def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# ---------- Helper function to parse time strings to seconds ----------
def time_to_seconds(time_str):
    """Convert time string (HH:MM:SS or MM:SS) to seconds"""
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            return 0
    except:
        return 0

# ---------- Timesearch Display Function ----------
def display_timesearch_page(styled_table, results, df):
    
    st.title("⏳️ Time Search")
    
    st.markdown("""
    Filter puzzlers by their 12-Month Average Completion time to get a sense of who puzzles at similar paces!
    This could help find people to team with or compare yourself against.
    """)
    
    # Ensure Date is datetime for sorting
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Sort by person (Name) and Date (most recent first)
    df_sorted = df.sort_values(["Name", "Date"], ascending=[True, False])
    
    # Keep the most recent row per person
    most_recent = df_sorted.groupby("Name", as_index=False).first()
    
    # Rename columns
    most_recent = most_recent.rename(
        columns={
            "Date": "Most Recent Event Date",
            "Full_Event": "Most Recent Event",
        }
    )
    
    # Format the date column to show only YYYY-MM-DD
    most_recent["Most Recent Event Date"] = most_recent["Most Recent Event Date"].dt.date
    
    # Convert 12-Month Avg Completion Time to seconds for filtering
    most_recent["Time_Seconds"] = most_recent["12-Month Avg Completion Time"].apply(time_to_seconds)
    
    # Get min time in seconds
    min_time_seconds = int(most_recent["Time_Seconds"].min())
    
    # Set maximum to 3 hours (10800 seconds)
    max_time_seconds = 10800  # 3 hours
    
    # Create the slider
    st.markdown("### Filter by 12-Month Average Completion Time")
    
    # Single range slider
    selected_range = st.slider(
        "Completion Time Range",
        min_value=min_time_seconds,
        max_value=max_time_seconds,
        value=(min_time_seconds, max_time_seconds),
        step=60,  # 1 minute increments
        format=""
    )
    
    selected_min_seconds, selected_max_seconds = selected_range
    
    # Display the selected time range in HH:MM:SS format
    st.markdown(f"**Selected Time Range:** {seconds_to_hms(selected_min_seconds)} - {seconds_to_hms(selected_max_seconds)}")
    
    # Filter the dataframe based on the slider values
    filtered_df = most_recent[
        (most_recent["Time_Seconds"] >= selected_min_seconds) & 
        (most_recent["Time_Seconds"] <= selected_max_seconds)
    ]
    
    # Sort by Time_Seconds (fastest first)
    filtered_df = filtered_df.sort_values("Time_Seconds", ascending=True)
    
    # Reorder and select final columns
    subset_df = filtered_df[
        [
            "Name",
            "12-Month Avg Completion Time",
            "Total Events",
            "Most Recent Event",
            "Most Recent Event Date",
        ]
    ]
    
    # Display count
    st.markdown(f"**Showing {len(subset_df)} of {len(most_recent)} puzzlers**")
    
    # Display dataframe
    st.dataframe(subset_df, use_container_width=True, height=600)

display_timesearch_page(styled_table, results, df)

st.markdown('---')
st.markdown(bottom_string)