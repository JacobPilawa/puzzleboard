import datetime
import plotly.graph_objects as go
from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# ✅ MUST BE FIRST
st.set_page_config(page_title="Speed Puzzling Dashboard", page_icon="🧩", layout="wide",initial_sidebar_state="expanded")

# ---------- Data Loading & Cleaning ----------
@st.cache_data
def scrape_data():
    '''
    - this function will scrape the data from rob's spreadsheets
    - takes ahwile to run (~10s) but not updated enough to do every time
    - so currently just storing scrapes in ./data/ 
    '''
    url = 'https://docs.google.com/spreadsheets/d/1aCENVOk-wroyW4-YS4OgtTTqr3rA-T9CZOjCQgALgSE/export?format=xlsx'
    xls = pd.ExcelFile(url)

    exclude = {"_Competition Factors", "_JPAR Ratings", "_Latest JPAR Ratings", "_Histograms", "_UTILITY FUNCTIONS"}
    sheets_to_read = [s for s in xls.sheet_names if s not in exclude]

    dfs = {
        name: xls.parse(name, dtype={'Time': str})
        for name in sheets_to_read
    }

    for name, df in dfs.items():
        df['Event'] = name

    combined_df = pd.concat(dfs.values(), ignore_index=True)
    combined_df = combined_df.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore')

    def time_to_seconds(time_str):
        try:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        except Exception:
            return np.nan

    combined_df['time_in_seconds'] = combined_df['Time'].apply(time_to_seconds)

    def clean_remaining(val):
        if pd.isna(val):
            return 0
        if isinstance(val, str) and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", val):
            return 0
        return val

    combined_df["Remaining"] = combined_df["Remaining"].apply(clean_remaining)

    combined_df['time_penalty'] = (((500 - combined_df['Remaining']) / combined_df['time_in_seconds'])**(-1)) * combined_df['Remaining']
    combined_df['corrected_time'] = combined_df['time_penalty'] + combined_df['time_in_seconds']
    
    ## DATA CLEANING
    combined_df['Name'] = combined_df['Name'].str.strip()
    combined_df['Pieces'] = pd.to_numeric(combined_df['Pieces'],errors='coerce')

    return combined_df
    
def load_data():
    '''
    read in the data from the pickle file
    '''
    data = pd.read_pickle('./data/250519_scape.pkl')
    return data

df = load_data()


# ---------- Sidebar Navigation ----------
st.sidebar.title("📍Navigation")
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("🧩 Home "):
    st.session_state.page = "Home"
if st.sidebar.button("🏆 Leaderboards "):
    st.session_state.page = "Leaderboards"
    st.session_state['selected_event'] = ""
if st.sidebar.button("👤 Puzzler Profiles "):
    st.session_state.page = "Puzzler Profiles"
    
if 'selected_event' not in st.session_state:
    st.session_state['selected_event'] = ""
if 'selected_puzzler' not in st.session_state:
    st.session_state['selected_puzzler'] = ""
if 'trigger_jump' not in st.session_state:
    st.session_state['trigger_jump'] = False
    
st.sidebar.markdown("### 🔍 Quick Search")

# Event quick jump
event_names_sidebar = sorted(df['Event'].unique())
selected_event_sidebar = st.sidebar.selectbox(
    "Jump to Leaderboard",
    [""] + event_names_sidebar,
    index=(event_names_sidebar.index(st.session_state['selected_event']) + 1) if st.session_state['selected_event'] in event_names_sidebar else 0,
    key="sidebar_event"
)
if selected_event_sidebar and selected_event_sidebar != st.session_state['selected_event']:
    st.session_state['selected_event'] = selected_event_sidebar
    st.session_state['page'] = "Leaderboards"
    st.session_state['trigger_jump'] = True

# Puzzler quick jump
puzzler_names_sidebar = sorted(df['Name'].dropna().unique())
selected_puzzler_sidebar = st.sidebar.selectbox(
    "Jump to Puzzler Profile",
    [""] + puzzler_names_sidebar,
    index=(puzzler_names_sidebar.index(st.session_state['selected_puzzler']) + 1) if st.session_state['selected_puzzler'] in puzzler_names_sidebar else 0,
    key="sidebar_puzzler"
)
if selected_puzzler_sidebar and selected_puzzler_sidebar != st.session_state['selected_puzzler']:
    st.session_state['selected_puzzler'] = selected_puzzler_sidebar
    st.session_state['page'] = "Puzzler Profiles"
    st.session_state['trigger_jump'] = True
    
# Deferred rerun to handle navigation smoothly
if st.session_state.get("trigger_jump"):
    st.session_state["trigger_jump"] = False
    st.rerun()
    
page = st.session_state.page

# ---------- Plotting Utilities ----------
@st.cache_data
def get_cumulative_stats(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')

    # Unique Puzzlers
    first_appearance = df.drop_duplicates(subset='Name', keep='first')
    cumulative_names = first_appearance.groupby('Date').size().cumsum().reset_index()
    cumulative_names.columns = ['Date', 'Cumulative Unique Names']

    # Total Events
    events = df.drop_duplicates(subset='Event', keep='first')
    cumulative_events = events.groupby('Date').size().cumsum().reset_index()
    cumulative_events.columns = ['Date', 'Cumulative Events']

    # Total Solves
    entry_counts = df.groupby('Date').size().sort_index()
    cumulative_entries = entry_counts.cumsum().reset_index()
    cumulative_entries.columns = ['Date', 'Cumulative Times Logged']

    return cumulative_names, cumulative_events, cumulative_entries

cumulative_names, cumulative_events, cumulative_entries = get_cumulative_stats(df)

@st.cache_data
def get_most_frequent_puzzlers(df):
    
    # get the 50 most frequent entrants to events
    frequent_puzzlers = df['Name'].value_counts().head(50)
    
    return frequent_puzzlers
    
frequent_puzzlers = get_most_frequent_puzzlers(df)

# ---------- Leaderboard Display Function ----------
def get_delta_color(percentile):
    if 40 <= percentile <= 60:
        return "off"      # gray
    elif percentile < 40:
        return "off"  # red
    else:
        return "off"   # green

def display_leaderboard(filtered_df: pd.DataFrame, df: pd.DataFrame, selected_event: str):
    st.title(f"Event: {selected_event}")
    filtered_df = filtered_df.copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date']).dt.date

    total_entrants = len(filtered_df['Name'].unique())
    fastest_time = filtered_df['time_in_seconds'].min()
    avg_time = filtered_df['time_in_seconds'].mean()

    # Calculate percentiles relative to full dataset df grouped by event:
    entrants_per_event = df.groupby('Event')['Name'].nunique()
    total_entrants_percentile = entrants_per_event.rank(pct=True).loc[selected_event] * 100

    # For fastest time: percentile of fastest_time relative to all times in full df
    fastest_time_percentile = (df['time_in_seconds'] < fastest_time).mean() * 100

    avg_time_per_event = df.groupby('Event')['time_in_seconds'].mean()
    avg_time_percentile = avg_time_per_event.rank(pct=True).loc[selected_event] * 100

    col1, col2, col3 = st.columns(3)
    # css to get rid of the arrows in the metrics
    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with col1:
        st.metric(
            "Total Entrants",
            total_entrants,
            delta=f"{total_entrants_percentile:.1f}%ile",
            delta_color=get_delta_color(total_entrants_percentile),
            border=True,
        )
    with col2:
        st.metric(
            "Fastest Time",
            str(timedelta(seconds=int(fastest_time))),
            delta=f"{fastest_time_percentile:.1f}%ile",
            delta_color=get_delta_color(fastest_time_percentile),
            border=True,
        )
    with col3:
        st.metric(
            "Average Time",
            str(timedelta(seconds=int(avg_time))),
            delta=f"{avg_time_percentile:.1f}%ile",
            delta_color=get_delta_color(avg_time_percentile),
            border=True,
        )


    # Plotting Functions on This Page
    def update_axis_layout(fig):
        """Apply common axis styling."""
        fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
        return fig
        
    def create_completion_time_plot(filtered_df: pd.DataFrame):
        RED_COLOR = '#b11f00'
        START_DATE = pd.to_datetime("2022-03-01")
        BASE_HEIGHT = 500
        BASE_LAYOUT = dict(template="plotly_white", font=dict(color="black", size=12), height=BASE_HEIGHT)
        
        # Prepare the data
        all_df = filtered_df.copy()
        all_df['time_in_hours'] = all_df['time_in_seconds'] / 3600
        all_df.sort_values('time_in_seconds', inplace=True)
        n_all = len(all_df)
        all_df['performance'] = np.linspace(1, 0, n_all) if n_all > 1 else 1.0
    
        # Hover template
        hover_template = (
            '<b>Solver:</b> %{customdata[0]}<br>'
            '<b>Rank:</b> %{customdata[4]:.0f}<br>'
            '<b>Time:</b> %{customdata[1]}<br>'
            '<b>Date:</b> %{customdata[2]|%Y-%m-%d}<br>'
            '<b>PPM:</b> %{customdata[3]:.1f}<br>'
            '<b>Performance:</b> %{x:.0%}'
        )
    
        # Create bar plot
        fig = go.Figure()
    
        fig.add_trace(go.Bar(
            x=all_df['performance'],
            y=all_df['time_in_hours'],
            marker=dict(color='#ff7f0e'),
            name='',
            customdata=all_df[['Name', 'Time', 'Date', 'PPM','Rank']].values,
            hovertemplate=hover_template
        ))
    
        # Format x-axis as percent scale
        tick_vals = [i/10 for i in range(0, 11)]
        tick_text = [f"{int(val*100)}%" for val in tick_vals]
    
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, autorange='reversed'),
            yaxis_title='Completion Time (hours)',
            title='Completion Times (All Attempts)',
            **BASE_LAYOUT
        )
    
        update_axis_layout(fig)
        return fig


    def create_kde_plot(filtered_df, df):
        return px.density_contour(filtered_df, x='time_in_seconds', title='KDE Plot')

    def create_attempt_boxplot(filtered_df):
        return px.box(filtered_df, y='attempt_number', title='Attempt Number Boxplot')

    st.plotly_chart(create_completion_time_plot(filtered_df), use_container_width=True)

    st.subheader("Leaderboard")
    
    view_df = filtered_df.copy()

    display_df = view_df.sort_values('time_in_seconds')[['Name', 'Time', 'Date', 'PPM','Pieces']].reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df, use_container_width=True)

    st.subheader("Additional Analytics")
    
# ---------- Puzzler Profile Display Function ----------
def display_puzzler_profile(df: pd.DataFrame, selected_puzzler: str):
    if not selected_puzzler:
        return

    puzzler_df = df[df['Name'] == selected_puzzler].copy()
    puzzler_df['Date'] = pd.to_datetime(puzzler_df['Date'], errors='coerce')
    puzzler_df = puzzler_df.dropna(subset=['Date'])

    st.markdown('---')
    st.header(f"{selected_puzzler}")
    st.subheader("📅 Event History")

    first_event_row = puzzler_df.loc[puzzler_df['Date'].idxmin()]
    latest_event_row = puzzler_df.loc[puzzler_df['Date'].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**First Event:** {first_event_row['Date'].date()} — {first_event_row['Event']}")
        if st.button("Go to First Event Leaderboard", key=f"first_event_{selected_puzzler}"):
            st.session_state['page'] = "Leaderboards"
            st.session_state['selected_event'] = first_event_row['Event']
            st.rerun()  # immediately rerun app to update the page variable

    with col2:
        st.markdown(f"**Most Recent Event:** {latest_event_row['Date'].date()} — {latest_event_row['Event']}")
        if st.button("Go to Most Recent Event Leaderboard", key=f"latest_event_{selected_puzzler}"):
            st.session_state['page'] = "Leaderboards"
            st.session_state['selected_event'] = latest_event_row['Event']
            st.rerun()  # immediately rerun app to update the page variable

    # Calculate metrics for selected puzzler
    total_events = puzzler_df['Event'].nunique()
    total_pieces = puzzler_df['Pieces'].sum()
    fastest_time_seconds = puzzler_df['time_in_seconds'].min()
    average_time_seconds = puzzler_df['time_in_seconds'].mean()

    # Calculate percentiles for each metric relative to full df grouped by 'Name'
    # 1) Total events percentile
    events_per_puzzler = df.groupby('Name')['Event'].nunique()
    total_events_percentile = events_per_puzzler.rank(pct=True).loc[selected_puzzler] * 100

    # 2) Total pieces percentile
    pieces_per_puzzler = df.groupby('Name')['Pieces'].sum()
    total_pieces_percentile = pieces_per_puzzler.rank(pct=True).loc[selected_puzzler] * 100

    # 3) Fastest time percentile (lower time = better, so rank inverted)
    fastest_times_per_puzzler = df.groupby('Name')['time_in_seconds'].min()
    fastest_time_percentile = (fastest_times_per_puzzler.rank(pct=True, ascending=True).loc[selected_puzzler]) * 100

    # 4) Average time percentile (lower time = better, so rank inverted)
    avg_times_per_puzzler = df.groupby('Name')['time_in_seconds'].mean()
    avg_time_percentile = (avg_times_per_puzzler.rank(pct=True, ascending=True).loc[selected_puzzler]) * 100

    st.subheader("📊 Statistics")

    col1, col2, col3, col4 = st.columns(4)
    st.write(
        """
        <style>
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with col1:
        st.metric(
            "Total Events",
            total_events,
            delta=f"{total_events_percentile:.1f}%ile",
            delta_color=get_delta_color(total_events_percentile),
            border=True
        )
    with col2:
        st.metric(
            "Total Pieces",
            int(total_pieces),
            delta=f"{total_pieces_percentile:.1f}%ile",
            delta_color=get_delta_color(total_pieces_percentile),
            border=True
        )
    with col3:
        st.metric(
            "Fastest Time",
            str(timedelta(seconds=int(fastest_time_seconds))),
            delta=f"{fastest_time_percentile:.1f}%ile",
            delta_color=get_delta_color(fastest_time_percentile),
            border=True
        )
    with col4:
        st.metric(
            "Average Time",
            str(timedelta(seconds=int(average_time_seconds))),
            delta=f"{avg_time_percentile:.1f}%ile",
            delta_color=get_delta_color(avg_time_percentile),
            border=True
        )

    st.subheader("📄 All Events")
    display_df = puzzler_df.sort_values('Date')[['Date', 'Event', 'Rank', 'Time', 'PPM', 'Pieces', 'Remaining']].copy()
    display_df['Date'] = display_df['Date'].dt.date

    # Calculate total entrants per event
    event_totals = df.groupby('Event')['Name'].count()

    # Create Rank column as "N/T"
    display_df['Rank'] = display_df.apply(lambda row: f"{int(row['Rank'])}/{event_totals.get(row['Event'], 0)}", axis=1)

    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)


# ---------- Home ----------
if page == "Home":
    st.markdown(
        "<h1 style='text-align: center;'>🧩 Speed Puzzling Competition Dashboard </h1>",
        unsafe_allow_html=True
    )
    st.markdown("Explore puzzler and event stats. Access the event leaderboard and puzzler profiles using the sidebar on the left.")

    # commenting out some buttons that are behaving poorly and migth be confusing
    # col_leaderboard, col_player= st.columns(2)
    #
    # with col_leaderboard:
    #     if st.button("🏆 Leaderboards 🏆"):
    #         st.session_state.page = "Leaderboards"
    #         st.rerun()
    #
    # with col_player:
    #     if st.button("👤 Puzzler Profiles 👤"):
    #         st.session_state.page = "Puzzler Profiles"
    #         st.rerun()
    

    # --- Metrics block ---
    df['Date'] = pd.to_datetime(df['Date'])
    latest_date = df['Date'].max()
    cutoff_date = latest_date - pd.DateOffset(months=1)
    df_prior = df[df['Date'] <= cutoff_date]

    total_puzzles = df['Event'].nunique()
    total_users = df['Name'].nunique()
    total_logs = len(df)
    total_time_seconds = int(df['time_in_seconds'].sum())
    total_time_str = str(timedelta(seconds=total_time_seconds))
    total_pieces_value = df['Pieces'].sum()
    total_pieces = "{:0,}".format(int(total_pieces_value))

    total_puzzles_str = f"{total_puzzles:,}"
    total_users_str = f"{total_users:,}"
    total_logs_str = f"{total_logs:,}"

    prior_total_puzzles = df_prior['Event'].nunique()
    prior_total_users = df_prior['Name'].nunique()
    prior_total_logs = len(df_prior)
    prior_total_time_seconds = int(df_prior['time_in_seconds'].sum())
    prior_total_pieces_value = df_prior['Pieces'].sum()

    delta_users = total_users - prior_total_users
    delta_logs = total_logs - prior_total_logs
    delta_puzzles = total_puzzles - prior_total_puzzles
    delta_time_seconds = total_time_seconds - prior_total_time_seconds
    delta_time_str = str(timedelta(seconds=delta_time_seconds))
    delta_pieces_value = total_pieces_value - prior_total_pieces_value
    delta_pieces = "{:,}".format(delta_pieces_value)

    delta_users_str = f"{delta_users:,} / month"
    delta_logs_str = f"{delta_logs:,} / month"
    delta_puzzles_str = f"{delta_puzzles:,} / month"
    delta_time_str = f"{delta_time_str} / month"
    delta_pieces_str = f"{delta_pieces} / month"

    col1, col2, col3 = st.columns(3)
    with col1:
        #st.metric("Total Users", total_users_str, delta=delta_users_str, delta_color="normal", border=True)
        st.metric("Total Unique Puzzlers", total_users_str, border=True)
    with col2:
        #st.metric("Total Logs", total_logs_str, delta=delta_logs_str, delta_color="normal", border=True)
        st.metric("Total Times", total_logs_str, border=True)
    with col3:
        #st.metric("Total Puzzles", total_puzzles_str, delta=delta_puzzles_str, delta_color="normal", border=True)
        st.metric("Total Events", total_puzzles_str, border=True)

    col1, col2 = st.columns(2)
    with col1:
        #st.metric("Total Time Spent Puzzling", total_time_str, delta=delta_time_str, delta_color="normal", border=True)
        st.metric("Total Time Spent Puzzling", total_time_str, border=True)
    with col2:
        #st.metric("Total Pieces", total_pieces, delta=delta_pieces_str, delta_color="normal", border=True)
        st.metric("Total Pieces", total_pieces, border=True)

    # --- Your 3 plots arranged 1x3 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.line(
            cumulative_names,
            x="Date", y="Cumulative Unique Names",
            title="Cumulative Number of Unique Puzzlers",
            labels={"Cumulative Unique Names": "Unique Puzzlers"},
            color_discrete_sequence=["darkred"]  
        )
        fig1.update_traces(line=dict(width=5))  # Make line thicker
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(
            cumulative_events,
            x="Date", y="Cumulative Events",
            title="Cumulative Number of Events",
            labels={"Cumulative Events": "Events"},
            color_discrete_sequence=["darkred"] 
        )
        fig2.update_traces(line=dict(width=5))
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        fig3 = px.line(
            cumulative_entries,
            x="Date", y="Cumulative Times Logged",
            title="Cumulative Number of Puzzle Solves",
            labels={"Cumulative Times Logged": "Solves"},  # Fixed label key
            color_discrete_sequence=["darkred"]  
        )
        fig3.update_traces(line=dict(width=5))
        st.plotly_chart(fig3, use_container_width=True)

        
    # --- Most Entrants ---
    fig4 = px.bar(
        frequent_puzzlers,
        x=frequent_puzzlers.index,
        y=frequent_puzzlers.values,
        color=frequent_puzzlers.values,
        labels={
            "y": "Events Entered",
            "index": "Puzzler",
            "color": "Events Entered"
        },
        title="Top 50 Puzzlers by Events Entered",
        color_continuous_scale='Tealgrn' 
    )
    fig4.update_layout(xaxis_tickangle=-75)
    st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown('---')
    st.markdown("Data curated by [Rob Shields of the Piece Talks podcast](https://podcasts.apple.com/us/podcast/piece-talks/id1742455250). Website and visualizations put together by [Jacob Pilawa](https://jacobpilawa.github.io/).")

        
# ---------- Leaderboards Page ----------
if page == "Leaderboards":
    st.title("🏆 Leaderboards ")
    st.markdown("Welcome to the Leaderboards! Select an event from the dropdown below to view its stats.")

    event_names = sorted(df['Event'].unique())
    # Get default event from session state or empty string if none
    default_event = st.session_state.get('selected_event', "")

    # Guard in case default_event is not in event_names (avoid ValueError)
    if default_event in event_names:
        default_index = event_names.index(default_event) + 1  # +1 because we add "" option at front
    else:
        default_index = 0

    selected_event = st.selectbox(
        "Choose an event to view its leaderboard:",
        [""] + event_names,
        index=default_index,
    )

    # Update session_state with user's manual change to the dropdown (optional)
    st.session_state['selected_event'] = selected_event

    st.markdown('---')
    if selected_event:
        event_df = df[df['Event'] == selected_event]
        display_leaderboard(event_df, df, selected_event)
    st.markdown('---')
    st.markdown("Data curated by [Rob Shields of the Piece Talks podcast](https://podcasts.apple.com/us/podcast/piece-talks/id1742455250). Website and visualizations put together by [Jacob Pilawa](https://jacobpilawa.github.io/).")

        
        
# ---------- Puzzler Profiles Page ----------
if page == "Puzzler Profiles":
    st.title("👤 Puzzler Profiles ")

    puzzler_list = sorted(df['Name'].dropna().unique())
    default_puzzler = st.session_state.get('selected_puzzler', "")
    selected_puzzler = st.selectbox("Select a puzzler", [""] + puzzler_list, index=(puzzler_list.index(default_puzzler) + 1) if default_puzzler in puzzler_list else 0)
    st.session_state['selected_puzzler'] = selected_puzzler
    
    if selected_puzzler:
        display_puzzler_profile(df, selected_puzzler)
    st.markdown('---')
    st.markdown("Data curated by [Rob Shields of the Piece Talks podcast](https://podcasts.apple.com/us/podcast/piece-talks/id1742455250). Website and visualizations put together by [Jacob Pilawa](https://jacobpilawa.github.io/).")
