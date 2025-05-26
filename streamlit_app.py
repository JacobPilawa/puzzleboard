import string
from helpers import get_ranking_table, load_data, load_jpar_data
from scipy.stats import gaussian_kde
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

bottom_string = "Data curated by [Rob Shields of the Piece Talks podcast](https://podcasts.apple.com/us/podcast/piece-talks/id1742455250). Website and visualizations put together by [Jacob Pilawa](https://jacobpilawa.github.io/). Feel free to reach out if you spot any bugs or inconsistencies. For logging your own times, check out [myspeedpuzzling](https://myspeedpuzzling.com/en/home)!"

# ---------- Data Loading & Cleaning ----------

df = load_data()
jpar_df = load_jpar_data()
styled_table, results = get_ranking_table(min_puzzles=3, min_event_attempts=10, weighted=False)

# ---------- Sidebar Navigation ----------
st.sidebar.title("📍Navigation")
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("🧩 Home "):
    st.session_state.page = "Home"
    st.session_state['selected_event'] = ""
    st.session_state['selected_puzzler'] = ""
if st.sidebar.button("🏆 Competitions "):
    st.session_state.page = "Competitions"
    st.session_state['selected_event'] = ""
if st.sidebar.button("👤 Puzzler Profiles "):
    st.session_state.page = "Puzzler Profiles"
if st.sidebar.button("📊 Puzzler Ratings "):
    st.session_state.page = "JPAR"
    
if 'selected_event' not in st.session_state:
    st.session_state['selected_event'] = ""
if 'selected_puzzler' not in st.session_state:
    st.session_state['selected_puzzler'] = ""
if 'trigger_jump' not in st.session_state:
    st.session_state['trigger_jump'] = False
    
st.sidebar.markdown("### 🔍 Quick Search")

# Event quick jump
event_names_sidebar = sorted(df['Full_Event'].unique())
selected_event_sidebar = st.sidebar.selectbox(
    "Jump to Competition",
    [""] + event_names_sidebar,
    index=(event_names_sidebar.index(st.session_state['selected_event']) + 1) if st.session_state['selected_event'] in event_names_sidebar else 0,
    key="sidebar_event"
)
if selected_event_sidebar and selected_event_sidebar != st.session_state['selected_event']:
    st.session_state['selected_event'] = selected_event_sidebar
    st.session_state['page'] = "Competitions"
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
    events = df.drop_duplicates(subset='Full_Event', keep='first')
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
    if 80 <= percentile:
        return "normal"      # gray
    elif percentile < 40:
        return "off"  # red
    else:
        return "off"   # green

def display_leaderboard(filtered_df: pd.DataFrame, df: pd.DataFrame, selected_event: str):
    
    st.title(f"Competition: {selected_event}")
    filtered_df = filtered_df.copy()
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date']).dt.date

    total_entrants = len(filtered_df['Name'].unique())
    fastest_time = filtered_df['time_in_seconds'].min()
    avg_time = filtered_df['time_in_seconds'].mean()

    # Calculate percentiles relative to full dataset df grouped by event:
    entrants_per_event = df.groupby('Full_Event')['Name'].nunique()
    total_entrants_percentile = entrants_per_event.rank(pct=True).loc[selected_event] * 100

    fastest_time_percentile = (df['time_in_seconds'] < fastest_time).mean() * 100

    avg_time_per_event = df.groupby('Full_Event')['time_in_seconds'].mean()
    avg_time_percentile = avg_time_per_event.rank(pct=True).loc[selected_event] * 100

    col1, col2, col3 = st.columns(3)

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

    def get_delta_color(pct):
        if pct >= 75:
            return "off"
        elif pct >= 50:
            return "off"
        else:
            return "off"

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

    def update_axis_layout(fig):
        fig.update_xaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(tickfont=dict(color="black"), title_font=dict(color="black"))
        return fig

    def create_completion_time_plot(filtered_df: pd.DataFrame):
        RED_COLOR = '#b11f00'
        START_DATE = pd.to_datetime("2022-03-01")
        BASE_HEIGHT = 500
        BASE_LAYOUT = dict(template="plotly_white", font=dict(color="black", size=12), height=BASE_HEIGHT)

        all_df = filtered_df.copy()
        all_df['time_in_hours'] = all_df['time_in_seconds'] / 3600
        all_df.sort_values('time_in_seconds', inplace=True)
        n_all = len(all_df)
        all_df['performance'] = np.linspace(1, 0, n_all) if n_all > 1 else 1.0

        hover_template = (
            '<b>Solver:</b> %{customdata[0]}<br>'
            '<b>Rank:</b> %{customdata[4]:.0f}<br>'
            '<b>Time:</b> %{customdata[1]}<br>'
            '<b>Date:</b> %{customdata[2]|%Y-%m-%d}<br>'
            '<b>PPM:</b> %{customdata[3]:.1f}<br>'
            '<b>Performance:</b> %{x:.0%}'
        )

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=all_df['performance'],
            y=all_df['time_in_hours'],
            marker=dict(color='tomato'),
            name='',
            customdata=all_df[['Name', 'Time', 'Date', 'PPM','Rank']].values,
            hovertemplate=hover_template
        ))

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

    def create_normalized_kde_plot(filtered_df: pd.DataFrame, df: pd.DataFrame):
        # Convert times to hours
        all_times = df['time_in_seconds'] / 3600
        filtered_times = filtered_df['time_in_seconds'] / 3600
    
        # Drop NaNs/infs and limit to <= 5 hours
        all_times = all_times.replace([np.inf, -np.inf], np.nan).dropna()
        filtered_times = filtered_times.replace([np.inf, -np.inf], np.nan).dropna()
    
        all_times = all_times[all_times <= 5]
        filtered_times = filtered_times[filtered_times <= 5]
    
        if len(all_times) < 2 or len(filtered_times) < 2:
            fig = go.Figure()
            fig.update_layout(
                title='KDE of Completion Times (Not enough data)',
                template='plotly_white',
                height=500
            )
            return fig
    
        # Compute KDEs
        kde_all = gaussian_kde(all_times)
        kde_filtered = gaussian_kde(filtered_times)
    
        x_vals = np.linspace(0, 5, 500)
    
        y_all = kde_all(x_vals)
        y_filtered = kde_filtered(x_vals)
    
        # Normalize y_filtered so its peak matches y_all's peak
        max_all = np.max(y_all)
        max_filtered = np.max(y_filtered)
        if max_filtered > 0:
            y_filtered = y_filtered * (max_all / max_filtered)
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_all,
            mode='lines',
            name='All Events',
            fill='tozeroy',
            line=dict(color='black', width=3)
        ))
    
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_filtered,
            mode='lines',
            name=filtered_df['Full_Event'].iloc[0],
            fill='tozeroy',
            line=dict(color='#1f77b4', width=3)
        ))
    
        fig.update_layout(
            xaxis_title='Completion Time (hours)',
            yaxis_title='Relative Number of Times',
            xaxis=dict(range=[0, 5]),
            yaxis=dict(range=[0, max_all * 1.05]),  # Add a small buffer above max
            template='plotly_white',
            height=500,
            legend=dict(x=0.9, y=0.99, bgcolor='rgba(255,255,255,0.8)', bordercolor='black', borderwidth=1)
        )
    
        return fig

    

    def create_attempt_boxplot(filtered_df):
        return px.box(filtered_df, y='attempt_number', title='Attempt Number Boxplot')

    # Main Plot
    st.plotly_chart(create_completion_time_plot(filtered_df), use_container_width=True)

    st.subheader("Leaderboard")

    view_df = filtered_df.copy()
    display_df = view_df.sort_values('time_in_seconds')[['Name', 'Time', 'Date', 'PPM','Pieces']].reset_index(drop=True)
    display_df.index = display_df.index + 1
    st.dataframe(display_df, use_container_width=True)

    # Additional Analytics Section
    st.subheader("Additional Statistics")
    st.plotly_chart(create_normalized_kde_plot(filtered_df, df), use_container_width=True)

    
# ---------- Puzzler Profile Display Function ----------
def display_puzzler_profile(df: pd.DataFrame, selected_puzzler: str):
    
    if not selected_puzzler:
        return

    puzzler_df = df[df['Name'] == selected_puzzler].copy()
    puzzler_df['Date'] = pd.to_datetime(puzzler_df['Date'], errors='coerce')
    puzzler_df = puzzler_df.dropna(subset=['Date'])

    st.markdown('---')
    st.header(f"{selected_puzzler}")

    # --------- Statistics -----------
    st.subheader("📊 Statistics")

    # Count medals
    num_gold = (puzzler_df['Rank'] == 1).sum()
    num_silver = (puzzler_df['Rank'] == 2).sum()
    num_bronze = (puzzler_df['Rank'] == 3).sum()

    medals_line = "🥇" * num_gold + "🥈" * num_silver + "🥉" * num_bronze
    if medals_line:
        st.markdown(medals_line)
        
    # --------- RANKINGS -----------
    if results['Name'].eq(selected_puzzler).any():
        selected_ranking = results[results['Name'] == selected_puzzler]
        col1, col2, col3, col4, col5 = st.columns(5)
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
                "Overall Ranking",
                selected_ranking.index+1,
                delta=f'/{len(results)} eligible',
                delta_color='off',
                border=True
            )
        with col2:
            st.metric(
                "PT Rank",
                selected_ranking['PT Rank'],
                delta=' ',
                delta_color='off',
                border=True
            )
        with col3:
            st.metric(
                "Z Rank",
                selected_ranking['Z Rank'],
                delta=' ',
                delta_color='off',
                border=True
            )
        with col4:
            st.metric(
                "Percentile Rank",
                selected_ranking['Percentile Rank'],
                delta=' ',
                delta_color='off',
                border=True
            )
        with col5:
            st.metric(
                "Average Rank Score",
                np.round(selected_ranking['Average Rank'],2),
                delta=' ',
                delta_color='off',
                border=True
            )
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
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
                "Overall Ranking",
                'N/A',
                border=True
            )
        with col2:
            st.metric(
                "PT Rank",
                'N/A',
                border=True
            )
        with col3:
            st.metric(
                "Z Rank",
                'N/A',
                border=True
            )
        with col4:
            st.metric(
                "Percentile Rank",
                'N/A',
                border=True
            )
        with col5:
            st.metric(
                "Average Rank Score",
                'N/A',
                border=True
            )

        
    # Calculate metrics for selected puzzler
    total_events = puzzler_df['Full_Event'].nunique()
    total_pieces = puzzler_df['Pieces'].sum()
    fastest_time_seconds = puzzler_df['time_in_seconds'].min()
    average_time_seconds = puzzler_df['time_in_seconds'].mean()

    # Calculate percentiles for each metric relative to full df grouped by 'Name'
    # 1) Total events percentile
    events_per_puzzler = df.groupby('Name')['Full_Event'].nunique()
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
            delta=f"{100-fastest_time_percentile:.1f}%ile",
            delta_color=get_delta_color(100-fastest_time_percentile),
            border=True
        )
    with col4:
        st.metric(
            "Average Time",
            str(timedelta(seconds=int(average_time_seconds))),
            delta=f"{100-avg_time_percentile:.1f}%ile",
            delta_color=get_delta_color(100-avg_time_percentile),
            border=True
        )


    st.subheader("⏱️ Solve Times for Most Recent 1.5 Years")
    
    # Prepare plotting data
    time_plot_df = puzzler_df.dropna(subset=['time_in_seconds', 'Date', 'Full_Event']).copy()
    time_plot_df = time_plot_df.sort_values('Date').reset_index(drop=True)

    # Filter to most recent 1.5 years
    most_recent_date = time_plot_df['Date'].max()
    cutoff_date = most_recent_date - timedelta(days=365 * 1.5)
    time_plot_df = time_plot_df[time_plot_df['Date'] >= cutoff_date].reset_index(drop=True)

    # Compute hours for y-axis
    time_plot_df['time_in_hours'] = time_plot_df['time_in_seconds'] / 3600

    # Compute total entrants per event
    event_totals = df.groupby('Full_Event')['Name'].count()
    time_plot_df['Total_Entrants'] = time_plot_df['Full_Event'].map(event_totals)

    # Label for plotting
    time_plot_df['EventLabel'] = time_plot_df['Date'].dt.strftime('%b %d, %Y')
    
    # ——————— 12-Month Moving Average ———————
    ma_df = time_plot_df.set_index('Date').sort_index()
    ma_df['MA_12M'] = ma_df['time_in_hours'].rolling(window='365D', min_periods=1).mean()
    time_plot_df['MA_12M'] = ma_df['MA_12M'].values
    
    # Toggle for date spacing
    date_spacing = st.checkbox("Use true date spacing on x-axis", value=False)
    
    if not time_plot_df.empty:
        def hours_to_hhmmss(hours):
            total_seconds = int(hours * 3600)
            h = total_seconds // 3600
            m = (total_seconds % 3600) // 60
            s = total_seconds % 60
            return f"{h:02d}:{m:02d}:{s:02d}"
    
        time_plot_df['time_hhmmss'] = time_plot_df['time_in_hours'].apply(hours_to_hhmmss)
        time_plot_df['MA_12M_hhmmss'] = time_plot_df['MA_12M'].apply(hours_to_hhmmss)
    
        def add_suffixes(df):
            new_labels = []
            current_label = None
            count = 0
            suffixes = list(string.ascii_lowercase)
    
            for label in df['EventLabel']:
                if label != current_label:
                    current_label = label
                    count = 0
                else:
                    count += 1
    
                if count == 0:
                    new_labels.append(label)
                else:
                    new_labels.append(f"{label} ({suffixes[count-1]})")
            return new_labels
    
        time_plot_df['EventLabelUnique'] = add_suffixes(time_plot_df)
    
        jittered_df = time_plot_df.copy()
    
        if date_spacing:
            used_dates = set()
            new_dates = []
    
            for _, row in jittered_df.iterrows():
                base_date = row['Date']
                offset = 0
                new_date = base_date
                while new_date in used_dates:
                    offset += 1
                    new_date = base_date + pd.Timedelta(days=offset)
                used_dates.add(new_date)
                new_dates.append(new_date)
    
            jittered_df['JitteredDate'] = new_dates
            x_col = 'JitteredDate'
        else:
            x_col = 'EventLabelUnique'
    
        fig = px.bar(
            jittered_df,
            x=x_col,
            y='time_in_hours',
            color='EventLabelUnique' if date_spacing else None,
            hover_data={
                'Full_Event': True,
                'time_hhmmss': True,
                'Rank': True,
                'Total_Entrants': True,
                'Pieces': True,
                'EventLabel': False,
                'time_in_hours': False,
            },
            labels={'time_in_hours': 'Time (hours)', 'EventLabelUnique': 'Event Date'},
            title="Solve Times",
        )
    
        fig.add_trace(go.Scatter(
            x=jittered_df[x_col],
            y=jittered_df['MA_12M'],
            mode='lines+markers',
            name='12-Month Mov. Avg.',
            line=dict(color='red', width=2, dash='solid'),
            marker=dict(color='red', size=0),
            customdata=jittered_df['MA_12M_hhmmss'],
            hovertemplate='12-Month Avg: %{customdata}<extra></extra>',
        ))
    
        if date_spacing:
            fig.update_layout(
                xaxis_title='Date',
                xaxis=dict(type='date'),
                showlegend=False,
            )
        else:
            fig.update_layout(
                showlegend=False,
                xaxis=dict(
                    type='category',
                    categoryorder='array',
                    categoryarray=time_plot_df['EventLabelUnique'],
                    tickangle=-70,
                    tickmode='array',
                    tickvals=time_plot_df['EventLabelUnique'],
                    ticktext=time_plot_df['EventLabelUnique'],
                ),
                xaxis_title='Event Date',
            )
    
        fig.update_layout(
            bargap=0.05,
            hoverlabel=dict(bgcolor="white"),
            legend=dict(
                x=0.98,
                y=0.98,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1
            ),
            yaxis_title='Time (hours)'
        )
    
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No solve time data available to plot for the selected filter.")



    st.subheader("📅 Event History")

    first_event_row = puzzler_df.loc[puzzler_df['Date'].idxmin()]
    latest_event_row = puzzler_df.loc[puzzler_df['Date'].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**First Event:** {first_event_row['Date'].date()} — {first_event_row['Full_Event']}")
        if st.button("Go to First Event Leaderboard", key=f"first_event_{selected_puzzler}"):
            st.session_state['page'] = "Competitions"
            st.session_state['selected_event'] = first_event_row['Full_Event']
            st.rerun()  # immediately rerun app to update the page variable

    with col2:
        st.markdown(f"**Most Recent Event:** {latest_event_row['Date'].date()} — {latest_event_row['Full_Event']}")
        if st.button("Go to Most Recent Event Leaderboard", key=f"latest_event_{selected_puzzler}"):
            st.session_state['page'] = "Competitions"
            st.session_state['selected_event'] = latest_event_row['Full_Event']
            st.rerun()  # immediately rerun app to update the page variable

    st.subheader("📄 All Events")
    display_df = puzzler_df.sort_values('Date')[['Date', 'Full_Event', 'Rank', 'Time', 'PPM', 'Pieces', 'Remaining','Avg PTR In (Event)', 'PTR Out','12-Month Avg Completion Time']].copy()
    display_df['Date'] = display_df['Date'].dt.date

    # Calculate total entrants per event
    event_totals = df.groupby('Full_Event')['Name'].count()

    # Create Rank column as "N/T"
    display_df['Rank'] = display_df.apply(lambda row: f"{int(row['Rank'])}/{event_totals.get(row['Full_Event'], 0)}", axis=1)
    
    # Sort by most recent
    display_df = display_df.sort_values(by='Date',ascending=False)

    # Display dataframe
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

# ---------- JPAR Ratings Display Function ----------
def display_jpar_ratings(styled_table, results):
    
    st.title("📊 Puzzler Ratings")
    st.markdown(f"""
    This page shows the 
    current ratings for puzzlers with {styled_table.data['Eligible Puzzles'].min()} or more times within the last year,
    and considering only events with more than 10 participants. Three ranking systems are showng in the table.
    PT Rank is a ranking system developed by [Rob Shields](https://podcasts.apple.com/us/podcast/piece-talks/id1742455250), 
    Z Rank is a rank based on the [number of standard deviations above average](https://en.wikipedia.org/wiki/Standard_score) for each competition, and Percentile Rank 
    is a rank based on average [percentiles](https://en.wikipedia.org/wiki/Percentile) in each competition.""")
    
    
    # Example number input
    # Create 3 columns, only use the first for input (approx 1/3 width)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        number = st.number_input("""Minimum number of eligible puzzles""", value=3, step=1, format="%d")
        
    # Check if number is entered
    if number is not None:
        # Do something if a number is entered
        filitered_styled_table, filtered_results = get_ranking_table(min_puzzles=number, min_event_attempts=10, weighted=False)
        st.dataframe(filitered_styled_table, use_container_width=True)
        # Add your conditional logic here
    else:
        # Default behavior
        st.dataframe(styled_table, use_container_width=True)

    # Generate distribution plot
    st.subheader("📈 Rating Comparison")
    # ------- Pair plot of the rankings
    # Drop non-numeric or irrelevant columns
    if number is not None:
        a = pd.DataFrame.copy(filtered_results)
    else:
        a = pd.DataFrame.copy(results)
        
    df = a.drop(columns=['Name','Eligible Puzzles'])

    # Select columns to include in the pairplot
    columns = df.columns

    # Filtered DataFrame
    filtered_df = df[columns]

    # Create the scatter matrix
    fig = px.scatter_matrix(
        filtered_df,
        dimensions=columns,
        color_discrete_sequence=['black'],  # black dots
        height=900,
        width=900
    )
    fig.update_traces(diagonal_visible=False,showupperhalf=False)

    # Add 1-to-1 dashed red lines
    for i, x_col in enumerate(columns):
        for j, y_col in enumerate(columns):
            if i < j:
                x = filtered_df[[x_col, y_col]].dropna()
                min_val = max(x[x_col].min(), x[y_col].min())
                max_val = min(x[x_col].max(), x[y_col].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(dash='dash', color='red', width=2),
                    showlegend=False,
                    hoverinfo='skip',
                    xaxis=f'x{i+1}',
                    yaxis=f'y{j+1}'
                ))

    # Update layout: disable diagonal axes, restrict axes to positive values
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        dragmode='select',
        hovermode='closest'
    )

    # Set axes to start at 0
    for i, col in enumerate(columns):
        fig.update_xaxes(range=[0, None], row=i+1, col=1)
        fig.update_yaxes(range=[0, None], row=1, col=i+1)
    

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    


# ---------- Home ----------
if page == "Home":
    
    st.markdown(
        "<h1 style='text-align: center;'>🧩 Speed Puzzling Competitions Dashboard </h1>",
        unsafe_allow_html=True
    )
    st.markdown("This dashboard contains only the **official results** from a number of **500 piece** speed puzzling competitions (speedpuzzling.com, EJJ, various local and national events, and more). Access competition results, puzzler profiles, and rankings using the sidebar on the left.")

    # commenting out some buttons that are behaving poorly and migth be confusing
    # col_leaderboard, col_player= st.columns(2)
    #
    # with col_leaderboard:
    #     if st.button("🏆 Competitions 🏆"):
    #         st.session_state.page = "Competitions"
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

    total_puzzles = df['Full_Event'].nunique()
    total_users = df['Name'].nunique()
    total_logs = len(df)
    total_time_seconds = int(df['time_in_seconds'].sum())
    total_time_str = str(timedelta(seconds=total_time_seconds))
    total_pieces_value = df['Pieces'].sum()
    total_pieces = "{:0,}".format(int(total_pieces_value))

    total_puzzles_str = f"{total_puzzles:,}"
    total_users_str = f"{total_users:,}"
    total_logs_str = f"{total_logs:,}"

    prior_total_puzzles = df_prior['Full_Event'].nunique()
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
        st.metric("Total Times Logged", total_logs_str, border=True)
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
            title="Cumulative Number of Times Logged",
            labels={"Cumulative Times Logged": "Times Logged"},  # Fixed label key
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
    st.markdown(bottom_string)

        
# ---------- Competitions Page ----------
if page == "Competitions":
    st.title("🏆 Competitions ")

    event_names = sorted(df['Full_Event'].unique())
    # Get default event from session state or empty string if none
    default_event = st.session_state.get('selected_event', "")

    # Guard in case default_event is not in event_names (avoid ValueError)
    if default_event in event_names:
        default_index = event_names.index(default_event) + 1  # +1 because we add "" option at front
    else:
        default_index = 0

    # selected_event = st.selectbox(
#         "Choose an event to view its leaderboard:",
#         [""] + event_names,
#         index=default_index,
#     )

    selected_event = st.session_state.get('selected_event', "")
    
    if not selected_event:
        st.info("Please select a competition using the sidebar.")
    else:
        event_df = df[df['Full_Event'] == selected_event]
        display_leaderboard(event_df, df, selected_event)

    # Update session_state with user's manual change to the dropdown (optional)
    st.session_state['selected_event'] = selected_event

    st.markdown('---')
        
    st.markdown(bottom_string)

        
        
# ---------- Puzzler Profiles Page ----------
if page == "Puzzler Profiles":
    st.title("👤 Puzzler Profiles ")

    # puzzler_list = sorted(df['Name'].dropna().unique())
    # default_puzzler = st.session_state.get('selected_puzzler', "")
    # selected_puzzler = st.selectbox("Select a puzzler", [""] + puzzler_list, index=(puzzler_list.index(default_puzzler) + 1) if default_puzzler in puzzler_list else 0)
    # st.session_state['selected_puzzler'] = selected_puzzler
    
    selected_puzzler = st.session_state.get('selected_puzzler',"")
    if not selected_puzzler:
        st.info("Please select a profile using the sidebar.")
    else:
        display_puzzler_profile(df, selected_puzzler)
        
    st.markdown('---')
    st.markdown(bottom_string)
    

# ---------- JPAR ----------
if page == "JPAR":
    display_jpar_ratings(styled_table, results)
    st.markdown('---')
    st.markdown(bottom_string)
