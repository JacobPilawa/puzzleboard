import pandas as pd
import streamlit as st
import numpy as np
from datetime import timedelta
import string
import plotly.express as px
import plotly.graph_objects as go
from utils.helpers import get_standard_ranking_table, load_data, get_bottom_string, get_delta_color

#### GET DATA
df = load_data()
styled_table, results = get_standard_ranking_table()
bottom_string = get_bottom_string()

# ---------- Puzzler Profile Display Function ----------
def display_puzzler_profile(df: pd.DataFrame, selected_puzzler: str, results):
    
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
                xaxis=dict(
                    type='date',
                    tickformat='%b\n%Y',  # Month abbreviation + year
                    dtick='M1',  # Monthly ticks
                    tickangle=0,
                ),
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
        
        if date_spacing:
            years = sorted(jittered_df['Date'].dt.year.unique())
            shapes = []
            for i, year in enumerate(years):
                if i % 2 == 1:
                    # Alternate (every second year): light gray rectangle
                    start_date = pd.Timestamp(f'{year}-01-01')
                    end_date = pd.Timestamp(f'{year+1}-01-01')
                    shapes.append(
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=start_date,
                            x1=end_date,
                            y0=0,
                            y1=1,
                            fillcolor="LightGray",
                            opacity=0.3,
                            layer="below",
                            line_width=0,
                        )
                    )
            fig.update_layout(shapes=shapes)
    
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No solve time data available to plot for the selected filter.")

    # ---------------- Individual Competition Results ----------------
    st.subheader("🏆 Individual Competition Results")

    # Get events this puzzler has entered
    available_events = ["Select a competition"] + sorted(puzzler_df['Full_Event'].unique())
    selected_event = st.selectbox("Select a competition event:", available_events, key=f"{selected_puzzler}_event")

    if selected_event != "Select a competition":
        event_df = df[df['Full_Event'] == selected_event].copy()
        filtered_df = event_df.copy()
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date']).dt.date

        # --- Stats ---
        total_entrants = filtered_df['Name'].nunique()
        fastest_time = filtered_df['time_in_seconds'].min()
        avg_time = filtered_df['time_in_seconds'].mean()

        # Puzzler’s own time
        puzzler_time = filtered_df.loc[filtered_df['Name'] == selected_puzzler, 'time_in_seconds'].iloc[0]

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
            st.metric("Total Entrants", total_entrants, border=True)
        with col2:
            st.metric("Fastest Time", str(timedelta(seconds=int(fastest_time))), border=True)
        with col3:
            st.metric("Average Time", str(timedelta(seconds=int(avg_time))), border=True)
        with col4:
            st.metric(f"{selected_puzzler}'s Time", str(timedelta(seconds=int(puzzler_time))), border=True)

        # --- Bar Plot ---
        plot_df = filtered_df.copy()
        plot_df['time_in_hours'] = plot_df['time_in_seconds'] / 3600
        plot_df.sort_values('time_in_seconds', inplace=True)
        n_all = len(plot_df)
        plot_df['performance'] = np.linspace(1, 0, n_all) if n_all > 1 else 1.0

        hover_template = (
            '<b>Solver:</b> %{customdata[0]}<br>'
            '<b>Rank:</b> %{customdata[4]:.0f}<br>'
            '<b>Time:</b> %{customdata[1]}<br>'
            '<b>Date:</b> %{customdata[2]|%Y-%m-%d}<br>'
            '<b>PPM:</b> %{customdata[3]:.1f}<br>'
            '<b>Performance:</b> %{x:.0%}'
        )

        fig = go.Figure()

        for _, row in plot_df.iterrows():
            color = "blue" if row['Name'] == selected_puzzler else "tomato"
            fig.add_trace(go.Bar(
                x=[row['performance']],
                y=[row['time_in_hours']],
                marker=dict(color=color),
                customdata=[[row['Name'], row['Time'], row['Date'], row['PPM'], row['Rank']]],
                hovertemplate=hover_template,
                showlegend=False
            ))

        tick_vals = [i/10 for i in range(0, 11)]
        tick_text = [f"{int(val*100)}%" for val in tick_vals]

        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, autorange='reversed'),
            yaxis_title='Completion Time (hours)',
            title=f"Completion Times – {selected_event}",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Leaderboard Table ---
        st.subheader(f"Leaderboard for {selected_event}")
        display_df = filtered_df.sort_values('Rank')[['Name', 'Time', 'Date', 'PPM', 'Pieces']].reset_index(drop=True)
        display_df.index = display_df.index + 1
        st.dataframe(display_df, use_container_width=True)


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

    

st.title("👤 Puzzler Profiles ")

# Add a blank option as the first item
available_puzzlers = ["Select a puzzler"] + sorted(df['Name'].unique())
selected_puzzler = st.selectbox("Select a puzzler:", available_puzzlers)

# Only proceed if a real event is selected
if selected_puzzler != "Select a puzzler":
    st.session_state['selected_puzzler'] = selected_puzzler
    puzzler_df = df[df['Name'] == selected_puzzler]
    display_puzzler_profile(df, selected_puzzler, results)
else:
    st.info("Please select a puzzler from the dropdown above.")

# Footer
st.markdown('---')
st.markdown(bottom_string)


