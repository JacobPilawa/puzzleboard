import streamlit as st
from utils.helpers import get_ranking_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"
    
    

def display_summary_stats(comparison_df, results, df):
    st.markdown("""Here's some summary information for the selected puzzlers, sorted by their average time over the last 12 months.""")

    # drop some irrelevant columns
    summary_df = comparison_df[['Name', 'Time', 'PPM', '12-Month Avg Completion Time', 'time_in_seconds', 'Date']].copy()

    # Get current date and filter for last 12 months
    latest_date = summary_df['Date'].max()
    one_year_ago = latest_date - pd.DateOffset(years=1)
    recent_df = summary_df[summary_df['Date'] >= one_year_ago]

    # get overall summary statistics
    grouped_summary = (
        summary_df
        .groupby('Name')
        .agg(
            avg_time_in_seconds=('time_in_seconds', 'mean'),
            avg_ppm=('PPM', 'mean'),
            count=('Name', 'size')
        )
        .reset_index()
    )

    # get last-year-only summary statistics
    recent_summary = (
        recent_df
        .groupby('Name')
        .agg(
            last_year_avg_time=('time_in_seconds', 'mean'),
            last_year_avg_ppm=('PPM', 'mean'),
            last_year_count=('Name', 'size')
        )
        .reset_index()
    )

    # merge both summaries
    grouped_summary = grouped_summary.merge(recent_summary, on='Name', how='left')

    # format full-period avg time
    grouped_summary['Average Time'] = grouped_summary['avg_time_in_seconds'].apply(
        lambda x: f"{int(x // 3600):02}:{int((x % 3600) // 60):02}:{int(x % 60):02}"
    )

    # format last-year avg time
    grouped_summary['Average Time (Last 12 Months)'] = grouped_summary['last_year_avg_time'].apply(
        lambda x: f"{int(x // 3600):02}:{int((x % 3600) // 60):02}:{int(x % 60):02}" if pd.notnull(x) else "-"
    )

    # round PPMs
    grouped_summary['Average PPM'] = grouped_summary['avg_ppm'].round(2)
    grouped_summary['Average PPM (Last 12 Months)'] = grouped_summary['last_year_avg_ppm'].round(2).fillna("-")

    # rename counts
    grouped_summary['Number of Events'] = grouped_summary['count']
    grouped_summary['Number of Events (Last 12 Months)'] = grouped_summary['last_year_count'].fillna(0).astype(int)

    # merge ranks
    results['Rank'] = results.index + 1
    final_summary = grouped_summary.merge(results, on="Name", how='left')

    # select and order final columns
    final_summary = final_summary[[
        'Name', 'Rank',
        'Number of Events', 'Average Time',
        'Number of Events (Last 12 Months)', 'Average Time (Last 12 Months)',
    ]]

    # sort by average time (last 12 months)
    final_summary = final_summary.sort_values(by='Average Time (Last 12 Months)')

    # display the dataframe
    st.dataframe(final_summary.reset_index(drop=True), use_container_width=True)

    # ---- Bar Chart of Avg Time (Last 12 Months) ----
    plot_df = grouped_summary.dropna(subset=['last_year_avg_time']).copy()
    plot_df = plot_df.sort_values(by='last_year_avg_time', ascending=True)  # Fastest on left
    plot_df['time_in_hours'] = plot_df['last_year_avg_time'] / 3600
    plot_df['avg_time_hms'] = plot_df['last_year_avg_time'].apply(seconds_to_hms)  # New column

    hover_template = (
        '<b>Solver:</b> %{x}<br>'
        '<b>Avg Time (Last 12 Months):</b> %{customdata[0]}<br>'
        '<b>PPM:</b> %{customdata[1]:.2f}<br>'
        '<b>Events:</b> %{customdata[2]:.0f}<br>'
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=plot_df['Name'],
        y=plot_df['time_in_hours'],
        marker=dict(color='tomato'),
        customdata=plot_df[['avg_time_hms', 'last_year_avg_ppm', 'last_year_count']].values,
        hovertemplate=hover_template
    ))

    fig.update_layout(
        title='Average Completion Time (Last 12 Months)',
        xaxis_title='Solver',
        yaxis_title='Avg Completion Time (hours)',
        template='plotly_white',
        font=dict(color='black', size=12),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)    
    
def display_all_stats(comparison_df, results, df):
    st.markdown("""Here's all the information for all of the puzzlers selected above, sorted first by name and then by date of event.""")

    # filter only the columns we want
    display_df = comparison_df.sort_values('Date')[['Name','Date', 'Full_Event', 'Rank', 
    'Time', 'PPM', 'Remaining','Avg PTR In (Event)', 'PTR Out','12-Month Avg Completion Time']].copy()

    # rename some columns
    display_df = display_df.rename(columns={'Full_Event': 'Event Name'})

    # convert to date
    display_df['Date'] = display_df['Date'].dt.date

    # Calculate total entrants per event
    event_totals = df.groupby('Full_Event')['Name'].count()

    # Create Rank column as "N/T"
    display_df['Rank'] = display_df.apply(lambda row: f"{int(row['Rank'])}/{event_totals.get(row['Event Name'], 0)}", axis=1)

    # Sort by most recent
    display_df = display_df.sort_values(by=['Name', 'Date'], ascending=False)

    # --- Bar Plot of All Times ---
    plot_df = comparison_df[['Name', 'time_in_seconds', 'Full_Event']].copy()
    plot_df = plot_df.sort_values(by='time_in_seconds')  # Fastest first
    plot_df['time_in_hours'] = plot_df['time_in_seconds'] / 3600
    
    # Create unique x labels for consistent spacing
    plot_df['label'] = [f'{n} {i}' for i, n in enumerate(plot_df['Name'])]
    
    # Assign one consistent color per puzzler
    color_seq = px.colors.qualitative.Plotly
    seen = {}
    ordered_names = []
    
    # Keep track of order of appearance
    for name in plot_df['Name']:
        if name not in seen:
            seen[name] = True
            ordered_names.append(name)
    
    color_map = {name: color_seq[i % len(color_seq)] for i, name in enumerate(ordered_names)}
    plot_df['color'] = plot_df['Name'].map(color_map)
    plot_df['time_in_hms'] = plot_df['time_in_seconds'].apply(seconds_to_hms)
    
    
    # Bar plot using go.Bar
    hover_template = (
        '<b>Solver:</b> %{customdata[0]}<br>'
        '<b>Time:</b> %{customdata[2]}<br>'
        '<b>Event:</b> %{customdata[1]}<br>'
        '<extra></extra>'
    )
    
    fig = go.Figure()
    
    # Single trace for all bars sorted by time
    fig.add_trace(go.Bar(
        x=plot_df['label'],
        y=plot_df['time_in_hours'],
        marker=dict(color=plot_df['color']),
        customdata=plot_df[['Name', 'Full_Event', 'time_in_hms']],
        hovertemplate=hover_template,
        showlegend=False
    ))
    
    # Add dummy traces for legend entries per person
    for name in ordered_names:
        fig.add_trace(go.Bar(
            x=[None], y=[None],  # Dummy invisible bar for legend only
            name=name,
            marker=dict(color=color_map[name]),
            showlegend=True
        ))
    
    fig.update_layout(
        title='All Completion Times (Sorted by Fastest)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(title='Completion Time (hours)'),
        template='plotly_white',
        font=dict(size=12),
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Final Table ---
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
    


def display_comparison(styled_table, results, df):
    
    # all names
    puzzler_names = sorted(df['Name'].dropna().unique())
    
    # names currently selected
    selected_puzzlers = st.multiselect(
        "Select puzzlers to compare:",
        puzzler_names,
    )

    if len(selected_puzzlers) < 2:
        st.info("Please select at least two puzzlers.")
    else:
        
        # get dataframe of these puzzlers
        comparison_df = df[df['Name'].isin(selected_puzzlers)]
        
        st.subheader("📊 Summary Statistics")
        display_summary_stats(comparison_df, results, df)
        
        st.subheader("📄 All Events")
        display_all_stats(comparison_df, results, df)
                        
        