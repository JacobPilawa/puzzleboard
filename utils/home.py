import pandas as pd
import streamlit as st
from datetime import timedelta
import plotly.express as px


# ---------- Plotting Utilities ----------
@st.cache_data
def get_top_medalists(df, top_n=20):
    # Count medals
    medal_counts = df[df['Rank'].isin([1, 2, 3])].copy()
    medal_counts['Gold'] = (medal_counts['Rank'] == 1).astype(int)
    medal_counts['Silver'] = (medal_counts['Rank'] == 2).astype(int)
    medal_counts['Bronze'] = (medal_counts['Rank'] == 3).astype(int)

    grouped = medal_counts.groupby('Name')[['Gold', 'Silver', 'Bronze']].sum()
    grouped['Total'] = grouped.sum(axis=1)
    top = grouped.sort_values('Total', ascending=False).head(top_n).drop(columns='Total')
    
    return top.reset_index()

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

@st.cache_data
def get_most_frequent_puzzlers(df):
    
    # get the 50 most frequent entrants to events
    frequent_puzzlers = df['Name'].value_counts().head(20)
    
    return frequent_puzzlers

# ---------- Display ----------
def display_home(df: pd.DataFrame):
    st.markdown(
        "<h1 style='text-align: center;'>🧩 Speed Puzzling Competitions Dashboard </h1>",
        unsafe_allow_html=True
    )
    st.markdown("This dashboard contains only the **official results** from a number of **500 piece** speed puzzling competitions (speedpuzzling.com, EJJ, various local and national events, and more). Access competition results, puzzler profiles, and rankings using the sidebar on the left.")

    # --- Metrics block ---
    cumulative_names, cumulative_events, cumulative_entries = get_cumulative_stats(df)
    frequent_puzzlers = get_most_frequent_puzzlers(df)
    
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
    st.subheader("🧮 Most Events Entered")
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
        title="Top 20 Puzzlers by Number of Events",
        color_continuous_scale='Tealgrn' 
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
    
    # --------- Medal Counts Plot ---------
    st.subheader("🏅 Medal Counts")

    top_medalists_df = get_top_medalists(df)  # Use the full df, not puzzler_df

    fig = px.bar(
        top_medalists_df,
        x='Name',
        y=['Gold', 'Silver', 'Bronze'],
        title="Top Medal Winners",
        labels={"value": "Number of Medals", "Name": "Puzzler", "variable": "Medal Type"},
        color_discrete_map={
            "Gold": "#FFD700",
            "Silver": "#C0C0C0",
            "Bronze": "#CD7F32"
        }
    )
    fig.update_layout(barmode='stack', xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)



