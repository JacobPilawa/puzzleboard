from utils.helpers import get_ranking_table, load_data, load_jpar_data, get_delta_color
from utils.profiles import display_puzzler_profile
from utils.leaderboards import display_leaderboard
from utils.ratings import display_jpar_ratings
from utils.home import display_home

import string
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
    st.session_state['selected_puzzler'] = ""
    
if st.sidebar.button("👤 Puzzler Profiles "):
    st.session_state.page = "Puzzler Profiles"
    st.session_state['selected_event'] = ""
    st.session_state['selected_puzzler'] = ""
    
if st.sidebar.button("📊 Puzzler Ratings "):
    st.session_state.page = "JPAR"
    st.session_state['selected_event'] = ""
    st.session_state['selected_puzzler'] = ""
    
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

# ---------- Home ----------
if page == "Home":
    
    display_home(df)
    st.markdown('---')
    st.markdown(bottom_string)
        
# ---------- Competitions Page ----------
if page == "Competitions":
    
    st.title("🏆 Competitions ")
    # check if there's an event selected
    selected_event = st.session_state.get('selected_event', "")
    if not selected_event:
        st.info("Please select a competition using the sidebar.")
    else:
        event_df = df[df['Full_Event'] == selected_event]
        display_leaderboard(event_df, df, selected_event)

    st.markdown('---')
        
    st.markdown(bottom_string)

# ---------- Puzzler Profiles Page ----------
if page == "Puzzler Profiles":
    
    st.title("👤 Puzzler Profiles ")
    # check if there's someone selected
    selected_puzzler = st.session_state.get('selected_puzzler',"")
    if not selected_puzzler:
        st.info("Please select a profile using the sidebar.")
    else:
        display_puzzler_profile(df, selected_puzzler, results)
        
    st.markdown('---')
    st.markdown(bottom_string)
    

# ---------- JPAR ----------
if page == "JPAR":
    
    display_jpar_ratings(styled_table, results, df)
    st.markdown('---')
    st.markdown(bottom_string)
