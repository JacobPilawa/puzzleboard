import streamlit as st
from utils.helpers import get_ranking_table, load_data, get_bottom_string

#### GET DATA
df = load_data()
styled_table, results = get_ranking_table(min_puzzles=3, min_event_attempts=10, weighted=False)
bottom_string = get_bottom_string()

# ✅ MUST BE FIRST
st.set_page_config(
    page_title="Speed Puzzling Dashboard",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Header
st.markdown(
    """
    <style>
        /* Style the sidebar header */
        [data-testid="stSidebarNav"]::before {
            content: "📍 Navigation";
            display: block;
            padding: 10px;
            font-size: 1.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

dashboard = st.Page("nav/dashboard.py", title="Home", icon='🧩', default=True)
competitions = st.Page("nav/competitions.py", title="Competitions", icon='🏆')
profiles = st.Page("nav/profiles.py", title="Puzzlers", icon='👤')
profiles_sel = st.Page("nav/profiles.py", title="Puzzlers", icon='👤')
comparisons = st.Page("nav/comparisons.py",title="Compare Puzzlers", icon='⚔️')
ratings = st.Page("nav/ratings.py", title="Rankings", icon='📊')
pages = { "": [dashboard, competitions, profiles, comparisons, ratings]}


# custom css to increase button and font sizes
st.markdown(
    """
    <style>
      /* Increase font size and padding of nav links */
      div[data-testid="stSidebarNav"] li div a {
        padding: 0.75rem 1rem;
      }
      div[data-testid="stSidebarNav"] li div a span {
        font-size: 1.2rem; /* or 18px, 20px as needed */
      }

      /* Add hover background for links */
      div[data-testid="stSidebarNav"] li div a:hover {
        background-color: rgba(100, 150, 200, 0.1);
      }

      /* Enlarge section headers, if you're using grouped pages */
      div[data-testid="stSidebarNav"] > ul[data-testid="stSidebarNavItems"] > header div p {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
      }

      /* Optional: make the nav area wider */
      div[data-testid="stSidebarNav"] {
        min-width: 220px;
      }
    </style>
    """,
    unsafe_allow_html=True
)


# --------- QUICK JUMP ----------
# to come

pg = st.navigation(pages,position="sidebar")
pg.run()