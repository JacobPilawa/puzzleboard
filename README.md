# Puzzleboard
This repository contains the code (and some of the data) for [puzzleboard](https://puzzleboard.streamlit.app/). Data curated by Rob Shields of the Piece Talks podcast.

#### Known Bugs
- weird dropdown behavior when something is already selected. for example, selecting an event page, and then trying to select a second event page does nothing. you need to do it an extra time. i suspect i need to add an st.re_run() or page state change somewhere, but it'll be fine for now.
