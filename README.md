# Puzzleboard
This repository contains the code (and some of the data) for [puzzleboard](https://puzzleboard.streamlit.app/). Data curated by Rob Shields of the Piece Talks podcast.

#### Known Bugs
- weird dropdown behavior when something is already selected. for example, selecting an event page, and then trying to select a second event page does nothing. you need to do it an extra time. i suspect i need to add an st.re_run() or page state change somewhere, but it'll be fine for now.

### Ideas
- add highest medal counts to leaderboard landing page/puzzler profile page instead of the search bars
- NN to take in images from ipdb and classify into box image/poster/normal image etc
