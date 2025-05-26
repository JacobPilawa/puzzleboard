# Puzzleboard
This repository contains the code (and some of the data) for [puzzleboard](https://puzzleboard.streamlit.app/). Data curated by Rob Shields of the Piece Talks podcast. Streamlit app in streamlit_app.py; helpers.py contains a variety of plotting/data cleaning functions.

#### Known Bugs
- inconsistencies in some calculations between tables and charts due to where the data is coming from. for example rob's calculation of the 12 month moving average is for PT rank qualified puzzles only, whereas mine uses the full dataset from all the sheets. not a "bug" but needs to be better documented.
- need better descriptions of everything throughout. very barebones currently
- if x-axis toggled on bar plot is toggled, and then another puzzler page is loaded, the box stays checked but does not apply to the puzzler

### Ideas
- add highest medal counts to leaderboard landing page/puzzler profile page instead of the search bars
- NN to take in images from ipdb and classify into box image/poster/normal image etc
