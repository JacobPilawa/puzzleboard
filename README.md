# Puzzleboard
This repository contains the code (and some of the data) for [puzzleboard](https://puzzleboard.streamlit.app/). Data curated by Rob Shields of the Piece Talks podcast. Streamlit app in streamlit_app.py; helpers.py contains a variety of plotting/data cleaning functions.

#### Updating Data
- For now, you have to manually rescrape data when it's updated in the Google Sheet (updating this to be automatic is on my to-do list). To update, you can pretty much just call the scrape_data(filename) function in utils/helpers.py, and update which files is being read in helpers.py load_data(). 


#### Known Bugs
- i think this is a streamlit issue, but there's weird case sensitivity issues for some of the st.selectbox() inputs. it appears as though it prioritizes string match and case, but in some very mysterious order
- inconsistencies in some calculations between tables and charts due to where the data is coming from. for example rob's calculation of the 12 month moving average is for PT rank qualified puzzles only, whereas mine uses the full dataset from all the sheets. not a "bug" but needs to be better documented.
- need better descriptions of everything throughout. very barebones currently
- if x-axis toggled on bar plot is toggled, and then another puzzler page is loaded, the box stays checked but does not apply to the puzzler

### Ideas
- add plot of PTR over time (and maybe the other rankings too?)
- NN to take in images from ipdb and classify into box image/poster/normal image etc
- predicting brand based on image
