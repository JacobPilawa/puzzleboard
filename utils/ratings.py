import streamlit as st
from helpers import get_ranking_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
