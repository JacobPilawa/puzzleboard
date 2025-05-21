import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

def compute_speed_puzzle_rankings(combined_df, min_puzzles=9, min_event_attempts=1, weighted=True):
    # Copy DataFrame for processing
    df = pd.DataFrame.copy(combined_df)

    # Filter puzzles from the past year
    start_time = pd.Timestamp.today() - pd.DateOffset(years=1)
    df = df[df['Date'] >= start_time]

    # Filter puzzles with enough solvers
    df = df[df['Event'].map(df['Event'].value_counts()) >= min_event_attempts]

    # Add total puzzles completed by each solver
    df['Total_Puzzles'] = df['Name'].map(df['Name'].value_counts())

    # Puzzle-level stats
    puzzle_stats = df.groupby('Event')['corrected_time'].agg(
        mean='mean',
        std='std',
        count='count',
        median='median',
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index().rename(columns={
        'mean': 'puzzle_mean',
        'median': 'puzzle_median',
        'std': 'puzzle_std',
        'count': 'puzzle_count',
        'iqr': 'puzzle_iqr'
    })

    # Merge stats into main DataFrame
    df = df.merge(puzzle_stats, on='Event', how='left')

    # Handle edge cases for std and iqr
    df.loc[(df['puzzle_count'] == 1) | (df['puzzle_std'] == 0) | (df['puzzle_std'].isna()), 'puzzle_std'] = 1
    df.loc[(df['puzzle_iqr'] == 0) | (df['puzzle_iqr'].isna()), 'puzzle_iqr'] = 1

    # Compute z-scores (higher is better)
    df['z_score'] = - (df['corrected_time'] - df['puzzle_mean']) / df['puzzle_std']

    # Percentile within each puzzle (lower corrected time = better)
    df['percentile'] = df.groupby('Event')['corrected_time'].rank(pct=True, ascending=True) * 100

    # Basic (unweighted) summaries
    z_summary = df.groupby('Name')['z_score'].mean().rename('Norm_Score').to_frame()
    z_summary['Avg_Percentile'] = df.groupby('Name')['percentile'].mean()

    # Weighted versions, if requested
    if weighted:
        df['weighted_z'] = df['z_score'] * df['puzzle_count']
        df['weighted_percentile'] = df['percentile'] * df['puzzle_count']
        weight_sums = df.groupby('Name')['puzzle_count'].sum()
        weighted_z_score = df.groupby('Name')['weighted_z'].sum() / weight_sums
        weighted_percentile = df.groupby('Name')['weighted_percentile'].sum() / weight_sums

        z_summary['Weighted_Norm_Score'] = weighted_z_score
        z_summary['Weighted_Avg_Percentile'] = weighted_percentile

    # Total puzzles per person
    total_puzzles = df['Name'].value_counts().rename('Total_Puzzles')
    z_summary = z_summary.merge(total_puzzles, left_index=True, right_index=True)

    # Filter by minimum puzzle count
    z_summary = z_summary[z_summary['Total_Puzzles'] >= min_puzzles]

    # Assign ranks
    z_summary = z_summary.sort_values(by='Norm_Score', ascending=False)
    z_summary["Z_Score_Rank"] = range(1, len(z_summary) + 1)
    z_summary['Percentile_Rank'] = z_summary['Avg_Percentile'].rank(method='min')

    if weighted:
        z_summary['Weighted_Z_Score_Rank'] = z_summary['Weighted_Norm_Score'].rank(ascending=False, method='min')
        z_summary['Weighted_Percentile_Rank'] = z_summary['Weighted_Avg_Percentile'].rank(method='min')

    # Merge with Rob's rankings
    rob_ranks = pd.read_csv('./data/Puzzle Competitor Data (Cleaned Up) - _Latest JPAR Ratings.csv')
    rob_ranks = rob_ranks.rename(columns={"Player": "Name"})
    rob_ranks = rob_ranks.sort_values(by='Latest JPAR Out')
    rob_ranks["rob_rank"] = range(1, len(rob_ranks) + 1)

    compare_df = rob_ranks.merge(z_summary, on='Name')
    compare_df["rob_rank"] = range(1, len(compare_df) + 1)

    # Final result table
    if weighted:
        results = compare_df[['Name','Total Events','Total_Puzzles','rob_rank','Z_Score_Rank',
                              'Weighted_Z_Score_Rank','Percentile_Rank','Weighted_Percentile_Rank']]
        results['avg_rank'] = results[['rob_rank','Z_Score_Rank','Weighted_Z_Score_Rank',
                                       'Percentile_Rank','Weighted_Percentile_Rank']].mean(axis=1)
    else:
        results = compare_df[['Name','Total Events','Total_Puzzles','rob_rank','Z_Score_Rank','Percentile_Rank']]
        results['avg_rank'] = results[['rob_rank','Z_Score_Rank','Percentile_Rank']].mean(axis=1)

    results = results.sort_values(by='avg_rank')

    # Convert appropriate columns to integers
    int_columns = results.columns[1:-1]
    results[int_columns] = results[int_columns].astype(int)

    # Display index from 1 to 100
    display_index = pd.Index(np.arange(1, 101))

    # drop events temporarily, need to FIX THIS
    results = results.drop(columns=['Total Events'])
    results = results.rename(columns={
        'Total_Puzzles': 'Eligible Puzzles',
        'rob_rank': 'JPAR Rank',
        'Z_Score_Rank': 'Z Rank',
        'Percentile_Rank': 'Percentile Rank',
        'avg_rank': 'Average Rank'
    })

    # Styled DataFrame
    if weighted:
        styled_df = (
            results.head(100)
            .set_index(display_index)
            .style
            .background_gradient(subset=[ 'Eligible Puzzles'], cmap='Purples')
            .background_gradient(
                subset=['JPAR Rank', 'Z Rank', 'Weighted_Z_Score_Rank',
                        'Percentile Rank', 'Weighted_Percentile_Rank', 'Average Rank'],
                cmap='RdYlGn'
            )
            .format({'avg_rank': '{:.1f}'})
        )
    else:
        styled_df = (
            results.head(100)
            .set_index(display_index)
            .style
            .background_gradient(subset=['Eligible Puzzles'], cmap='Purples')
            .background_gradient(
                subset=['JPAR Rank', 'Z Rank', 'Percentile Rank', 'Average Rank'],
                cmap='RdYlGn'
            )
            .format({'avg_rank': '{:.1f}'})
        )


    return styled_df, results.reset_index(drop=True)


def get_ranking_table(min_puzzles=10, min_event_attempts=1, weighted=False):
    data = pd.read_pickle('./data/250519_scape_with_event_names.pkl')

    styled_df, results = compute_speed_puzzle_rankings(data, min_puzzles, min_event_attempts, weighted)
    
    return styled_df, results