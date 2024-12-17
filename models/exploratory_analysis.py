import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

GRAPHS_DIR = Path(__file__).parent / 'graphs'
GRAPHS_DIR.mkdir(exist_ok=True)

from tabular_regression import load_player_stats, create_dataset, get_maps

def clean_dataset(df):
    """Clean the dataset and prepare it for analysis"""
    df_clean = df.copy()
    
    # Store categorical columns separately
    categorical_cols = ['name', 'map_name']
    categorical_data = {}
    for col in categorical_cols:
        if col in df_clean.columns:
            categorical_data[col] = df_clean.pop(col)
    
    # First, explicitly convert rating_diff to numeric
    if 'rating_diff' in df_clean.columns:
        df_clean['rating_diff'] = pd.to_numeric(df_clean['rating_diff'], errors='coerce')
    
    # Convert all remaining columns to numeric, replacing errors with NaN
    for col in df_clean.columns:
        if col != 'rating_diff_bin':  # Skip the bin column if it exists
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove rows where all numeric columns are NaN
    df_clean = df_clean.dropna(how='all', subset=[col for col in df_clean.columns])
    
    # Add back categorical columns
    for col, data in categorical_data.items():
        df_clean[col] = data
    
    # Print some debug information
    print("\nDebug information after cleaning:")
    print("Number of non-null values in rating_diff:", df_clean['rating_diff'].count())
    print("Number of total rows:", len(df_clean))
    print("Sample of rating_diff values:", df_clean['rating_diff'].head())
    print("Unique map names:", df_clean['map_name'].unique() if 'map_name' in df_clean.columns else "No map_name column")
    
    return df_clean

def analyze_correlations(df):
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlations with kills_y
    correlations = numeric_df.corr()['kills_y'].sort_values(ascending=False)
    print("\nTop 10 Positive Correlations with kills_y:")
    print(correlations[1:11])  # Skip kills_y itself
    print("\nTop 10 Negative Correlations with kills_y:")
    print(correlations[-10:])
    
    # Create correlation heatmap for most important features
    top_features = correlations.abs().nlargest(15).index
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_df[top_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap (Top 15 Features)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'correlation_heatmap.png')
    plt.close()

def analyze_prediction_confidence(df):
    """Analyze situations where predictions might be more reliable"""
    
    # Drop rows where rating_diff is NaN
    df_valid = df.dropna(subset=['rating_diff'])
    
    if len(df_valid) == 0:
        print("\nNo valid rating_diff values found. Skipping prediction confidence analysis.")
        return
    
    # Create bins for rating difference
    try:
        df_valid['rating_diff_bin'] = pd.qcut(df_valid['rating_diff'].abs(), q=10, duplicates='drop')
    except ValueError as e:
        print(f"\nError creating rating difference bins: {e}")
        print("Rating diff statistics:")
        print(df_valid['rating_diff'].describe())
        return
    
    # Analyze prediction error by rating difference
    print("\nAnalyzing kill patterns by rating difference:")
    for bin_label in df_valid['rating_diff_bin'].unique():
        bin_data = df_valid[df_valid['rating_diff_bin'] == bin_label]
        mean_kills = bin_data['kills_y'].mean()
        std_kills = bin_data['kills_y'].std()
        print(f"\nRating diff bin {bin_label}:")
        print(f"Mean kills: {mean_kills:.2f}")
        print(f"Std kills: {std_kills:.2f}")
        print(f"Coefficient of variation: {std_kills/mean_kills:.2f}")
    
    # Look at extreme matchups
    extreme_matches = df_valid[df_valid['rating_diff'].abs() > df_valid['rating_diff'].quantile(0.9)]
    print("\nExtreme matchups (top 10% rating difference):")
    print(extreme_matches['kills_y'].describe())

def analyze_player_variance(df):
    """Analyze and visualize player kill variance and distribution shape"""
    # Calculate player statistics including skewness
    player_stats = df.groupby('name').agg({
        'kills_y': ['mean', 'std', 'count', 'skew']
    }).reset_index()
    player_stats.columns = ['name', 'mean_kills', 'std_kills', 'match_count', 'skewness']
    
    # Filter for players with enough matches
    min_games = 30
    player_stats = player_stats[player_stats['match_count'] >= min_games]
    
    # Create two sorted versions - one by std dev and one by absolute skewness
    player_stats_std_sorted = player_stats.sort_values('std_kills', ascending=True)
    player_stats_skew_sorted = player_stats.sort_values('skewness', key=abs, ascending=False)
    
    # Create visualization for standard deviation
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(player_stats_std_sorted)), player_stats_std_sorted['std_kills'], 'b-', alpha=0.7)
    plt.title('Players Sorted by Kill Standard Deviation')
    plt.xlabel('Player Rank (by std dev)')
    plt.ylabel('Kill Standard Deviation')
    
    # Add some annotations for extreme cases
    n_annotate = 5  # Number of players to annotate at each end
    
    # Annotate lowest variance players
    for i in range(n_annotate):
        player = player_stats_std_sorted.iloc[i]
        plt.annotate(f"{player['name']}\n(μ={player['mean_kills']:.1f})", 
                    (i, player['std_kills']),
                    xytext=(10, 10), textcoords='offset points')
    
    # Annotate highest variance players
    for i in range(-n_annotate, 0):
        player = player_stats_std_sorted.iloc[i]
        plt.annotate(f"{player['name']}\n(μ={player['mean_kills']:.1f})", 
                    (len(player_stats_std_sorted) + i, player['std_kills']),
                    xytext=(10, -10), textcoords='offset points')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'player_kill_variance.png')
    plt.close()
    
    # Create visualization for skewness
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(player_stats_skew_sorted)), player_stats_skew_sorted['skewness'], 'r-', alpha=0.7)
    plt.title('Players Sorted by Kill Distribution Skewness')
    plt.xlabel('Player Rank')
    plt.ylabel('Skewness')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Add line at zero for reference
    
    # Annotate most skewed players (both positive and negative)
    n_annotate = 5  # Number of players to annotate at each end
    
    # Annotate most positively skewed
    for i in range(n_annotate):
        player = player_stats_skew_sorted.iloc[i]
        plt.annotate(f"{player['name']}\n(skew={player['skewness']:.2f})", 
                    (i, player['skewness']),
                    xytext=(10, 10), textcoords='offset points')
    
    # Annotate most negatively skewed
    for i in range(-n_annotate, 0):
        player = player_stats_skew_sorted.iloc[i]
        plt.annotate(f"{player['name']}\n(skew={player['skewness']:.2f})", 
                    (len(player_stats_skew_sorted) + i, player['skewness']),
                    xytext=(10, -10), textcoords='offset points')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'player_kill_skewness.png')
    plt.close()
    
    # Print statistics for highest and lowest variance players
    print("\nPlayers with Most Consistent Kill Counts:")
    print(player_stats_std_sorted.head().to_string(index=False))
    
    print("\nPlayers with Most Variable Kill Counts:")
    print(player_stats_std_sorted.tail().to_string(index=False))
    
    # Print statistics for most skewed distributions
    print("\nPlayers with Most Positively Skewed Kill Distributions (tendency for occasional very high kill games):")
    print(player_stats_skew_sorted.head().to_string(index=False))
    
    print("\nPlayers with Most Negatively Skewed Kill Distributions (tendency for occasional very low kill games):")
    print(player_stats_skew_sorted.tail().to_string(index=False))
    
    # Create example distribution plots for most skewed players
    plt.figure(figsize=(15, 5))
    
    # Plot most positively skewed player
    most_pos_skewed = player_stats_skew_sorted.iloc[0]['name']
    plt.subplot(1, 2, 1)
    sns.histplot(data=df[df['name'] == most_pos_skewed], x='kills_y', kde=True)
    plt.title(f'Most Positively Skewed: {most_pos_skewed}')
    
    # Plot most negatively skewed player
    most_neg_skewed = player_stats_skew_sorted.iloc[-1]['name']
    plt.subplot(1, 2, 2)
    sns.histplot(data=df[df['name'] == most_neg_skewed], x='kills_y', kde=True)
    plt.title(f'Most Negatively Skewed: {most_neg_skewed}')
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'extreme_skewness_distributions.png')
    plt.close()

def analyze_feature_groups(df):
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Group features by type
    player_features = [col for col in numeric_df.columns if not (col.startswith('ally_') or col.startswith('enemy_') or col == 'kills_y')]
    ally_features = [col for col in numeric_df.columns if col.startswith('ally_')]
    enemy_features = [col for col in numeric_df.columns if col.startswith('enemy_')]
    
    # Calculate average correlation magnitude for each group
    correlations = numeric_df.corr()['kills_y'].abs()
    avg_player_corr = correlations[player_features].mean()
    avg_ally_corr = correlations[ally_features].mean()
    avg_enemy_corr = correlations[enemy_features].mean()
    
    print("\nAverage Correlation Magnitude by Feature Group:")
    print(f"Player's own stats: {avg_player_corr:.3f}")
    print(f"Ally stats: {avg_ally_corr:.3f}")
    print(f"Enemy stats: {avg_enemy_corr:.3f}")

def analyze_distributions(df):
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Plot distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(data=numeric_df, x='kills_y', kde=True)
    plt.title('Distribution of Kills in Matches')
    plt.savefig(GRAPHS_DIR / 'kills_distribution.png')
    plt.close()
    
    # Plot distributions of key predictive features
    correlations = numeric_df.corr()['kills_y'].abs().sort_values(ascending=False)
    top_features = correlations[1:7].index  # Top 6 features excluding kills_y
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(numeric_df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'top_features_distributions.png')
    plt.close()

def analyze_relationships(df):
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Scatter plots of top correlating features vs kills
    correlations = numeric_df.corr()['kills_y'].abs().sort_values(ascending=False)
    top_features = correlations[1:5].index  # Top 4 features excluding kills_y
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 2, i)
        sns.scatterplot(data=numeric_df, x=feature, y='kills_y', alpha=0.5)
        plt.title(f'Kills vs {feature}')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'feature_relationships.png')
    plt.close()

def analyze_rating_diff_impact(df):
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=numeric_df, x='rating_diff', y='kills_y')
    plt.title('Impact of Rating Difference on Kills')
    plt.savefig(GRAPHS_DIR / 'rating_diff_impact.png')
    plt.close()

def analyze_round_distributions(df):
    """Analyze and compare kill distributions across different map rounds"""
    # Debug prints
    print("\nDebug: Round distribution analysis")
    print("DataFrame columns:", df.columns.tolist())
    print("\nUnique values in 'round' column:", df['round'].unique())
    print("\nValue counts for 'round' column:")
    print(df['round'].value_counts())
    
    plt.figure(figsize=(12, 8))
    
    # Create separate datasets for each round
    round3_data = df[df['round'] == 3]['kills_y']
    early_rounds_data = df[df['round'].isin([1, 2])]['kills_y']
    
    # More debug info
    print("\nNumber of round 3 entries:", len(round3_data))
    print("Number of round 1-2 entries:", len(early_rounds_data))
    
    if len(round3_data) == 0 and len(early_rounds_data) == 0:
        print("\nNo round data found. Skipping round distribution analysis.")
        return
    
    # Plot the distributions
    sns.kdeplot(data=round3_data, label='Round 3 Maps', color='red', alpha=0.7)
    sns.kdeplot(data=early_rounds_data, label='Round 1-2 Maps', color='blue', alpha=0.7)
    
    plt.title('Distribution of Kills: Round 3 vs Earlier Rounds')
    plt.xlabel('Number of Kills')
    plt.ylabel('Density')
    plt.legend()
    
    # Add statistical information
    round3_mean = round3_data.mean()
    early_rounds_mean = early_rounds_data.mean()
    round3_std = round3_data.std()
    early_rounds_std = early_rounds_data.std()
    
    stats_text = (
        f'Round 3 Maps (n={len(round3_data)})\n'
        f'Mean: {round3_mean:.2f}\n'
        f'Std: {round3_std:.2f}\n\n'
        f'Round 1-2 Maps (n={len(early_rounds_data)})\n'
        f'Mean: {early_rounds_mean:.2f}\n'
        f'Std: {early_rounds_std:.2f}'
    )
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'round_kill_distributions.png')
    plt.close()
    
    # Print additional analysis
    print("\nKill Distribution Analysis by Round:")
    print(f"Number of Round 3 maps: {len(round3_data)}")
    print(f"Number of Round 1-2 maps: {len(early_rounds_data)}")
    print(f"\nRound 3 mean kills: {round3_mean:.2f} (±{round3_std:.2f})")
    print(f"Round 1-2 mean kills: {early_rounds_mean:.2f} (±{early_rounds_std:.2f})")
    
    # Calculate and print the percentage of matches that go to round 3
    total_maps = len(df['round'].unique())
    round3_maps = len(df[df['round'] == 3]['round'].unique())
    round3_percentage = (round3_maps / total_maps) * 100
    print(f"\nPercentage of matches going to Round 3: {round3_percentage:.1f}%")

def analyze_map_performance(df):
    """Analyze how players perform differently across different maps"""
    # Debug prints
    print("\nDebug: Map performance analysis")
    print("DataFrame columns:", df.columns.tolist())
    print("\nUnique map names:", df['map_name'].unique())
    print("\nMap name value counts:")
    print(df['map_name'].value_counts())
    
    # Calculate overall average kills for each player
    player_overall_avg = df.groupby('name')['kills_y'].mean()
    print("\nNumber of players with overall averages:", len(player_overall_avg))
    
    # Calculate average kills per map for each player
    player_map_avg = df.groupby(['name', 'map_name'])['kills_y'].agg(['mean', 'count']).reset_index()
    print("\nShape of player_map_avg before filtering:", player_map_avg.shape)
    
    # Only consider maps where player has played at least 5 times
    min_maps = 20
    player_map_avg = player_map_avg[player_map_avg['count'] >= min_maps]
    print("\nShape of player_map_avg after filtering:", player_map_avg.shape)
    
    if len(player_map_avg) == 0:
        print("\nNo players found with sufficient map data. Skipping map performance analysis.")
        return
    
    # Add overall average to the map averages
    player_map_avg['overall_avg'] = player_map_avg['name'].map(player_overall_avg)
    player_map_avg['kill_diff_from_avg'] = player_map_avg['mean'] - player_map_avg['overall_avg']
    
    # Find the biggest variations
    player_variations = []
    for player in player_map_avg['name'].unique():
        player_data = player_map_avg[player_map_avg['name'] == player]
        if len(player_data) >= 2:  # Only consider players with data for at least 2 maps
            max_diff = player_data['kill_diff_from_avg'].max()
            min_diff = player_data['kill_diff_from_avg'].min()
            variation = max_diff - min_diff
            best_map = player_data.loc[player_data['kill_diff_from_avg'].idxmax()]
            worst_map = player_data.loc[player_data['kill_diff_from_avg'].idxmin()]
            
            player_variations.append({
                'name': player,
                'variation': variation,
                'best_map': best_map['map_name'],
                'worst_map': worst_map['map_name'],
                'best_map_kills': best_map['mean'],
                'worst_map_kills': worst_map['mean'],
                'overall_avg': best_map['overall_avg'],
                'best_map_games': best_map['count'],
                'worst_map_games': worst_map['count']
            })
    
    print("\nNumber of players with variations:", len(player_variations))
    if len(player_variations) == 0:
        print("No players found with sufficient map variety")
        return
    
    # Convert to DataFrame and sort by variation
    variations_df = pd.DataFrame(player_variations).sort_values('variation', ascending=False)
    
    # Print top 15 players with highest variations
    print("\nTop 15 Players with Highest Map-Based Performance Variation:")
    print("(Minimum {} games per map)".format(min_maps))
    for _, row in variations_df.head(15).iterrows():
        print(f"\n{row['name']}:")
        print(f"  Overall average: {row['overall_avg']:.2f} kills")
        print(f"  Best map: {row['best_map']} - {row['best_map_kills']:.2f} kills (n={int(row['best_map_games'])})")
        print(f"  Worst map: {row['worst_map']} - {row['worst_map_kills']:.2f} kills (n={int(row['worst_map_games'])})")
        print(f"  Variation: {row['variation']:.2f} kills")
        
        # Calculate percentage of games above overall average for each map
        player_data = df[df['name'] == row['name']]
        map_counts = player_data['map_name'].value_counts()
        valid_maps = map_counts[map_counts >= min_maps].index
        player_data = player_data[player_data['map_name'].isin(valid_maps)]
        map_averages = player_data.groupby('map_name')['kills_y'].mean().sort_values(ascending=False)
        
        print(f"  Percentage of games above overall average ({row['overall_avg']:.2f} kills) by map:")
        for map_name in map_averages.index:
            map_data = player_data[player_data['map_name'] == map_name]
            games_above_avg = (map_data['kills_y'] > row['overall_avg']).sum()
            total_games = len(map_data)
            pct_above_avg = (games_above_avg / total_games) * 100
            print(f"    {map_name}: {pct_above_avg:.1f}% ({games_above_avg}/{total_games} games)")
    
    # Create visualization for top 10 players
    plt.figure(figsize=(15, 10))
    top_10_players = variations_df.head(10)
    
    x = np.arange(len(top_10_players))
    width = 0.35
    
    plt.bar(x - width/2, top_10_players['best_map_kills'], width, label='Best Map', color='green', alpha=0.7)
    plt.bar(x + width/2, top_10_players['worst_map_kills'], width, label='Worst Map', color='red', alpha=0.7)
    plt.axhline(y=top_10_players['overall_avg'].mean(), color='blue', linestyle='--', label='Average Performance', alpha=0.5)
    
    plt.xlabel('Player')
    plt.ylabel('Average Kills')
    plt.title('Top 10 Players with Highest Map-Based Performance Variation')
    plt.xticks(x, top_10_players['name'], rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on the bars
    for i, player in enumerate(top_10_players.itertuples()):
        plt.text(i - width/2, player.best_map_kills, f'{player.best_map_kills:.1f}', 
                ha='center', va='bottom')
        plt.text(i + width/2, player.worst_map_kills, f'{player.worst_map_kills:.1f}', 
                ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'map_performance_variation.png')
    plt.close()
    
    # Create detailed distribution plot for player with highest variation
    most_variable_player = variations_df.iloc[0]['name']
    print(f"\nCreating detailed map distribution plot for player with highest variation: {most_variable_player}")
    
    # Filter data for the most variable player
    player_data = df[df['name'] == most_variable_player]
    map_counts = player_data['map_name'].value_counts()
    valid_maps = map_counts[map_counts >= min_maps].index
    player_data = player_data[player_data['map_name'].isin(valid_maps)]
    
    # Calculate average kills per map for ordering
    map_averages = player_data.groupby('map_name')['kills_y'].mean().sort_values(ascending=False)
    
    # Create violin plot
    plt.figure(figsize=(15, 8))
    sns.violinplot(data=player_data, x='map_name', y='kills_y', order=map_averages.index)
    plt.title(f'Kill Distribution by Map for {most_variable_player}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Map')
    plt.ylabel('Number of Kills')
    
    # Add mean values as text
    for i, map_name in enumerate(map_averages.index):
        mean_kills = map_averages[map_name]
        count = map_counts[map_name]
        plt.text(i, plt.ylim()[0], f'n={count}\nμ={mean_kills:.1f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / f'{most_variable_player.lower().replace(" ", "_")}_map_distributions.png')
    plt.close()
    
    # Print detailed statistics
    print(f"\nDetailed statistics for {most_variable_player} by map:")
    stats = player_data.groupby('map_name')['kills_y'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
    print(stats.to_string())
    
    # Calculate percentage of games above overall average for each map
    overall_avg = player_data['kills_y'].mean()
    print(f"\nPercentage of games above overall average ({overall_avg:.2f} kills) by map:")
    for map_name in map_averages.index:
        map_data = player_data[player_data['map_name'] == map_name]
        games_above_avg = (map_data['kills_y'] > overall_avg).sum()
        total_games = len(map_data)
        pct_above_avg = (games_above_avg / total_games) * 100
        print(f"{map_name}: {pct_above_avg:.1f}% ({games_above_avg}/{total_games} games)")

def main():
    # Load data from cache
    print("Loading dataset...")
    from tabular_regression import load_or_create_dataset
    df = load_or_create_dataset(force_refresh=False)
    print("Cleaning dataset...")
    df = clean_dataset(df)
    
    print("\nDataset Shape:", df.shape)
    
    # Run analyses
    #analyze_correlations(df)
    #analyze_feature_groups(df)
    #analyze_distributions(df)
    #analyze_relationships(df)
    #analyze_rating_diff_impact(df)
    #analyze_prediction_confidence(df)
    #analyze_player_variance(df)
    #analyze_round_distributions(df)
    analyze_map_performance(df)

if __name__ == "__main__":
    main() 