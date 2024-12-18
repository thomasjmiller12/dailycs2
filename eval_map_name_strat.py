from db.db import get_map3_kill_projections, get_player_data, get_map3_matches_on_date
from calculate_player_map_stats import PlayerMapStats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, List, Dict
from thefuzz import fuzz
import os
from tqdm import tqdm
import pickle
import hashlib

def ensure_graphs_dir():
    """Create strategy_graphs directory if it doesn't exist."""
    graphs_dir = "strategy_graphs"
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)
    return graphs_dir

def find_matching_games(verbose: bool = False):
    """
    Find all games where we have both projection and result data.
    Uses fuzzy matching for team names.
    
    Args:
        verbose: Whether to print detailed progress information
    
    Returns:
        List of dicts containing matched projection and result data
    """
    # Get all projections
    all_projections = get_map3_kill_projections()
    if len(all_projections) == 0:
        print("No projections found!")
        return []
        
    # Convert start_time to date for easier matching
    all_projections['date'] = all_projections['start_time'].dt.date
    
    if verbose:
        print(f"\nFound {len(all_projections)} total projections across {len(all_projections['date'].unique())} dates")
    
    matched_games = []
    
    # Process each date
    for date in tqdm(all_projections['date'].unique(), desc="Processing dates"):
        if verbose:
            print(f"\nProcessing {date}:")
        
        # Get projections for this date
        date_projections = all_projections[all_projections['date'] == date]
        if verbose:
            print(f"  Found {len(date_projections)} projections for this date")
        
        # Get matches for this date
        matches = get_map3_matches_on_date(date)
        if len(matches) == 0:
            continue
            
        if verbose:
            print(f"  Found {len(matches)} Map 3 matches")
        
        # Group matches by game
        for match_time, match_group in matches.groupby('datetime'):
            teams_in_match = match_group['team_name'].unique()
            map_name = match_group['map_name'].iloc[0]
            
            # For each player in the match
            for _, player_data in match_group.iterrows():
                player_name = player_data['player_name']
                team_name = player_data['team_name']
                
                # Look for matching projection
                for _, proj in date_projections.iterrows():
                    # Check if player names match exactly
                    if proj['player_name'] != player_name:
                        continue
                        
                    # Use fuzzy matching for team names
                    team_match_ratio = fuzz.ratio(proj['team'].lower(), team_name.lower())
                    if team_match_ratio < 50:  # Threshold for fuzzy matching
                        continue
                    
                    # We found a match! Record all relevant data
                    matched_game = {
                        'date': date,
                        'match_time': match_time,
                        'player_name': player_name,
                        'team_name': team_name,
                        'map_name': map_name,
                        'actual_kills': player_data['kills'],
                        'line': proj['line_score'],
                        'projection_team': proj['team'],
                        'team_match_ratio': team_match_ratio
                    }
                    matched_games.append(matched_game)
                    if verbose:
                        print(f"    Matched: {player_name} ({team_name} / {proj['team']}) - {team_match_ratio}% match")
    
    if verbose:
        print(f"\nFound {len(matched_games)} total matched games")
    return matched_games

def initialize_player_stats(matched_games: List[dict], use_cache: bool = True, cache_dir: str = "strategy_cache") -> Dict[str, PlayerMapStats]:
    """
    Pre-initialize PlayerMapStats objects for all players in matched_games.
    Can load/save from cache to speed up repeated runs.
    
    Args:
        matched_games: List of matched projection and result data
        use_cache: Whether to try loading from and saving to cache
        cache_dir: Directory to store cache files
        
    Returns:
        Dict mapping player names to their PlayerMapStats objects
    """
    # Create cache directory if it doesn't exist
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a cache key based on the matched games data
    # We'll use the latest date in matched_games as part of the key
    latest_date = max(game['date'] for game in matched_games)
    player_names = sorted(set(game['player_name'] for game in matched_games))
    cache_key = f"player_stats_{latest_date}_{len(player_names)}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        try:
            print(f"Loading player stats from cache ({cache_file})...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cached_date = cached_data.get('date')
                cached_players = cached_data.get('players')
                
                # Only use cache if it's from today (to ensure fresh data)
                if cached_date == date.today():
                    print("Using cached player stats")
                    return cached_players
                else:
                    print("Cache is from a different day, regenerating...")
        except Exception as e:
            print(f"Error loading cache: {str(e)}, regenerating...")
    
    # Initialize stats for each player
    player_stats = {}
    for player_name in tqdm(player_names, desc="Initializing player stats"):
        try:
            player_stats[player_name] = PlayerMapStats(player_name, 1)
        except Exception as e:
            print(f"Error initializing stats for {player_name}: {str(e)}")
            continue
    
    # Save to cache if enabled
    if use_cache:
        try:
            cache_data = {
                'date': date.today(),
                'players': player_stats
            }
            print(f"Saving player stats to cache ({cache_file})...")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
    
    return player_stats

class BaseMapStrategy:
    def __init__(self):
        """Base class for map betting strategies."""
        self.stats = {
            'total_checked': 0,
            'no_map_data': 0,
            'insufficient_maps': 0,
            'no_bet_criteria': 0,
            'bets_placed': 0
        }
        self.name = self.__class__.__name__
        self.verbose = False
        self.player_stats = None  # Will be set in evaluate_matches
    
    def evaluate_matches(self, matched_games: List[dict], player_stats: Dict[str, PlayerMapStats], verbose: bool = False) -> dict:
        """
        Evaluate the betting strategy on pre-matched games.
        
        Args:
            matched_games: List of dicts containing matched projection and result data
            player_stats: Dict mapping player names to their PlayerMapStats objects
            verbose: Whether to print detailed progress information
            
        Returns:
            Dict containing evaluation metrics
        """
        self.verbose = verbose
        self.player_stats = player_stats  # Store player_stats for use in analyze_game
        # Reset stats
        self.stats = {key: 0 for key in self.stats}
        bets = []
        
        # Track over/under separately
        over_bets = {'total': 0, 'correct': 0}
        under_bets = {'total': 0, 'correct': 0}
        
        for game in tqdm(matched_games, desc=f"Evaluating {self.name}"):
            self.stats['total_checked'] += 1
            
            # Analyze using strategy-specific logic
            analysis = self.analyze_game(game)
            if analysis is None:
                continue
                
            bet_decision, stats = analysis
                
            # Record bet details
            was_correct = (
                (bet_decision == "OVER" and game['actual_kills'] > game['line']) or 
                (bet_decision == "UNDER" and game['actual_kills'] < game['line'])
            )
            
            # Track over/under stats
            if bet_decision == "OVER":
                over_bets['total'] += 1
                if was_correct:
                    over_bets['correct'] += 1
            else:  # UNDER
                under_bets['total'] += 1
                if was_correct:
                    under_bets['correct'] += 1
            
            bet_info = {
                'date': game['date'],
                'player_name': game['player_name'],
                'map_name': game['map_name'],
                'line': game['line'],
                'actual_kills': game['actual_kills'],
                'bet': bet_decision,
                'correct': was_correct,
                'team_match_ratio': game['team_match_ratio'],
                **stats  # Include all strategy-specific stats
            }
            bets.append(bet_info)
            
            if verbose:
                self.print_bet_analysis(bet_info)
        
        # Calculate metrics
        total_bets = len(bets)
        if total_bets == 0:
            return {
                'total_bets': 0, 
                'success_rate': 0, 
                'total_correct': 0, 
                'bets': [], 
                'stats': self.stats,
                'over_stats': {'total': 0, 'correct': 0, 'rate': 0},
                'under_stats': {'total': 0, 'correct': 0, 'rate': 0}
            }
            
        correct_bets = sum(1 for bet in bets if bet['correct'])
        success_rate = (correct_bets / total_bets) * 100
        
        # Calculate over/under rates
        over_rate = (over_bets['correct'] / over_bets['total'] * 100) if over_bets['total'] > 0 else 0
        under_rate = (under_bets['correct'] / under_bets['total'] * 100) if under_bets['total'] > 0 else 0
        
        over_stats = {**over_bets, 'rate': round(over_rate, 2)}
        under_stats = {**under_bets, 'rate': round(under_rate, 2)}
        
        return {
            'total_bets': total_bets,
            'success_rate': round(success_rate, 2),
            'total_correct': correct_bets,
            'bets': bets,
            'stats': self.stats,
            'over_stats': over_stats,
            'under_stats': under_stats
        }
    
    def analyze_game(self, game: dict) -> Optional[Tuple[str, dict]]:
        """To be implemented by child classes."""
        raise NotImplementedError
        
    def print_bet_analysis(self, bet_info: dict):
        """To be implemented by child classes."""
        raise NotImplementedError
    
    def plot_performance(self, results: dict):
        """Create standard performance visualization for any strategy."""
        graphs_dir = ensure_graphs_dir()
        plt.style.use('default')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Success Rates
        rates = [
            results['success_rate'],
            results['over_stats']['rate'],
            results['under_stats']['rate']
        ]
        labels = ['Overall', 'Overs', 'Unders']
        colors = ['blue', 'green', 'red']
        
        bars = ax1.bar(labels, rates, alpha=0.7, color=colors)
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rates')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Plot 2: Bet Distribution
        bet_counts = [
            results['over_stats']['total'],
            results['under_stats']['total']
        ]
        
        pie = ax2.pie(bet_counts, labels=['Overs', 'Unders'], autopct='%1.1f%%',
                     colors=['lightgreen', 'lightcoral'])
        ax2.set_title('Bet Distribution')
        
        # Plot 3: Decision Flow
        decision_stats = [
            results['stats']['total_checked'],
            results['stats']['no_map_data'],
            results['stats']['insufficient_maps'],
            results['stats']['no_bet_criteria'],
            results['stats']['bets_placed']
        ]
        decision_labels = [
            'Total Checked',
            'No Map Data',
            'Insufficient Maps',
            'No Bet Criteria',
            'Bets Placed'
        ]
        
        bars = ax3.bar(decision_labels, decision_stats, alpha=0.7)
        ax3.set_ylabel('Count')
        ax3.set_title('Decision Flow')
        plt.xticks(rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{self.name}_performance.png"))
        plt.close()
    
    def plot_strategy_specific(self, results: dict):
        """To be implemented by child classes for strategy-specific visualizations."""
        pass

class MapVarianceStrategy(BaseMapStrategy):
    def __init__(self, kill_difference_threshold: float = 3.0, min_maps_threshold: int = 5):
        """
        Strategy based on difference between map average and overall average.
        
        Args:
            kill_difference_threshold: How many kills +/- from average to trigger a bet
            min_maps_threshold: Minimum number of maps needed on the specific map
        """
        super().__init__()
        self.kill_difference_threshold = kill_difference_threshold
        self.min_maps_threshold = min_maps_threshold
        
    def analyze_game(self, game: dict) -> Optional[Tuple[str, dict]]:
        """Analyze using map variance approach."""
        try:
            player_name = game['player_name']
            if player_name not in self.player_stats:
                self.stats['no_map_data'] += 1
                return None
            
            player_stats = self.player_stats[player_name]
            
            # Get all data at once
            df = player_stats._df
            if len(df) == 0:
                self.stats['no_map_data'] += 1
                return None
            
            # Calculate map-specific stats using vectorized operations
            map_mask = df['map_name'] == game['map_name']
            map_count = map_mask.sum()
            
            if map_count == 0:
                self.stats['no_map_data'] += 1
                return None
                
            # Check if we have enough data for this specific map
            if map_count < self.min_maps_threshold:
                self.stats['insufficient_maps'] += 1
                return None
            
            # Calculate averages using vectorized operations
            map_avg = df[map_mask]['kills'].mean()
            overall_avg = df['kills'].mean()
                
            # Calculate difference from average
            diff = map_avg - overall_avg
            
            # Determine if we should bet
            if abs(diff) < self.kill_difference_threshold:
                self.stats['no_bet_criteria'] += 1
                return None
                
            self.stats['bets_placed'] += 1
            stats_dict = {
                'overall_avg': overall_avg,
                'map_avg': map_avg,
                'maps_played': map_count,
                'total_maps': len(df),
                'diff_from_avg': diff
            }
            return ("OVER" if diff > 0 else "UNDER", stats_dict)
            
        except Exception as e:
            print(f"Error analyzing {game['player_name']} on {game['map_name']}: {str(e)}")
            return None
            
    def print_bet_analysis(self, bet_info: dict):
        """Print detailed analysis for variance-based strategy."""
        print(f"\n  {bet_info['player_name']} on {bet_info['map_name']} ({bet_info['maps_played']}/{bet_info['total_maps']} maps):")
        print(f"    Overall Avg: {bet_info['overall_avg']:.2f}")
        print(f"    {bet_info['map_name']} Avg: {bet_info['map_avg']:.2f} ({bet_info['diff_from_avg']:+.2f} vs overall)")
        print(f"    Line: {bet_info['line']}, Actual: {bet_info['actual_kills']}")
        print(f"    Bet {bet_info['bet']}: {'✓' if bet_info['correct'] else '✗'}")
    
    def plot_strategy_specific(self, results: dict):
        """Create visualization specific to variance-based strategy."""
        graphs_dir = ensure_graphs_dir()
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distribution of Map vs Overall Differences
        diffs = [bet['diff_from_avg'] for bet in results['bets']]
        correct = [bet['correct'] for bet in results['bets']]
        
        ax1.hist([
            [d for d, c in zip(diffs, correct) if c],  # Correct bets
            [d for d, c in zip(diffs, correct) if not c]  # Incorrect bets
        ], label=['Correct', 'Incorrect'], bins=10, alpha=0.7)
        ax1.set_xlabel('Difference from Average')
        ax1.set_ylabel('Number of Bets')
        ax1.set_title('Distribution of Map vs Overall Differences')
        ax1.legend()
        
        # Plot 2: Success Rate by Difference Magnitude
        diff_bins = pd.cut(abs(pd.Series(diffs)), bins=5)
        success_by_diff = pd.Series(correct).groupby(diff_bins).mean() * 100
        
        ax2.plot(range(len(success_by_diff)), success_by_diff.values, marker='o')
        ax2.set_xlabel('Difference Magnitude Bins')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate by Difference Magnitude')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{self.name}_specific.png"))
        plt.close()

class ThresholdStrategy(BaseMapStrategy):
    def __init__(self, 
                 over_overall_threshold: float = 0.66,
                 over_map_threshold: float = 0.55,
                 under_overall_threshold: float = 0.66,
                 under_map_threshold: float = 0.55,
                 min_maps_threshold: int = 5):
        """
        Strategy based on percentage of games over/under the line.
        
        Args:
            over_overall_threshold: Percentage threshold for overall games over (e.g., 0.66 for 66%)
            over_map_threshold: Percentage threshold for map-specific games over
            under_overall_threshold: Percentage threshold for overall games under
            under_map_threshold: Percentage threshold for map-specific games under
            min_maps_threshold: Minimum number of maps needed on the specific map
        """
        super().__init__()
        self.over_overall_threshold = over_overall_threshold
        self.over_map_threshold = over_map_threshold
        self.under_overall_threshold = under_overall_threshold
        self.under_map_threshold = under_map_threshold
        self.min_maps_threshold = min_maps_threshold
        
    def analyze_game(self, game: dict) -> Optional[Tuple[str, dict]]:
        """Analyze using threshold approach."""
        try:
            player_name = game['player_name']
            if player_name not in self.player_stats:
                self.stats['no_map_data'] += 1
                return None
            
            player_stats = self.player_stats[player_name]
            line = game['line']
            
            # Get all data at once and compute kills comparison
            df = player_stats._df
            if len(df) == 0:
                self.stats['no_map_data'] += 1
                return None
                
            # Pre-compute kills comparison for all games
            kills_over_line = df['kills'] > line
            kills_under_line = df['kills'] < line
            
            # Get map-specific data using boolean indexing
            map_mask = df['map_name'] == game['map_name']
            map_count = map_mask.sum()
            
            # Check if we have enough data for this specific map
            if map_count < self.min_maps_threshold:
                self.stats['insufficient_maps'] += 1
                return None
            
            # Calculate percentages using pre-computed values
            total_games = len(df)
            overall_over_pct = kills_over_line.mean()
            overall_under_pct = kills_under_line.mean()
            
            # Calculate map-specific percentages using boolean indexing
            map_over_pct = kills_over_line[map_mask].mean()
            map_under_pct = kills_under_line[map_mask].mean()
            
            # Track which criteria are met
            overall_over_met = overall_over_pct >= self.over_overall_threshold
            overall_under_met = overall_under_pct >= self.under_overall_threshold
            map_over_met = map_over_pct >= self.over_map_threshold
            map_under_met = map_under_pct >= self.under_map_threshold
            
            # Determine bet based on thresholds
            bet_decision = None
            bet_source = []
            
            # Check over criteria
            if overall_over_met or map_over_met:
                bet_decision = "OVER"
                if overall_over_met and map_over_met:
                    bet_source = ["both"]
                elif overall_over_met:
                    bet_source = ["overall"]
                else:
                    bet_source = ["map"]
                    
            # Check under criteria if no over bet was made
            if bet_decision is None and (overall_under_met or map_under_met):
                bet_decision = "UNDER"
                if overall_under_met and map_under_met:
                    bet_source = ["both"]
                elif overall_under_met:
                    bet_source = ["overall"]
                else:
                    bet_source = ["map"]
                
            if bet_decision is None:
                self.stats['no_bet_criteria'] += 1
                return None
                
            self.stats['bets_placed'] += 1
            stats_dict = {
                'total_games': total_games,
                'map_games': map_count,
                'overall_over_pct': overall_over_pct,
                'overall_under_pct': overall_under_pct,
                'map_over_pct': map_over_pct,
                'map_under_pct': map_under_pct,
                'bet_source': bet_source[0],  # Add source of the bet decision
                'overall_criteria_met': overall_over_met or overall_under_met,
                'map_criteria_met': map_over_met or map_under_met
            }
            return (bet_decision, stats_dict)
            
        except Exception as e:
            print(f"Error analyzing {game['player_name']} on {game['map_name']}: {str(e)}")
            return None
            
    def print_bet_analysis(self, bet_info: dict):
        """Print detailed analysis for threshold-based strategy."""
        print(f"\n  {bet_info['player_name']} on {bet_info['map_name']} ({bet_info['map_games']}/{bet_info['total_games']} maps):")
        print(f"    Overall: Over {bet_info['overall_over_pct']:.1%}, Under {bet_info['overall_under_pct']:.1%}")
        print(f"    On {bet_info['map_name']}: Over {bet_info['map_over_pct']:.1%}, Under {bet_info['map_under_pct']:.1%}")
        print(f"    Line: {bet_info['line']}, Actual: {bet_info['actual_kills']}")
        print(f"    Bet {bet_info['bet']} (Source: {bet_info['bet_source']}): {'✓' if bet_info['correct'] else '✗'}")
    
    def evaluate_matches(self, matched_games: List[dict], player_stats: dict, verbose: bool = False) -> dict:
        """Override evaluate_matches to add source tracking."""
        results = super().evaluate_matches(matched_games, player_stats, verbose)
        
        if results['total_bets'] > 0:
            # Track success rates by source
            source_stats = {
                'overall': {'total': 0, 'correct': 0, 'rate': 0},
                'map': {'total': 0, 'correct': 0, 'rate': 0},
                'both': {'total': 0, 'correct': 0, 'rate': 0}
            }
            
            for bet in results['bets']:
                source = bet['bet_source']
                source_stats[source]['total'] += 1
                if bet['correct']:
                    source_stats[source]['correct'] += 1
            
            # Calculate rates
            for stats in source_stats.values():
                if stats['total'] > 0:
                    stats['rate'] = round(stats['correct'] / stats['total'] * 100, 2)
            
            results['source_stats'] = source_stats
        
        return results
    
    def plot_strategy_specific(self, results: dict):
        """Create visualization specific to threshold-based strategy."""
        graphs_dir = ensure_graphs_dir()
        plt.style.use('default')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot 1: Success Rate by Threshold Level
        over_pcts = [bet['map_over_pct'] for bet in results['bets'] if bet['bet'] == 'OVER']
        under_pcts = [bet['map_under_pct'] for bet in results['bets'] if bet['bet'] == 'UNDER']
        over_correct = [bet['correct'] for bet in results['bets'] if bet['bet'] == 'OVER']
        under_correct = [bet['correct'] for bet in results['bets'] if bet['bet'] == 'UNDER']
        
        if over_pcts:
            # Create bins and get bin edges for x-axis labels
            over_bins = pd.cut(pd.Series(over_pcts), bins=5)
            over_success = pd.Series(over_correct).groupby(over_bins).mean() * 100
            
            # Get bin ranges for x-axis
            bin_ranges = [f"{interval.left:.1%}-{interval.right:.1%}" 
                         for interval in over_success.index]
            
            ax1.plot(range(len(over_success)), over_success.values, 
                    marker='o', label='Overs', color='green')
            
        if under_pcts:
            under_bins = pd.cut(pd.Series(under_pcts), bins=5)
            under_success = pd.Series(under_correct).groupby(under_bins).mean() * 100
            
            # Use the same bin ranges if we don't have overs
            if not over_pcts:
                bin_ranges = [f"{interval.left:.1%}-{interval.right:.1%}" 
                            for interval in under_success.index]
            
            ax1.plot(range(len(under_success)), under_success.values, 
                    marker='o', label='Unders', color='red')
        
        if over_pcts or under_pcts:
            ax1.set_xticks(range(len(bin_ranges)))
            ax1.set_xticklabels(bin_ranges, rotation=45, ha='right')
            
        ax1.set_xlabel('Historical Win Rate Range')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate by Historical Win Rate')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Bet Distribution by Map (unchanged)
        map_bets = pd.DataFrame(results['bets']).groupby('map_name').agg({
            'bet': 'count',
            'correct': 'mean'
        }).sort_values('bet', ascending=False)
        
        ax2.bar(range(len(map_bets)), map_bets['bet'], alpha=0.7)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(range(len(map_bets)), map_bets['correct'] * 100, 
                     color='red', marker='o')
        
        ax2.set_xlabel('Maps')
        ax2.set_ylabel('Number of Bets')
        ax2_twin.set_ylabel('Success Rate (%)', color='red')
        ax2.set_title('Bets and Success Rate by Map')
        ax2.set_xticks(range(len(map_bets)))
        ax2.set_xticklabels(map_bets.index, rotation=45, ha='right')
        
        # Plot 3: Success Rate by Bet Source (new)
        if 'source_stats' in results:
            sources = list(results['source_stats'].keys())
            rates = [results['source_stats'][s]['rate'] for s in sources]
            counts = [results['source_stats'][s]['total'] for s in sources]
            
            bars = ax3.bar(sources, rates, alpha=0.7)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Success Rate by Bet Source')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%\n(n={count})',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"{self.name}_specific.png"))
        plt.close()

def analyze_overall_projections(matched_games: List[dict]) -> dict:
    """
    Analyze all Map 3 projections to see how often overs and unders hit.
    
    Args:
        matched_games: List of matched projection and result data
        
    Returns:
        Dict containing overall analysis metrics
    """
    total_games = len(matched_games)
    if total_games == 0:
        return {
            'total_games': 0,
            'over_hits': 0,
            'under_hits': 0,
            'over_rate': 0,
            'under_rate': 0,
            'by_map': {}
        }
    
    over_hits = sum(1 for game in matched_games if game['actual_kills'] > game['line'])
    under_hits = sum(1 for game in matched_games if game['actual_kills'] < game['line'])
    pushes = total_games - over_hits - under_hits
    
    # Calculate by map
    map_stats = {}
    for game in matched_games:
        map_name = game['map_name']
        if map_name not in map_stats:
            map_stats[map_name] = {'total': 0, 'over_hits': 0, 'under_hits': 0, 'pushes': 0}
        
        map_stats[map_name]['total'] += 1
        if game['actual_kills'] > game['line']:
            map_stats[map_name]['over_hits'] += 1
        elif game['actual_kills'] < game['line']:
            map_stats[map_name]['under_hits'] += 1
        else:
            map_stats[map_name]['pushes'] += 1
    
    # Calculate percentages for each map
    for map_data in map_stats.values():
        total = map_data['total']
        map_data['over_rate'] = (map_data['over_hits'] / total * 100) if total > 0 else 0
        map_data['under_rate'] = (map_data['under_hits'] / total * 100) if total > 0 else 0
        map_data['push_rate'] = (map_data['pushes'] / total * 100) if total > 0 else 0
    
    return {
        'total_games': total_games,
        'over_hits': over_hits,
        'under_hits': under_hits,
        'pushes': pushes,
        'over_rate': over_hits / total_games * 100,
        'under_rate': under_hits / total_games * 100,
        'push_rate': pushes / total_games * 100,
        'by_map': map_stats
    }

def print_overall_analysis(analysis: dict):
    """Print the overall analysis results in a formatted way."""
    print("\n=== Overall Map 3 Projection Analysis ===")
    print(f"Total Games: {analysis['total_games']}")
    print(f"Over Hits: {analysis['over_hits']} ({analysis['over_rate']:.1f}%)")
    print(f"Under Hits: {analysis['under_hits']} ({analysis['under_rate']:.1f}%)")
    print(f"Pushes: {analysis['pushes']} ({analysis['push_rate']:.1f}%)")
    
    print("\nBreakdown by Map:")
    print("-" * 60)
    print(f"{'Map Name':<12} {'Total':<8} {'Over %':<10} {'Under %':<10} {'Push %':<10}")
    print("-" * 60)
    
    # Sort maps by total games
    sorted_maps = sorted(analysis['by_map'].items(), 
                        key=lambda x: x[1]['total'], 
                        reverse=True)
    
    for map_name, stats in sorted_maps:
        print(f"{map_name:<12} {stats['total']:<8} "
              f"{stats['over_rate']:>7.1f}%  "
              f"{stats['under_rate']:>7.1f}%  "
              f"{stats['push_rate']:>7.1f}%")

def main(strategies: str = "both", use_cache: bool = True):
    """
    Run strategy evaluations based on input parameter.
    
    Args:
        strategies: Which strategies to run. Options: "variance", "threshold", or "both"
        use_cache: Whether to use cached player stats between runs
    """
    # Create graphs directory
    ensure_graphs_dir()
    
    verbose = False  # Set to True for detailed logging
    
    matched_games = find_matching_games(verbose=verbose)
    
    if not matched_games:
        print("No matches found to evaluate!")
        return
    
    # Pre-initialize player stats with caching
    print("\nPre-initializing player stats...")
    player_stats = initialize_player_stats(matched_games, use_cache=use_cache)
    print(f"Initialized stats for {len(player_stats)} players")
    
    # Add overall analysis
    overall_analysis = analyze_overall_projections(matched_games)
    print_overall_analysis(overall_analysis)
    
    # Initialize results
    variance_results = None
    threshold_results = None
        
    # Test Variance Strategy
    if strategies.lower() in ["variance", "both"]:
        variance_strategy = MapVarianceStrategy(
            kill_difference_threshold=0.5,
            min_maps_threshold=5
        )
        variance_results = variance_strategy.evaluate_matches(matched_games, player_stats, verbose=verbose)
        
        print("\n=== Variance Strategy Results ===")
        print(f"Total Bets Placed: {variance_results['total_bets']}")
        print(f"Overall Success Rate: {variance_results['success_rate']}%")
        print(f"Over Success Rate: {variance_results['over_stats']['rate']}% ({variance_results['over_stats']['correct']}/{variance_results['over_stats']['total']})")
        print(f"Under Success Rate: {variance_results['under_stats']['rate']}% ({variance_results['under_stats']['correct']}/{variance_results['under_stats']['total']})")
        
        # Print detailed stats
        stats = variance_results['stats']
        print(f"\nVariance Strategy Analysis Stats:")
        print(f"Total Games Checked: {stats['total_checked']}")
        print(f"No Map Data: {stats['no_map_data']}")
        print(f"Insufficient Maps: {stats['insufficient_maps']}")
        print(f"No Bet Criteria Met: {stats['no_bet_criteria']}")
        print(f"Bets Placed: {stats['bets_placed']}")
        
        # Generate plots
        print("\nGenerating Variance Strategy plots...")
        variance_strategy.plot_performance(variance_results)
        variance_strategy.plot_strategy_specific(variance_results)
    
    # Test Threshold Strategy
    if strategies.lower() in ["threshold", "both"]:
        threshold_strategy = ThresholdStrategy(
            over_overall_threshold=0.65,
            over_map_threshold=0.60,
            under_overall_threshold=0.70,  # More strict for unders
            under_map_threshold=0.70,      # More strict for unders
            min_maps_threshold=5
        )
        threshold_results = threshold_strategy.evaluate_matches(matched_games, player_stats, verbose=verbose)
        
        print("\n=== Threshold Strategy Results ===")
        print(f"Total Bets Placed: {threshold_results['total_bets']}")
        print(f"Overall Success Rate: {threshold_results['success_rate']}%")
        print(f"Over Success Rate: {threshold_results['over_stats']['rate']}% ({threshold_results['over_stats']['correct']}/{threshold_results['over_stats']['total']})")
        print(f"Under Success Rate: {threshold_results['under_stats']['rate']}% ({threshold_results['under_stats']['correct']}/{threshold_results['under_stats']['total']})")
        
        # Print source stats
        if 'source_stats' in threshold_results:
            print("\nThreshold Strategy Source Analysis:")
            for source, stats in threshold_results['source_stats'].items():
                print(f"{source.title()} Source: {stats['rate']}% ({stats['correct']}/{stats['total']})")
        
        # Print detailed stats
        stats = threshold_results['stats']
        print(f"\nThreshold Strategy Analysis Stats:")
        print(f"Total Games Checked: {stats['total_checked']}")
        print(f"No Map Data: {stats['no_map_data']}")
        print(f"Insufficient Maps: {stats['insufficient_maps']}")
        print(f"No Bet Criteria Met: {stats['no_bet_criteria']}")
        print(f"Bets Placed: {stats['bets_placed']}")
        
        # Generate plots
        print("\nGenerating Threshold Strategy plots...")
        threshold_strategy.plot_performance(threshold_results)
        threshold_strategy.plot_strategy_specific(threshold_results)
    
    print("\nPlots saved to strategy_graphs directory")
    
    return variance_results, threshold_results

if __name__ == "__main__":
    import sys
    
    # Get strategy from command line argument, default to "both"
    strategy = "both"
    use_cache = True
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1].lower()
        if strategy not in ["variance", "threshold", "both"]:
            print("Invalid strategy. Please use 'variance', 'threshold', or 'both'")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        use_cache = sys.argv[2].lower() == "true"
    
    main(strategy, use_cache)
