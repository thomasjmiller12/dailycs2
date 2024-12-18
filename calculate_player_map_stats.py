from db.db import get_player_data
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class PlayerMapStats:
    def __init__(self, player_name: str, min_maps_threshold: int = 5):
        """
        Initialize PlayerMapStats with player data and analysis settings.
        
        Args:
            player_name: Name of the player to analyze
            min_maps_threshold: Minimum number of maps required for reliable statistics
        """
        self.player_name = player_name
        self.min_maps_threshold = min_maps_threshold
        self._df = get_player_data(player_name)
        
        # Calculate and cache basic stats
        self._calculate_basic_stats()

    def _calculate_basic_stats(self) -> None:
        """Calculate and store basic statistics about the player."""
        self.total_maps = len(self._df)
        self.overall_avg_kills = self._df['kills'].mean()
        self.overall_std_kills = self._df['kills'].std()
        
        # Calculate per-map statistics
        self.map_stats = (self._df.groupby('map_name')
                         .agg({
                             'kills': ['count', 'mean', 'std', 'min', 'max'],
                             'deaths': ['mean']
                         }))
        self.map_stats.columns = ['maps_played', 'avg_kills', 'std_kills', 'min_kills', 'max_kills', 'avg_deaths']
        
        # Filter maps based on minimum threshold
        self.reliable_maps = self.map_stats[self.map_stats['maps_played'] >= self.min_maps_threshold]

    def get_overall_stats(self) -> Dict:
        """Get overall statistics for the player."""
        return {
            'player_name': self.player_name,
            'total_maps': self.total_maps,
            'overall_avg_kills': round(self.overall_avg_kills, 2),
            'overall_std_kills': round(self.overall_std_kills, 2)
        }

    def get_map_stats(self, map_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get statistics for specific map or all maps.
        
        Args:
            map_name: Optional specific map name to get stats for
            
        Returns:
            DataFrame with map statistics
        """
        if map_name:
            if map_name not in self.map_stats.index:
                print(self.map_stats.index)
                raise ValueError(f"No data found for map: {map_name}")
            return self.map_stats.loc[[map_name]]
        return self.map_stats

    def get_best_and_worst_maps(self) -> Tuple[pd.Series, pd.Series]:
        """Get the player's best and worst maps (based on reliable data only)."""
        if len(self.reliable_maps) == 0:
            raise ValueError(f"No maps with {self.min_maps_threshold}+ games played")
            
        best_map = self.reliable_maps.loc[self.reliable_maps['avg_kills'].idxmax()]
        worst_map = self.reliable_maps.loc[self.reliable_maps['avg_kills'].idxmin()]
        
        return best_map, worst_map

    def analyze_kill_threshold(self, map_name: str, kill_threshold: float) -> Dict:
        """
        Analyze how often the player gets more or less kills than a threshold on a specific map.
        
        Args:
            map_name: Name of the map to analyze
            kill_threshold: Number of kills to use as threshold
            
        Returns:
            Dictionary with analysis results
        """
        if map_name not in self.map_stats.index:
            raise ValueError(f"No data found for map: {map_name}")
            
        map_data = self._df[self._df['map_name'] == map_name]
        total_maps = len(map_data)
        maps_above = len(map_data[map_data['kills'] > kill_threshold])
        maps_below = len(map_data[map_data['kills'] < kill_threshold])
        maps_equal = len(map_data[map_data['kills'] == kill_threshold])
        
        return {
            'map_name': map_name,
            'kill_threshold': kill_threshold,
            'total_maps': total_maps,
            'maps_above': maps_above,
            'maps_below': maps_below,
            'maps_equal': maps_equal,
            'pct_above': round(maps_above / total_maps * 100, 1),
            'pct_below': round(maps_below / total_maps * 100, 1),
            'pct_equal': round(maps_equal / total_maps * 100, 1)
        }

def main():
    """Example usage of the PlayerMapStats class."""
    # Example player analysis
    player_stats = PlayerMapStats("DemQQ", min_maps_threshold=5)
    
    # Print overall stats
    print("\nOverall Stats:")
    print(player_stats.get_overall_stats())
    
    # Print best and worst maps
    try:
        best_map, worst_map = player_stats.get_best_and_worst_maps()
        print(f"\nBest Map: {best_map.name}")
        print(f"Avg Kills: {best_map['avg_kills']:.2f} (±{best_map['std_kills']:.2f})")
        print(f"Maps Played: {best_map['maps_played']}")
        
        print(f"\nWorst Map: {worst_map.name}")
        print(f"Avg Kills: {worst_map['avg_kills']:.2f} (±{worst_map['std_kills']:.2f})")
        print(f"Maps Played: {worst_map['maps_played']}")
    except ValueError as e:
        print(f"\nCouldn't determine best/worst maps: {e}")
    
    # Example threshold analysis for a specific map
    try:
        best_map_name = best_map.name
        threshold_analysis = player_stats.analyze_kill_threshold(best_map_name, 20)
        print(f"\nThreshold Analysis for {best_map_name}:")
        print(f"Games with > 20 kills: {threshold_analysis['pct_above']}%")
        print(f"Games with < 20 kills: {threshold_analysis['pct_below']}%")
        print(f"Games with exactly 20 kills: {threshold_analysis['pct_equal']}%")
    except (ValueError, UnboundLocalError) as e:
        print(f"\nCouldn't perform threshold analysis: {e}")

if __name__ == "__main__":
    main()
