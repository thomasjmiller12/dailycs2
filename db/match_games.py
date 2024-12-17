from sqlalchemy import func
import pandas as pd
from datetime import timedelta
from .db_utils import db_connect, create_session
from .models import Match, Map, MapTeam, CS2Projection

def find_matching_games(time_window_hours=5):
    """
    Find matching games between CS2Projections and actual match data.
    
    Args:
        time_window_hours (int): Number of hours to consider for matching start times
        
    Returns:
        pd.DataFrame: DataFrame containing matched games with their details
    """
    engine, _ = db_connect()
    session = create_session(engine)
    
    try:
        # Get unique games from CS2Projections
        proj_games = (
            session.query(
                CS2Projection.game_id,
                CS2Projection.team,
                CS2Projection.start_time,
                CS2Projection.board_time
            )
            .distinct()
            .all()
        )
        print(f"\nFound {len(proj_games)} unique games in CS2Projections")
        if proj_games:
            print("Sample projection game:", {
                'game_id': proj_games[0].game_id,
                'team': proj_games[0].team,
                'start_time': proj_games[0].start_time
            })
        
        # Get matches with team information
        matches = (
            session.query(
                Match.id,
                Match.datetime,
                Match.url,
                func.array_agg(MapTeam.team_name).label('teams')
            )
            .join(Map, Match.id == Map.match_id)
            .join(MapTeam, Map.id == MapTeam.map_id)
            .group_by(Match.id, Match.datetime, Match.url)
            .all()
        )
        print(f"\nFound {len(matches)} matches in Match table")
        if matches:
            print("Sample match:", {
                'id': matches[0].id,
                'datetime': matches[0].datetime,
                'teams': matches[0].teams
            })
        
        # Store matched games
        matched_games = []
        
        # For each projection game
        for proj in proj_games:
            proj_time = proj.start_time
            time_lower = proj_time - timedelta(hours=time_window_hours)
            time_upper = proj_time + timedelta(hours=time_window_hours)
            
            # Find matches within time window
            for match in matches:
                if time_lower <= match.datetime <= time_upper:
                    # Check if projection team is in match teams
                    if proj.team in match.teams:
                        matched_games.append({
                            'game_id': proj.game_id,
                            'projection_team': proj.team,
                            'projection_start_time': proj.start_time,
                            'board_time': proj.board_time,
                            'match_id': match.id,
                            'match_datetime': match.datetime,
                            'match_url': match.url,
                            'match_teams': match.teams
                        })
        
        # Convert to DataFrame
        df = pd.DataFrame(matched_games)
        
        if len(df) > 0:
            print("\nMatched games sample:")
            print(df.head().to_string())
        else:
            print("\nNo matches found between the datasets")
        
        return df
    
    finally:
        session.close()

if __name__ == "__main__":
    # Test the function
    matched_games_df = find_matching_games()
    print(f"\nTotal matched games: {len(matched_games_df)}") 