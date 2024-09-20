from .db_utils import db_connect, create_session
from .models import Player, Match, Map, MapTeam  # Import necessary models
from sqlalchemy import func
from sqlalchemy.orm import joinedload
import numpy as np

def get_players(include_count=False):
    engine, _ = db_connect()
    session = create_session(engine)

    try:
        if include_count:
            # Query for unique player names and their datapoint counts
            query = session.query(
                Player.name, 
                func.count(Player.id).label('datapoint_count')
            ).group_by(Player.name)
            
            results = query.all()
            players = {name: count for name, count in results}
        else:
            # Query for unique player names only
            query = session.query(Player.name).distinct()
            results = query.all()
            players = [result[0] for result in results]

        return players
    finally:
        session.close()


def get_maps(start_date=None, end_date=None):
    """
    Retrieve map data from the database within an optional date range.

    Args:
        start_date (datetime, optional): The start date for filtering maps.
        end_date (datetime, optional): The end date for filtering maps.

    Returns:
        list: A list of dictionaries containing map data, including map_id, map_name,
              match_datetime, and teams information (team names and player stats).

    Note:
        This function uses eager loading to optimize database queries.
    """
    
    engine, _ = db_connect()
    session = create_session(engine)

    try:
        # Use joinedload to eagerly load related entities
        query = session.query(Map).options(
            joinedload(Map.match),
            joinedload(Map.map_teams).joinedload(MapTeam.players)
        )

        # Apply date filters if provided
        if start_date:
            query = query.filter(Map.match.has(Match.datetime >= start_date))
        if end_date:
            query = query.filter(Map.match.has(Match.datetime <= end_date))

        maps = query.all()
        
        result = []
        for map_instance in maps:
            map_data = {
                "map_id": map_instance.id,
                "map_name": map_instance.name,
                "match_datetime": map_instance.match.datetime.isoformat(),
                "teams": [
                    {
                        "team_name": map_team.team_name,
                        "players": [
                            {
                                "name": player.name,
                                "kills": player.kills,
                                "deaths": player.deaths
                            }
                            for player in map_team.players
                        ]
                    }
                    for map_team in map_instance.map_teams
                ]
            }
            result.append(map_data)
        
        return result
    finally:
        session.close()

def get_player_stats(player_name, expanded=False, start_date=None, end_date=None):
    """
    Retrieve statistics for a specific player from the database.

    Args:
        player_name (str): The name of the player to retrieve stats for.
        expanded (bool, optional): If True, include additional detailed statistics.
        start_date (datetime, optional): The start date for filtering player stats.
        end_date (datetime, optional): The end date for filtering player stats.

    Returns:
        dict: A dictionary containing player statistics, including average kills,
              average deaths, K/D ratio, total maps played, teams played for,
              and average rounds per map. If expanded is True, additional stats
              such as all kills, all deaths, total kills, total deaths, kill variance,
              and death variance are included.

    Note:
        Returns None if the player is not found in the database.
    """
    engine, _ = db_connect()
    session = create_session(engine)

    try:
        # Query for the player and their related data
        query = session.query(Player).filter(Player.name == player_name)

        # Apply date filters if provided
        if start_date:
            query = query.join(MapTeam).join(Map).join(Match).filter(Match.datetime >= start_date)
        if end_date:
            query = query.join(MapTeam).join(Map).join(Match).filter(Match.datetime <= end_date)

        player_data = query.all()

        if not player_data:
            return None

        total_kills = 0
        total_deaths = 0
        total_rounds = 0
        team_names = set()
        all_kills = []
        all_deaths = []
        map_count = 0

        for player in player_data:
            total_kills += player.kills
            total_deaths += player.deaths
            all_kills.append(player.kills)
            all_deaths.append(player.deaths)
            team_names.add(player.map_team.team_name)
            total_rounds += player.map_team.map.total_rounds
            map_count += 1

        avg_kills = total_kills / map_count if map_count > 0 else 0
        avg_deaths = total_deaths / map_count if map_count > 0 else 0
        avg_rounds = total_rounds / map_count if map_count > 0 else 0
        kd_ratio = total_kills / total_deaths if total_deaths > 0 else total_kills

        stats = {
            "player_name": player_name,
            "average_kills": round(avg_kills, 2),
            "average_deaths": round(avg_deaths, 2),
            "kd_ratio": round(kd_ratio, 2),
            "total_maps_played": map_count,
            "teams_played_for": list(team_names),
            "average_rounds_per_map": round(avg_rounds, 2)
        }

        if expanded:
            stats.update({
                "all_kills": all_kills,
                "all_deaths": all_deaths,
                "total_kills": total_kills,
                "total_deaths": total_deaths,
                "kill_variance": round(np.var(all_kills), 2) if len(all_kills) > 0 else 0,
                "death_variance": round(np.var(all_deaths), 2) if len(all_deaths) > 0 else 0
            })

        return stats
    finally:
        session.close()
