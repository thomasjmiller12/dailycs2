from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, JSON, UniqueConstraint  # Add JSON import and UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime  # Add datetime import

Base = declarative_base()

class Match(Base):
    __tablename__ = 'matches'

    id = Column(Integer, primary_key=True)
    url = Column(String)
    time = Column(String)
    datetime = Column(DateTime)  # New column for datetime
    maps = relationship("Map", back_populates="match")

class Map(Base):
    __tablename__ = 'maps'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    score = Column(String)
    total_rounds = Column(Integer)
    round = Column(Integer)  # New column for round number
    match_id = Column(Integer, ForeignKey('matches.id'))
    match = relationship("Match", back_populates="maps")
    map_teams = relationship("MapTeam", back_populates="map")

class MapTeam(Base):
    __tablename__ = 'map_teams'

    id = Column(Integer, primary_key=True)
    team_name = Column(String)
    map_id = Column(Integer, ForeignKey('maps.id'))
    score = Column(Integer)
    map = relationship("Map", back_populates="map_teams")
    players = relationship("Player", back_populates="map_team")

class Player(Base):
    __tablename__ = 'players'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    kills = Column(Integer)
    deaths = Column(Integer)
    map_team_id = Column(Integer, ForeignKey('map_teams.id'))
    map_team = relationship("MapTeam", back_populates="players")

class CS2Projection(Base):
    __tablename__ = 'cs2_projections'

    id = Column(Integer, primary_key=True)
    projection_id = Column(Integer)
    player_name = Column(String)  # Renamed from name
    player_display_name = Column(String)  # New field
    league = Column(String)
    team = Column(String)
    position = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    stat_type = Column(String)
    line_score = Column(Float)
    odds_type = Column(String)
    projection_type = Column(String)
    rank = Column(Integer)
    status = Column(String)
    game_id = Column(String)  # Changed to String
    board_time = Column(DateTime)  # New field
    combo = Column(String)  # New field
    timestamp = Column(DateTime, default=datetime.utcnow)  # New field to track when the line was recorded
