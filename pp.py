import requests
import pandas as pd
from sqlalchemy.orm import sessionmaker
from db.db_utils import db_connect, create_tables
from db.models import CS2Projection
from datetime import datetime
from dump_to_db import dump_df_to_db

def call_endpoint(url, params=None, max_level=3):
    resp = requests.get(url, params=params).json()
    data = pd.json_normalize(resp['data'], max_level=max_level)
    included = pd.json_normalize(resp['included'], max_level=max_level)
    
    inc_cop = included[included['type'] == 'new_player'].copy().dropna(axis=1)
    data = pd.merge(data
                    , inc_cop
                    , how='left'
                    , left_on=['relationships.new_player.data.id'
                                ,'relationships.new_player.data.type']
                    , right_on=['id', 'type']
                    , suffixes=('', '_new_player'))
    return resp, data  # Return the raw response as well

def get_projections():
    url = 'https://partner-api.prizepicks.com/projections'
    params = {'per_page': 10000}

    raw_resp, df = call_endpoint(url, params=params)

    print(f"DataFrame size: {df.shape}")

    if df.empty:
        print("DataFrame is empty after fetching data.")
        return


    # Connect to the database
    engine, _ = db_connect()
    create_tables(engine)  # Ensure tables are created

    try:
        # Insert raw data into prizepicks_dump
        try:
            dump_df_to_db(df, 'prizepicks_dump', engine)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to insert or update prizepicks dump: {e}")

        df_cs2 = df[df['attributes.league'] == 'CS2']  # Filter for CS2 league

        print(f"Filtered DataFrame size (CS2): {df_cs2.shape}")

        Session = sessionmaker(bind=engine)
        session = Session()

        for _, row in df_cs2.iterrows():
            # Check if the projection_id already exists
            existing_projections = session.query(CS2Projection).filter_by(projection_id=row['id']).order_by(CS2Projection.timestamp.desc()).all()
            
            if existing_projections:
                # Get the most recent line_score
                most_recent_projection = existing_projections[0]
                if most_recent_projection.line_score == row['attributes.line_score']:
                    # Skip if the line_score is the same as the most recent one
                    continue
            
            # Add new projection entry
            projection = CS2Projection(
                projection_id=row['id'],
                player_name=row['attributes.name'],
                player_display_name=row['attributes.display_name'],
                league=row['attributes.league'],
                team=row['attributes.team'],
                position=row['attributes.position'],
                start_time=row['attributes.start_time'],
                end_time=row['attributes.end_time'],
                stat_type=row['attributes.stat_type'],
                line_score=row['attributes.line_score'],
                odds_type=row['attributes.odds_type'],
                projection_type=row['attributes.projection_type'],
                rank=row['attributes.rank'],
                status=row['attributes.status'],
                game_id=row['attributes.game_id'],  # No need to convert to int
                board_time=row['attributes.board_time'],
                combo=row['attributes.combo'],
                timestamp=datetime.utcnow()
            )
            session.add(projection)
        session.commit()
    finally:
        session.close()

if __name__ == "__main__":
    get_projections()
    print("Projections saved to the database")








