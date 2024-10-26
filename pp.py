import requests
import pandas as pd
from sqlalchemy.orm import sessionmaker
from db.db_utils import db_connect, create_tables
from db.models import CS2Projection
from datetime import datetime
from dump_to_db import dump_df_to_db
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def call_endpoint(url, params=None, max_level=3):
    try:
        start_time = time.time()
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
        
        logger.info(f"API call completed in {time.time() - start_time:.2f} seconds")
        return resp, data
    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        raise

def get_projections():
    start_time = time.time()
    logger.info("Starting projection fetch")
    
    url = 'https://partner-api.prizepicks.com/projections'
    params = {'per_page': 10000}

    raw_resp, df = call_endpoint(url, params=params)

    logger.info(f"Fetched DataFrame size: {df.shape}")

    if df.empty:
        logger.warning("DataFrame is empty after fetching data.")
        return

    # Connect to the database
    engine, _ = db_connect()
    create_tables(engine)

    try:
        # Insert raw data into prizepicks_dump
        try:
            dump_df_to_db(df, 'prizepicks_dump', engine)
        except Exception as e:
            logger.error(f"Failed to insert prizepicks dump: {str(e)}")
            raise

        df_cs2 = df[df['attributes.league'] == 'CS2']
        logger.info(f"Filtered CS2 DataFrame size: {df_cs2.shape}")

        Session = sessionmaker(bind=engine)
        session = Session()
        
        new_entries = 0
        try:
            for _, row in df_cs2.iterrows():
                existing_projections = session.query(CS2Projection).filter_by(projection_id=row['id']).order_by(CS2Projection.timestamp.desc()).all()
                
                if existing_projections and existing_projections[0].line_score == row['attributes.line_score']:
                    continue
                
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
                    game_id=row['attributes.game_id'],
                    board_time=row['attributes.board_time'],
                    combo=row['attributes.combo'],
                    timestamp=datetime.utcnow()
                )
                session.add(projection)
                new_entries += 1
            
            session.commit()
            logger.info(f"Added {new_entries} new CS2 projections")
            
        except Exception as e:
            logger.error(f"Error processing CS2 projections: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
            
    finally:
        logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    get_projections()
    logger.info("Script completed successfully")








