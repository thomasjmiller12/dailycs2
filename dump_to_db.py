import pandas as pd
from sqlalchemy import Table, MetaData, Column, Integer, String, Float, Boolean, DateTime, UniqueConstraint, text, inspect
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)

def get_existing_columns(engine: Engine, table_name: str):
    inspector = inspect(engine)
    return [column['name'] for column in inspector.get_columns(table_name)]

def create_table_if_not_exists(engine: Engine, table_name: str, df: pd.DataFrame):
    start_time = time.time()
    metadata = MetaData()
    columns = []
    for column_name, dtype in df.dtypes.items():
        if dtype == 'int64':
            columns.append(Column(column_name, Integer))
        elif dtype == 'float64':
            columns.append(Column(column_name, Float))
        elif dtype == 'bool':
            columns.append(Column(column_name, Boolean))
        elif dtype == 'datetime64[ns]':
            columns.append(Column(column_name, DateTime))
        else:
            columns.append(Column(column_name, String))
    
    table = Table(table_name, metadata, *columns, UniqueConstraint('id', name='uq_id'))
    metadata.create_all(engine)
    logger.info(f"Table '{table_name}' setup completed in {time.time() - start_time:.2f} seconds")

def dump_df_to_db(df: pd.DataFrame, table_name: str, engine: Engine, chunk_size: int = 500):
    start_time = time.time()
    
    if df.empty:
        logger.warning("DataFrame is empty. No data to insert.")
        return

    create_table_if_not_exists(engine, table_name, df)
    
    existing_columns = get_existing_columns(engine, table_name)
    columns_to_drop = [col for col in df.columns if col not in existing_columns]
    df = df.drop(columns=columns_to_drop)
    
    if columns_to_drop:
        logger.info(f"Dropped columns not in database: {columns_to_drop}")
    
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    total_records = 0
    
    with engine.begin() as conn:
        # Get initial count
        initial_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        
        for chunk in tqdm(chunks, desc="Processing chunks"):
            records = chunk.to_dict('records')
            
            stmt = insert(table).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={c.key: c for c in stmt.excluded if c.key != 'id'}
            )
            conn.execute(stmt)
            total_records += len(records)

        # Get final count
        final_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        new_entries = final_count - initial_count
        
    logger.info(f"Added {new_entries} new entries to {table_name}")
    logger.info(f"Updated {total_records - new_entries} existing entries")
    logger.info(f"Final row count in {table_name}: {final_count}")
