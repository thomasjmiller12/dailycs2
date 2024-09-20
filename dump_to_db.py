import pandas as pd
from sqlalchemy import Table, MetaData, Column, Integer, String, Float, Boolean, DateTime, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from tqdm import tqdm

def create_table_if_not_exists(engine: Engine, table_name: str, df: pd.DataFrame):
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
    
    # Add unique constraint to the 'id' column
    table = Table(table_name, metadata, *columns, UniqueConstraint('id', name='uq_id'))
    metadata.create_all(engine)
    print(f"Table '{table_name}' created or already exists.")

def dump_df_to_db(df: pd.DataFrame, table_name: str, engine: Engine):
    if df.empty:
        print("DataFrame is empty. No data to insert.")
        return

    create_table_if_not_exists(engine, table_name, df)
    
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    with engine.begin() as conn:  # This will automatically commit at the end
        for index, row in tqdm(df.iterrows(), total=len(df)):
            stmt = insert(table).values(row.to_dict())
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={c.key: c for c in stmt.excluded if c.key != 'id'}
            )
            conn.execute(stmt)

        # Query the table to check the number of rows
        print(f"Table name: {table_name}")
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        row_count = result.scalar()
        print(f"Number of rows in {table_name}: {row_count}")

    # Verify that the changes are visible outside the transaction
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        row_count = result.scalar()
        print(f"Verified number of rows in {table_name}: {row_count}")