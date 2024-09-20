# CS2 Match Data and Projections

This repository contains scripts for collecting CS2 match data and player projections.

## Setup

1. Clone this repository
2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with Supabase credentials:
   ```
   DATABASE_USERNAME=your_username
   DATABASE_PASSWORD=your_password
   DATABASE_NAME=your_database_name
   DATABASE_PORT=your_port
   DATABASE_HOST=your_host
   ```
   You can find these credentials in your Supabase project settings.

## Main Scripts

### bo3_v4.py

Collects CS2 match data from HLTV and stores it in the database. Includes information about matches, maps, teams, and player performance.

To run:
```
python bo3_v4.py
```

### pp.py

Retrieves player projections and saves them to the database.

To run:
```
python pp.py
```

## Database

The `db` folder contains database-related files:

- `models.py`: Defines the database schema using SQLAlchemy ORM
- `db_utils.py`: Provides utility functions for database connections and operations

## Note

Please ignore the `pytorch_model` and `claude_model` folders as they are works in progress and not part of the main functionality of this repository.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or bug fixes.
