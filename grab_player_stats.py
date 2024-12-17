from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
from urllib.parse import urlencode

BASE_URL = "https://bo3.gg/players"
BASE_PARAMS = {
    "period": "last_6_months",
    "tiers": "s,a,b",
    "sort": "rating",
    "order": "desc",
    "games_count": "30"
}

TAB_CONFIGS = {
    'main': {
        'tab': 'main',
        'sort': 'rating',
        'columns': {
            'rating': {'class': 'rating', 'is_progress_bar': True},
            'maps': {'class': 'games_count', 'is_progress_bar': False},
            'kills': {'class': 'kills', 'is_progress_bar': True},
            'deaths': {'class': 'death', 'is_progress_bar': True},
            'damage': {'class': 'damage', 'is_progress_bar': True}
        }
    },
    'performance': {
        'tab': 'performance',
        'sort': 'opening_kills',
        'columns': {
            'opening_kills': {'class': 'open_kills', 'is_progress_bar': True},
            'opening_deaths': {'class': 'open_death', 'is_progress_bar': True},
            'trades': {'class': 'trades', 'is_progress_bar': True},
            'assists': {'class': 'assists', 'is_progress_bar': True}
        }
    },
    'aim': {
        'tab': 'aim',
        'sort': 'headshots',
        'columns': {
            'headshots': {'class': 'headshots', 'is_progress_bar': True},
            'headshot_percentage': {'class': 'headshot_kills_accuracy', 'is_progress_bar': False},
            'shots': {'class': 'shots', 'is_progress_bar': True},
            'accuracy': {'class': 'accuracy', 'is_progress_bar': False}
        }
    }
}

def setup_driver():
    """Set up Chrome driver with appropriate options"""
    print("\nInitializing Chrome driver...")
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    print("Chrome driver initialized successfully")
    return driver

def fetch_page(driver, tab_config, page_number=1):
    """Fetch a single page of player data using Selenium"""
    params = BASE_PARAMS.copy()
    params['tab'] = tab_config['tab']
    params['sort'] = tab_config['columns'][tab_config['sort']]['class']
    if page_number > 1:
        params['page'] = str(page_number)
    url = f"{BASE_URL}?{urlencode(params)}"
    
    print(f"Fetching {tab_config['tab']} page {page_number}...")
    driver.get(url)
    
    try:
        # Wait for the table group to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "table-group"))
        )
        
        # Additional wait to ensure content is loaded
        time.sleep(3)
        
        # Verify we have the correct sorting column
        sort_class = f".table-cell.{tab_config['columns'][tab_config['sort']]['class']}.current-sorting"
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, sort_class))
        )
        
        # Verify we have actual content
        table_group = driver.find_element(By.CLASS_NAME, "table-group")
        rows = table_group.find_elements(By.CLASS_NAME, "table-row")
        if not rows:
            return False
            
        return True
    except Exception as e:
        return False

def parse_table(driver, tab_config):
    """Parse the table data from the rendered page"""
    players_data = []
    
    try:
        # Wait for table to be present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "table-group"))
        )
        
        # Get all row elements
        rows = driver.find_elements(By.CLASS_NAME, "table-row")
        
        # Skip the header row
        for row_idx in range(1, len(rows)):
            try:
                # Get fresh reference to the row
                row = driver.find_elements(By.CLASS_NAME, "table-row")[row_idx]
                
                # Build the JavaScript for this tab's columns
                column_scripts = []
                for col_name, col_config in tab_config['columns'].items():
                    if col_config['is_progress_bar']:
                        column_scripts.append(
                            f"'{col_name}': (function() {{ " +
                            f"var cell = row.querySelector('.table-cell.{col_config['class']}, .table-cell.{col_config['class']}.current-sorting'); " +
                            "if (!cell) return ''; " +
                            "var progressBar = cell.querySelector('.c-global-progress-bar, .c-global-progress-bar--big, .c-global-progress-bar--no-active'); " +
                            "if (!progressBar) return ''; " +
                            "var span = progressBar.querySelector('.c-global-progress-bar__value p.value span'); " +
                            "return span ? span.textContent.trim() : ''; " +
                            "})()"
                        )
                    else:
                        column_scripts.append(
                            f"'{col_name}': (function() {{ " +
                            f"var cell = row.querySelector('.table-cell.{col_config['class']}'); " +
                            "if (!cell) return ''; " +
                            "var p = cell.querySelector('p.light.u-text-right'); " +
                            "return p ? p.textContent.trim() : ''; " +
                            "})()"
                        )
                
                script = f"""
                    var row = arguments[0];
                    var name = (function() {{
                        var cell = row.querySelector('.table-cell.player');
                        if (!cell) return '';
                        var nickname = cell.querySelector('.nickname');
                        return nickname ? nickname.textContent.trim() : '';
                    }})();
                    
                    return {{
                        'name': name,
                        {','.join(column_scripts)}
                    }};
                """
                
                player_data = driver.execute_script(script, row)
                
                if any(player_data.values()):
                    players_data.append(player_data)
                
            except Exception as e:
                continue
        
        return players_data
    except Exception as e:
        return []

def collect_tab_data(driver, tab_name):
    """Collect all pages of data for a specific tab"""
    tab_config = TAB_CONFIGS[tab_name]
    all_players = []
    page = 1
    max_empty_retries = 2  # Number of empty pages before giving up
    empty_page_count = 0
    
    print(f"\nCollecting {tab_name} tab data...")
    
    while True:
        if not fetch_page(driver, tab_config, page):
            print(f"Failed to load page {page}, trying one more time...")
            time.sleep(2)
            if not fetch_page(driver, tab_config, page):
                break
        
        players = parse_table(driver, tab_config)
        
        if not players:
            empty_page_count += 1
            if empty_page_count >= max_empty_retries:
                break
        else:
            empty_page_count = 0  # Reset counter when we find players
            all_players.extend(players)
            print(f"Collected {len(players)} players from page {page}")
            time.sleep(1)  # Small delay between pages
        
        page += 1
    
    print(f"Total players collected from {tab_name} tab: {len(all_players)}")
    return pd.DataFrame(all_players) if all_players else pd.DataFrame()

def save_to_csv(df, tab_name):
    """Save the data to a CSV file"""
    filename = f"player_stats_{tab_name}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    driver = None
    try:
        driver = setup_driver()
        
        # Collect data from each tab
        main_df = collect_tab_data(driver, 'main')
        performance_df = collect_tab_data(driver, 'performance')
        aim_df = collect_tab_data(driver, 'aim')
        
        # Save individual tab data
        if not main_df.empty:
            save_to_csv(main_df, 'main')
        if not performance_df.empty:
            save_to_csv(performance_df, 'performance')
        if not aim_df.empty:
            save_to_csv(aim_df, 'aim')
        
        # Merge dataframes on player name
        if not any([main_df.empty, performance_df.empty, aim_df.empty]):
            print("\nMerging dataframes...")
            # First merge main and performance
            merged_df = main_df.merge(performance_df, on='name', how='outer').drop_duplicates()
            # Then merge with aim data
            merged_df = merged_df.merge(aim_df, on='name', how='outer').drop_duplicates()
            
            # Save merged data
            merged_df.to_csv('player_stats_complete.csv', index=False)
            print("Complete dataset saved to player_stats_complete.csv")
            print(f"Total players in complete dataset: {len(merged_df)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
