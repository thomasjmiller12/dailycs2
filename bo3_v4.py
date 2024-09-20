import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import sessionmaker
from db.db_utils import db_connect, create_session
from db.models import Match, Map, MapTeam, Player
from urllib.parse import urljoin
from tqdm import tqdm
from datetime import datetime
import json

# Function to fetch HTML content
def fetch_html_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch content from {url}: {e}")
        return None

# Function to get finished matches from a page
def get_finished_matches(page=1):
    url = f"https://bo3.gg/matches/finished?&period=all_time&page={page}"
    base_url = "https://bo3.gg"
    matches = []

    html_content = fetch_html_content(url)
    if not html_content:
        return matches

    soup = BeautifulSoup(html_content, 'html.parser')
    match_rows = soup.find_all('div', class_='table-row table-row--finished data-advantage')
    for row in match_rows:
        match_link = row.find('a', class_='c-global-match-link table-cell')

        if match_link:
            match_url = urljoin(base_url, match_link['href'])
            
            time_elem = row.find('span', class_='time')
            match_time = time_elem.text if time_elem else "Time not found"
            
            team_elems = row.find_all('div', class_='team-name')
            teams = [team.text for team in team_elems] if team_elems else []
            
            match_info = {
                'url': match_url,
                'time': match_time,
                'teams': teams,
                'maps': []
            }
            
            matches.append(match_info)

    return matches

# Function to scrape match data from the website
def scrape_match_data(url):
    html_content = fetch_html_content(url)
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract start date and time from micro-markup
    micro_markup = soup.find('script', {'id': 'micro-markup', 'type': 'application/ld+json'})
    if micro_markup:
        data = json.loads(micro_markup.string)
        start_date = data[0].get('startDate')
        if start_date:
            match_datetime = datetime.fromisoformat(start_date)
        else:
            print(f"Start date not found in micro-markup for {url}")
            return None
    else:
        print(f"Micro-markup not found for {url}")
        return None
    
    match_data = {
        'url': url,
        'datetime': match_datetime,
        'time': match_datetime.strftime('%H:%M'),
        'teams': [team.text for team in soup.find_all('div', class_='team-name')],
        'maps': []
    }
    
    map_sections = soup.find_all('div', class_='c-nav-match-menu-item c-nav-match-menu-item--game c-nav-match-menu-item--finished')
    for map_section in map_sections:
        map_name_elem = map_section.find('div', class_='map-name')
        map_name = map_name_elem.text.strip() if map_name_elem else "Map name not found"
        
        score_elem = map_section.find('div', class_='c-match-score-map score')
        score = score_elem.text.strip() if score_elem else "Score not found"
        
        map_link_elem = map_section.find('a', class_='menu-link')
        if map_link_elem and 'href' in map_link_elem.attrs:
            map_link = map_link_elem['href']
            full_map_url = urljoin(url, map_link)
            
            map_html_content = fetch_html_content(full_map_url)
            if not map_html_content:
                continue
            
            map_soup = BeautifulSoup(map_html_content, 'html.parser')
            team_tables = map_soup.find_all('div', class_='c-widget-match-scoreboard')
            
            map_data = {
                "name": map_name,
                "score": score,
                "teams": []
            }
            
            for table in team_tables:
                team_name_elem = table.find('div', class_='o-widget__header')
                team_name = team_name_elem.text.strip() if team_name_elem else "Team name not found"
                players_data = extract_player_data(table)
                
                team_data = {
                    "name": team_name,
                    "players": players_data
                }
                map_data["teams"].append(team_data)
            
            match_data['maps'].append(map_data)
        else:
            print(f"Could not find map link for map in match: {url}")
    
    return match_data

def extract_player_data(table):
    players_data = []
    player_rows = table.find_all('div', class_='table-row')
    for row in player_rows:
        player_name_elem = row.find('span', class_='nickname')
        if player_name_elem:
            player_name = player_name_elem.text
            kills_elem = row.find('div', class_='table-cell kills')
            deaths_elem = row.find('div', class_='table-cell deaths')
            if kills_elem and deaths_elem:
                kills = kills_elem.find('p', class_='value').text.strip()
                deaths = deaths_elem.find('p', class_='value').text.strip()
                try:
                    kills = int(kills) if kills else 0
                    deaths = int(deaths) if deaths else 0
                    players_data.append({
                        "name": player_name,
                        "kills": kills,
                        "deaths": deaths
                    })
                except ValueError:
                    print(f"Warning: Invalid kills/deaths data for player {player_name}. Skipping.")
    return players_data

# Function to insert match data into the database
def insert_match_data(session, match_data):
    if match_data is None:
        return False
    
    # Check if match already exists
    existing_match = session.query(Match).filter_by(url=match_data['url']).first()
    if existing_match:
        print(f"Match {match_data['url']} already exists in the database.")
        return True
    
    # Create Match instance
    match = Match(url=match_data['url'], datetime=match_data['datetime'], time=match_data['time'])
    session.add(match)
    
    # Create Map instances and related data
    for map_data in match_data['maps']:
        score_parts = map_data['score'].split('-')
        team1_score = int(score_parts[0])
        team2_score = int(score_parts[1])
        total_rounds = team1_score + team2_score
        
        round_number = int(map_data['teams'][0]['name'].split('(M')[1].split(')')[0])
        
        map_instance = Map(name=map_data['name'], score=map_data['score'], total_rounds=total_rounds, round=round_number, match=match)
        session.add(map_instance)
        
        for i, team_data in enumerate(map_data['teams']):
            team_score = team1_score if i == 0 else team2_score
            team_name = team_data['name'].split(' Scoreboard ')[0]
            map_team = MapTeam(team_name=team_name, map=map_instance, score=team_score)
            session.add(map_team)
            
            for player_data in team_data['players']:
                player = Player(
                    name=player_data['name'],
                    kills=player_data['kills'],
                    deaths=player_data['deaths'],
                    map_team=map_team
                )
                session.add(player)
    
    session.commit()
    print(f"Match {match_data['url']} inserted successfully.")
    return False

# Main function
def main(num_pages=15):
    engine, connection = db_connect()
    session = create_session(engine)
    
    for page in range(1, num_pages + 1):
        print(f"\nProcessing page {page}")
        finished_matches = get_finished_matches(page)
        
        for match_info in tqdm(finished_matches):
            match_data = scrape_match_data(match_info['url'])
            if insert_match_data(session, match_data):
                session.close()
                print("Reached a match that already exists in the database. Stopping scraping.")
                return  # Exit the function if we've found an existing match
    
    session.close()

if __name__ == "__main__":
    main(num_pages=15)  # You can change the number of max