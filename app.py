import streamlit as st
import soccerdata as sd
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import math
from collections import Counter

# --- Page Configuration ---
st.set_page_config(
    page_title="Football Elo Odds Calculator",
    page_icon="⚽",
    layout="wide"
)

# --- Caching Data Loading ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_current_elo_data():
    """
    Fetches the latest club Elo ratings for the main display.
    """
    status_message = st.empty()
    try:
        status_message.info("Fetching latest Elo ratings from ClubElo.com...")
        elo = sd.ClubElo()
        df = elo.read_by_date()
        status_message.empty()
        return df
    except Exception as e:
        status_message.error(f"Failed to load data. Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_historical_snapshot(year):
    """
    Fetches a snapshot of Elo ratings for all clubs on Jan 1st of the selected year.
    """
    with st.spinner(f"Fetching historical snapshot for {year}..."):
        try:
            elo = sd.ClubElo()
            df = elo.read_by_date(f"{year}-01-01")
            return df
        except Exception as e:
            st.error(f"Failed to load historical data for {year}. It's possible no data exists for that date. Error: {e}")
            return None

@st.cache_data(ttl=3600)
def load_team_history(team_name):
    """
    Fetches the complete historical Elo ratings for a single club.
    """
    with st.spinner(f"Fetching team history for {team_name}..."):
        try:
            elo = sd.ClubElo()
            df = elo.read_team_history(team_name)
            return df
        except Exception as e:
            st.error(f"Failed to load historical data for {team_name}. Error: {e}")
            return None

# --- Elo & Simulation Calculation Functions ---
def calculate_single_match_probs(elo_home, elo_away, hfa_value=0):
    """
    Calculates the probabilities of a home win, draw, or away win for a single match.
    """
    elo_home_adj = elo_home + hfa_value
    p_home_win_no_draw = 1 / (1 + 10**((elo_away - elo_home_adj) / 400))
    D = 0.55
    prob_draw = D * np.sqrt(p_home_win_no_draw * (1 - p_home_win_no_draw))
    prob_home_win = p_home_win_no_draw * (1 - prob_draw)
    prob_away_win = (1 - p_home_win_no_draw) * (1 - prob_draw)
    
    total_prob = prob_home_win + prob_away_win + prob_draw
    if total_prob > 0:
        prob_home_win /= total_prob
        prob_away_win /= total_prob
        prob_draw /= total_prob
    return prob_home_win, prob_draw, prob_away_win

def update_elo(elo_home, elo_away, actual_home_score, expected_home_score):
    """Updates Elo ratings based on match outcome."""
    K_FACTOR = 20
    new_elo_home = elo_home + K_FACTOR * (actual_home_score - expected_home_score)
    new_elo_away = elo_away + K_FACTOR * ((1 - actual_home_score) - (1 - expected_home_score))
    return new_elo_home, new_elo_away

def run_league_simulation(league_teams_df, num_simulations, hfa_value, league_format="Standard Round Robin", playoff_teams=4, split_teams=6, post_split_format="Double Round Robin"):
    """Performs a full league simulation for a specified number of seasons."""
    teams = league_teams_df.index.tolist()
    num_teams = len(teams)
    
    initial_elos = league_teams_df['elo'].to_dict()
    position_counts = {team: {pos: 0 for pos in range(1, num_teams + 1)} for team in teams}
    total_points = {team: 0 for team in teams}
    playoff_champions = {team: 0 for team in teams}
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(num_simulations):
        status_text.text(f"Running simulation {i + 1}/{num_simulations}...")
        
        sim_elos = initial_elos.copy()
        sim_points = {team: 0 for team in teams}

        # Determine the number of initial rounds based on format
        if league_format == "Three Round Robin + Split":
            initial_fixtures = list(itertools.permutations(teams, 2)) * 3
        else: # Standard Round Robin, with or without split/playoff
            initial_fixtures = list(itertools.permutations(teams, 2))

        # Run initial stage fixtures
        for home_team, away_team in initial_fixtures:
            elo_home, elo_away = sim_elos[home_team], sim_elos[away_team]
            p_home, p_draw, p_away = calculate_single_match_probs(elo_home, elo_away, hfa_value)
            result = np.random.choice(['H', 'D', 'A'], p=[p_home, p_draw, p_away])
            
            if result == 'H':
                sim_points[home_team] += 3
                actual_home_score = 1.0
            elif result == 'D':
                sim_points[home_team] += 1
                sim_points[away_team] += 1
                actual_home_score = 0.5
            else: # Away win
                sim_points[away_team] += 3
                actual_home_score = 0.0
                
            expected_home_score = 1 / (1 + 10**((elo_away - (elo_home + hfa_value)) / 400))
            new_elo_home, new_elo_away = update_elo(elo_home, elo_away, actual_home_score, expected_home_score)
            sim_elos[home_team] = new_elo_home
            sim_elos[away_team] = new_elo_away

        # Stage 2: Split and Final Round (if applicable)
        if "Split" in league_format:
            standings_before_split = pd.Series(sim_points).sort_values(ascending=False)
            championship_group = standings_before_split.head(split_teams).index.tolist()
            relegation_group = standings_before_split.tail(num_teams - split_teams).index.tolist()

            for group in [championship_group, relegation_group]:
                if post_split_format == "Double Round Robin":
                    group_fixtures = list(itertools.permutations(group, 2))
                else: # Single Round Robin
                    group_fixtures_tuples = list(itertools.combinations(group, 2))
                    group_fixtures = []
                    for t1, t2 in group_fixtures_tuples:
                        home, away = np.random.choice([t1, t2], 2, replace=False)
                        group_fixtures.append((home, away))

                for home_team, away_team in group_fixtures:
                    elo_home, elo_away = sim_elos[home_team], sim_elos[away_team]
                    p_home, p_draw, p_away = calculate_single_match_probs(elo_home, elo_away, hfa_value)
                    result = np.random.choice(['H', 'D', 'A'], p=[p_home, p_draw, p_away])
                    if result == 'H': sim_points[home_team] += 3
                    elif result == 'D': sim_points[home_team] += 1; sim_points[away_team] += 1
                    else: sim_points[away_team] += 3
            
        final_standings = pd.Series(sim_points).sort_values(ascending=False)
        for pos, team in enumerate(final_standings.index, 1):
            position_counts[team][pos] += 1
            total_points[team] += final_standings[team]

        if league_format == "Round Robin with Playoff":
            playoff_teams_list = final_standings.head(playoff_teams).index.tolist()
            if len(playoff_teams_list) >= 4:
                sf1_winner_prob, _, _ = calculate_single_match_probs(sim_elos[playoff_teams_list[0]], sim_elos[playoff_teams_list[3]], hfa_value=0)
                sf2_winner_prob, _, _ = calculate_single_match_probs(sim_elos[playoff_teams_list[1]], sim_elos[playoff_teams_list[2]], hfa_value=0)
                sf1_winner = np.random.choice([playoff_teams_list[0], playoff_teams_list[3]], p=[sf1_winner_prob, 1-sf1_winner_prob])
                sf2_winner = np.random.choice([playoff_teams_list[1], playoff_teams_list[2]], p=[sf2_winner_prob, 1-sf2_winner_prob])
                final_winner_prob, _, _ = calculate_single_match_probs(sim_elos[sf1_winner], sim_elos[sf2_winner], hfa_value=0)
                champion = np.random.choice([sf1_winner, sf2_winner], p=[final_winner_prob, 1-final_winner_prob])
                playoff_champions[champion] += 1

        progress_bar.progress((i + 1) / num_simulations)
        
    status_text.success("Simulation complete!")
    
    results_df = pd.DataFrame.from_dict(position_counts, orient='index')
    results_df['Expected Pts'] = [total_points[team] / num_simulations for team in teams]
    
    for pos in range(1, num_teams + 1):
        results_df[pos] = results_df[pos] / num_simulations
        
    if league_format == "Round Robin with Playoff":
        results_df['Champion %'] = pd.Series(playoff_champions) / num_simulations

    results_df.columns = [f"{col}" for col in results_df.columns]
    return results_df.sort_values(by='Expected Pts', ascending=False)

def run_knockout_simulation(teams_df, num_simulations, first_round_matchups):
    """Simulates a single-elimination knockout tournament with defined first-round matchups."""
    teams = teams_df.index.tolist()
    initial_elos = teams_df['elo'].to_dict()
    winners = {team: 0 for team in teams}

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Running simulation {i + 1}/{num_simulations}...")
        
        # Simulate the user-defined first round
        next_round_teams = []
        for team1, team2 in first_round_matchups:
            prob1_wins, _, prob2_wins = calculate_single_match_probs(initial_elos[team1], initial_elos[team2], hfa_value=0)
            winner = np.random.choice([team1, team2], p=[prob1_wins / (prob1_wins + prob2_wins), prob2_wins / (prob1_wins + prob2_wins)])
            next_round_teams.append(winner)
        
        round_teams = next_round_teams
        
        # Simulate subsequent rounds
        while len(round_teams) > 1:
            np.random.shuffle(round_teams) # Shuffle winners for random matchups in next round
            next_round_teams = []
            for j in range(0, len(round_teams), 2):
                team1, team2 = round_teams[j], round_teams[j+1]
                prob1_wins, _, prob2_wins = calculate_single_match_probs(initial_elos[team1], initial_elos[team2], hfa_value=0)
                winner = np.random.choice([team1, team2], p=[prob1_wins / (prob1_wins + prob2_wins), prob2_wins / (prob1_wins + prob2_wins)])
                next_round_teams.append(winner)
            round_teams = next_round_teams
        
        if round_teams:
            winners[round_teams[0]] += 1
        
        progress_bar.progress((i + 1) / num_simulations)
    
    status_text.success("Simulation complete!")
    
    results = pd.Series(winners) / num_simulations
    return results.sort_values(ascending=False)

def run_custom_knockout_simulation(teams_df, num_simulations, custom_rounds, hfa_value):
    """Simulates a knockout tournament with a custom bracket structure, including two-legged ties."""
    teams = teams_df.index.tolist()
    initial_elos = teams_df['elo'].to_dict()
    tournament_winners = {team: 0 for team in teams}
    
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Running simulation {i + 1}/{num_simulations}...")
        
        match_results = {} # Stores {'M1.1': ('Winner', 'Loser'), ...} for this simulation run
        
        for round_idx, round_matchups in enumerate(custom_rounds):
            for match_idx, matchup_details in enumerate(round_matchups):
                match_id = f"M{round_idx+1}.{match_idx+1}"
                
                team1_desc = matchup_details['team_a']
                team2_desc = matchup_details['team_b']
                
                team1 = team1_desc if ' of ' not in team1_desc else \
                        match_results[team1_desc.split(' of ')[1]][0] if 'Winner' in team1_desc else \
                        match_results[team1_desc.split(' of ')[1]][1]
                
                team2 = team2_desc if ' of ' not in team2_desc else \
                        match_results[team2_desc.split(' of ')[1]][0] if 'Winner' in team2_desc else \
                        match_results[team2_desc.split(' of ')[1]][1]

                # --- Simulate the Match ---
                if not matchup_details['is_two_legged']:
                    # Standard single-leg neutral venue match
                    prob1_wins, _, prob2_wins = calculate_single_match_probs(initial_elos[team1], initial_elos[team2], hfa_value=0)
                    if prob1_wins + prob2_wins == 0:
                        winner = np.random.choice([team1, team2])
                    else:
                        winner = np.random.choice([team1, team2], p=[prob1_wins / (prob1_wins + prob2_wins), prob2_wins / (prob1_wins + prob2_wins)])
                    loser = team2 if winner == team1 else team1
                else:
                    # Two-legged tie with a given first leg result
                    leg1_a_score = matchup_details['leg1_a']
                    leg1_b_score = matchup_details['leg1_b']
                    
                    # Simulate the second leg (team B is home)
                    p_b_wins_leg2, p_draw_leg2, p_a_wins_leg2 = calculate_single_match_probs(initial_elos[team2], initial_elos[team1], hfa_value)
                    result = np.random.choice(['H', 'D', 'A'], p=[p_b_wins_leg2, p_draw_leg2, p_a_wins_leg2])
                    
                    if result == 'H': leg2_b_score, leg2_a_score = 1, 0
                    elif result == 'D': leg2_b_score, leg2_a_score = 1, 1
                    else: leg2_b_score, leg2_a_score = 0, 1
                    
                    agg_a = leg1_a_score + leg2_a_score
                    agg_b = leg1_b_score + leg2_b_score
                    
                    if agg_a > agg_b:
                        winner, loser = team1, team2
                    elif agg_b > agg_a:
                        winner, loser = team2, team1
                    else: # Aggregate is drawn, check away goals then coin flip
                        if leg2_a_score > leg1_b_score:
                            winner, loser = team1, team2
                        elif leg1_b_score > leg2_a_score:
                            winner, loser = team2, team1
                        else:
                            winner = np.random.choice([team1, team2])
                            loser = team2 if winner == team1 else team1
                
                match_results[match_id] = (winner, loser)

        if match_results:
            final_match_id = list(match_results.keys())[-1]
            tournament_champion = match_results[final_match_id][0]
            tournament_winners[tournament_champion] += 1
            
        progress_bar.progress((i + 1) / num_simulations)
        
    status_text.success("Simulation complete!")
    results = pd.Series(tournament_winners) / num_simulations
    return results.sort_values(ascending=False)


def run_round_robin_simulation(teams_df, num_simulations, hfa_value, robin_format="Double"):
    """Simulates a round-robin tournament."""
    teams = teams_df.index.tolist()
    initial_elos = teams_df['elo'].to_dict()
    total_points = {team: 0 for team in teams}

    if robin_format == "Double":
        fixtures = list(itertools.permutations(teams, 2))
    else: # Single
        fixtures_tuples = list(itertools.combinations(teams, 2))
        fixtures = []
        for t1, t2 in fixtures_tuples:
            home, away = np.random.choice([t1, t2], 2, replace=False)
            fixtures.append((home, away))

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Running simulation {i + 1}/{num_simulations}...")
        sim_points = {team: 0 for team in teams}
        for home_team, away_team in fixtures:
            prob_home, prob_draw, _ = calculate_single_match_probs(initial_elos[home_team], initial_elos[away_team], hfa_value)
            result = np.random.choice(['H', 'D', 'A'], p=[prob_home, prob_draw, 1 - prob_home - prob_draw])
            if result == 'H': sim_points[home_team] += 3
            elif result == 'D': sim_points[home_team] += 1; sim_points[away_team] += 1
            else: sim_points[away_team] += 3
        
        for team in teams:
            total_points[team] += sim_points[team]
        
        progress_bar.progress((i + 1) / num_simulations)

    status_text.success("Simulation complete!")
    
    avg_points = pd.Series({team: total_points[team] / num_simulations for team in teams})
    return avg_points.sort_values(ascending=False)


# --- UI Helper Functions ---
def format_value(prob, display_format):
    """Formats a probability as a percentage or decimal odd."""
    if display_format == "Probabilities":
        return f"{prob:.1%}"
    else:
        return f"{(1/prob):.2f}" if prob > 0 else "—"

def display_outcome_cards(team1_name, team2_name, prob1, prob_draw, prob2, elo1, elo2, display_format="Probabilities", draw_label="Draw"):
    """Helper function to display outcome cards in columns."""
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            if st.button(team1_name, key=f"{team1_name}_card_{draw_label}"):
                st.session_state.selected_team = team1_name
            st.markdown(f"<p style='text-align: center; font-size: 0.9em;'>Elo: {elo1:.0f}</p>", unsafe_allow_html=True)
            if display_format == "Probabilities": st.progress(prob1)
            st.markdown(f"<h4 style='text-align: center;'>{format_value(prob1, display_format)}</h4>", unsafe_allow_html=True)
    with col2:
        with st.container(border=True):
            st.markdown(f"<h5 style='text-align: center;'>{draw_label}</h5>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 0.9em;'>&nbsp;</p>", unsafe_allow_html=True)
            if display_format == "Probabilities": st.progress(prob_draw)
            st.markdown(f"<h4 style='text-align: center;'>{format_value(prob_draw, display_format)}</h4>", unsafe_allow_html=True)
    with col3:
        with st.container(border=True):
            if st.button(team2_name, key=f"{team2_name}_card_{draw_label}"):
                st.session_state.selected_team = team2_name
            st.markdown(f"<p style='text-align: center; font-size: 0.9em;'>Elo: {elo2:.0f}</p>", unsafe_allow_html=True)
            if display_format == "Probabilities": st.progress(prob2)
            st.markdown(f"<h4 style='text-align: center;'>{format_value(prob2, display_format)}</h4>", unsafe_allow_html=True)

# --- Main Application UI ---
st.title("⚽ Football Elo Odds Calculator")

if 'selected_team' not in st.session_state: st.session_state.selected_team = None
if 'custom_rounds' not in st.session_state: st.session_state.custom_rounds = []

current_elo_df = load_current_elo_data()

if not current_elo_df.empty:
    # --- Sidebar Setup ---
    st.sidebar.header("Calculation Mode")
    calculation_mode = st.sidebar.radio("Use:", ("Current Ratings", "Historical Ratings"), horizontal=True)
    
    selected_year = None
    if calculation_mode == "Historical Ratings":
        min_year, max_year = 1946, datetime.now().year
        selected_year = st.sidebar.slider("Select Year", min_value=min_year, max_value=max_year, value=1987)
    else:
        st.session_state.selected_team = None

    current_elo_df['league_full'] = current_elo_df.apply(lambda row: f"{row['country']} (L{int(row['level'])})" if pd.notna(row['country']) and pd.notna(row['level']) else np.nan, axis=1)
    unique_leagues = current_elo_df['league_full'].dropna().unique()
    leagues = ["All"] + sorted(unique_leagues)
    selected_league = st.sidebar.selectbox("Filter by League", options=leagues)

    if selected_league == "All":
        league_club_names = sorted(current_elo_df.index.unique().tolist())
    else:
        league_club_names = sorted(current_elo_df[current_elo_df['league_full'] == selected_league].index.unique().tolist())
    all_club_names = sorted(current_elo_df.index.unique().tolist())

    st.sidebar.header("Match Setup")
    match_type = st.sidebar.radio("Select Match Type", ("Single Match", "Two-Legged Tie"), horizontal=True)
    apply_hfa = st.sidebar.toggle("Apply Home-Field Advantage (HFA)", value=True)
    HFA_VALUE = 65
    hfa_to_apply = HFA_VALUE if apply_hfa else 0
    if apply_hfa: st.sidebar.info(f"Applying **{HFA_VALUE}** Elo HFA bonus.")

    st.sidebar.header("Display Format")
    display_format = st.sidebar.radio("Show as:", ("Probabilities", "Decimal Odds"), horizontal=True, label_visibility="collapsed")

    # --- Main Content Tabs ---
    tab1, tab2, tab3 = st.tabs(["Match Prediction", "League Simulation", "Custom Tournament"])

    with tab1:
        st.header("Team Selection")
        col1, col2 = st.columns(2)
        with col1:
            team_a_name = st.selectbox("Select Home Team (Leg 1)", options=all_club_names, index=all_club_names.index("Man City") if "Man City" in all_club_names else 0)
        with col2:
            team_b_name = st.selectbox("Select Away Team (Leg 1)", options=all_club_names, index=all_club_names.index("Real Madrid") if "Real Madrid" in all_club_names else 1)

        if team_a_name and team_b_name:
            elo_a, elo_b, header_text, error_found = None, None, "", False
            if calculation_mode == "Current Ratings":
                header_text = "Match Prediction (Based on Current Ratings)"
                try:
                    elo_a = current_elo_df.loc[team_a_name]['elo']
                    elo_b = current_elo_df.loc[team_b_name]['elo']
                except KeyError:
                    st.error("Could not retrieve current Elo ratings for one or both teams.")
                    error_found = True
            else: # Historical Ratings
                header_text = f"Match Prediction (Based on {selected_year} Ratings)"
                historical_df = load_historical_snapshot(selected_year)
                if historical_df is not None and not historical_df.empty:
                    try:
                        elo_a = historical_df.loc[team_a_name]['elo']
                        elo_b = historical_df.loc[team_b_name]['elo']
                    except KeyError as e:
                        st.error(f"Team {e} does not have Elo data for the year {selected_year}.")
                        error_found = True
                else: error_found = True

            if not error_found and elo_a is not None and elo_b is not None:
                st.header(header_text)
                
                if match_type == "Single Match":
                    st.subheader(f"Outcome: {team_a_name} (Home) vs. {team_b_name} (Away)")
                    prob_a_wins, prob_draw, prob_b_wins = calculate_single_match_probs(elo_a, elo_b, hfa_to_apply)
                    display_outcome_cards(team_a_name, team_b_name, prob_a_wins, prob_draw, prob_b_wins, elo_a, elo_b, display_format)
                else: # Two-Legged Tie
                    leg_choice = st.radio("Calculate for:", ("First Leg (Pre-Match)", "Second Leg (Live)"), horizontal=True)
                    st.divider()

                    p_a_wins_leg1, p_draw_leg1, p_b_wins_leg1 = calculate_single_match_probs(elo_a, elo_b, hfa_to_apply)
                    p_b_wins_leg2, p_draw_leg2, p_a_wins_leg2 = calculate_single_match_probs(elo_b, elo_a, hfa_to_apply)

                    if leg_choice == "First Leg (Pre-Match)":
                        # --- Pre-match Calculations ---
                        prob_a_prog_outright = (p_a_wins_leg1 * p_a_wins_leg2) + (p_a_wins_leg1 * p_draw_leg2) + (p_draw_leg1 * p_a_wins_leg2)
                        prob_b_prog_outright = (p_b_wins_leg1 * p_b_wins_leg2) + (p_b_wins_leg1 * p_draw_leg2) + (p_draw_leg1 * p_b_wins_leg2)
                        prob_extra_time = 1 - prob_a_prog_outright - prob_b_prog_outright
                        final_prob_a = prob_a_prog_outright + (prob_extra_time * 0.5)
                        final_prob_b = prob_b_prog_outright + (prob_extra_time * 0.5)

                        # --- UI Display ---
                        st.subheader("Pre-Match Progression Probability")
                        prog_col1, prog_col2 = st.columns(2, gap="medium")
                        with prog_col1:
                            with st.container(border=True):
                                if st.button(f"{team_a_name} Progresses", key="team_a_progress"): st.session_state.selected_team = team_a_name
                                st.markdown(f"<p style='text-align: center; font-size: 0.9em;'>Elo: {elo_a:.0f}</p>", unsafe_allow_html=True)
                                if display_format == "Probabilities": st.progress(final_prob_a)
                                st.markdown(f"<h4 style='text-align: center;'>{format_value(final_prob_a, display_format)}</h4>", unsafe_allow_html=True)
                        with prog_col2:
                            with st.container(border=True):
                                if st.button(f"{team_b_name} Progresses", key="team_b_progress"): st.session_state.selected_team = team_b_name
                                st.markdown(f"<p style='text-align: center; font-size: 0.9em;'>Elo: {elo_b:.0f}</p>", unsafe_allow_html=True)
                                if display_format == "Probabilities": st.progress(final_prob_b)
                                st.markdown(f"<h4 style='text-align: center;'>{format_value(final_prob_b, display_format)}</h4>", unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # --- Leg 1 Section ---
                        st.subheader(f"Leg 1 Outcome: {team_a_name} (Home) vs. {team_b_name} (Away)")
                        display_outcome_cards(team_a_name, team_b_name, p_a_wins_leg1, p_draw_leg1, p_b_wins_leg1, elo_a, elo_b, display_format, draw_label="Draw Leg 1")

                    elif leg_choice == "Second Leg (Live)":
                        st.subheader("Enter First Leg Result")
                        score_col1, score_col2 = st.columns(2)
                        leg1_a_score = score_col1.number_input(f"{team_a_name} (Home) Score", min_value=0, step=1, key="leg1_a_live")
                        leg1_b_score = score_col2.number_input(f"{team_b_name} (Away) Score", min_value=0, step=1, key="leg1_b_live")
                        
                        st.divider()
                        
                        # --- Leg 2 Section ---
                        st.subheader(f"Leg 2 Outcome: {team_b_name} (Home) vs. {team_a_name} (Away)")
                        display_outcome_cards(team_b_name, team_a_name, p_b_wins_leg2, p_draw_leg2, p_a_wins_leg2, elo_b, elo_a, display_format, draw_label="Draw Leg 2")
                        
                        st.divider()

                        # --- Live H2H Calculation Section ---
                        num_h2h_sims = 10000
                        team_a_progress_count = 0
                        for _ in range(num_h2h_sims):
                            result = np.random.choice(['H', 'D', 'A'], p=[p_b_wins_leg2, p_draw_leg2, p_a_wins_leg2])
                            # This is a simplified goal model, a more complex one could be used
                            if result == 'H': leg2_b_score, leg2_a_score = 1, 0
                            elif result == 'D': leg2_b_score, leg2_a_score = 1, 1
                            else: leg2_b_score, leg2_a_score = 0, 1
                            
                            agg_a = leg1_a_score + leg2_a_score
                            agg_b = leg1_b_score + leg2_b_score
                            
                            if agg_a > agg_b: 
                                team_a_progress_count += 1
                            elif agg_a == agg_b:
                                if leg2_a_score > leg1_b_score: # Team A wins on away goals
                                    team_a_progress_count += 1
                                elif leg1_b_score > leg2_a_score: # Team B wins on away goals
                                    pass
                                else: # Simplified tie-breaker (50/50 for extra time/penalties)
                                    if np.random.rand() > 0.5: 
                                        team_a_progress_count += 1
                        
                        live_prob_a = team_a_progress_count / num_h2h_sims
                        live_prob_b = 1 - live_prob_a

                        st.subheader("Live Progression Probability (after Leg 1)")
                        live_prog_col1, live_prog_col2 = st.columns(2, gap="medium")
                        with live_prog_col1:
                            if display_format == "Probabilities":
                                st.metric(f"{team_a_name} to Progress", f"{live_prob_a:.1%}")
                            else:
                                st.metric(f"{team_a_name} to Progress", f"{format_value(live_prob_a, display_format)}")
                        with live_prog_col2:
                            if display_format == "Probabilities":
                                st.metric(f"{team_b_name} to Progress", f"{live_prob_b:.1%}")
                            else:
                                st.metric(f"{team_b_name} to Progress", f"{format_value(live_prob_b, display_format)}")

            if st.session_state.selected_team and calculation_mode == "Historical Ratings":
                st.divider()
                st.header(f"Historical Elo Rating for {st.session_state.selected_team}")
                team_history = load_team_history(st.session_state.selected_team)
                if team_history is not None and not team_history.empty: st.line_chart(team_history['elo'])
    
    with tab2:
        st.header(f"League Simulation: {selected_league}")
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            num_sims = st.number_input("Number of simulations:", min_value=10, max_value=10000, value=100, step=10)
        with sim_col2:
            league_format = st.selectbox("League Format:", ["Standard Round Robin", "Round Robin with Playoff", "Standard + Split", "Three Round Robin + Split"])
        
        playoff_teams_count = 0
        split_teams_count = 0
        post_split_format = "Double Round Robin"
        if league_format == "Round Robin with Playoff":
            playoff_teams_count = st.slider("Number of playoff teams:", min_value=2, max_value=8, value=4, step=2)
        elif "Split" in league_format:
            split_teams_count = st.slider("Number of teams in Championship group:", min_value=4, max_value=10, value=6, step=1)
            post_split_format = st.radio("Post-Split Format:", ["Single Round Robin", "Double Round Robin"], horizontal=True)

        with st.expander("Customize Participants and Ratings"):
            initial_sim_teams = league_club_names.copy()
            sim_participants = st.multiselect("Select teams for the simulation", options=all_club_names, default=initial_sim_teams)
            
            if sim_participants:
                sim_teams_df = current_elo_df.loc[sim_participants][['elo']].copy()
                st.write("Edit Elo ratings below:")
                edited_teams_df = st.data_editor(sim_teams_df, column_config={"elo": st.column_config.NumberColumn("Elo Rating", min_value=1000, max_value=2500, step=10, format="%d")}, key="league_sim_editor")
            else:
                edited_teams_df = pd.DataFrame(columns=['elo'])

        # --- Display League and Match Info ---
        num_teams_in_league = len(edited_teams_df)
        st.metric("Number of Teams in Simulation", num_teams_in_league)

        matches_info = ""
        if league_format in ["Standard Round Robin", "Round Robin with Playoff"]:
            matches_per_team = (num_teams_in_league - 1) * 2
            matches_info = f"Each team plays **{matches_per_team}** matches in the regular season."
            if league_format == "Round Robin with Playoff":
                matches_info += f" Top {playoff_teams_count} teams advance to a knockout playoff."
        elif "Split" in league_format:
            if league_format == "Standard + Split":
                reg_season_matches = (num_teams_in_league - 1) * 2
            else: # Three Round Robin + Split
                reg_season_matches = (num_teams_in_league - 1) * 3
            
            if post_split_format == "Double Round Robin":
                champ_group_matches = (split_teams_count - 1) * 2
                relegation_group_size = num_teams_in_league - split_teams_count
                relegation_group_matches = (relegation_group_size - 1) * 2
            else: # Single Round Robin
                champ_group_matches = split_teams_count - 1
                relegation_group_size = num_teams_in_league - split_teams_count
                relegation_group_matches = relegation_group_size - 1

            matches_info = (f"Regular season: **{reg_season_matches}** matches per team.\n\n"
                            f"After split, top {split_teams_count} teams play **{champ_group_matches}** more matches.\n\n"
                            f"Bottom {relegation_group_size} teams play **{relegation_group_matches}** more matches.")
        st.info(matches_info)

        if st.button("Run Simulation"):
            if calculation_mode == "Current Ratings":
                if not edited_teams_df.empty:
                    results = run_league_simulation(edited_teams_df, num_sims, hfa_to_apply, league_format, playoff_teams_count, split_teams_count, post_split_format)
                    position_cols = [col for col in results.columns if col not in ['Expected Pts', 'Champion %']]
                    
                    if display_format == "Decimal Odds":
                        display_df = results.copy()
                        for col in position_cols:
                            display_df[col] = display_df[col].apply(lambda p: 1/p if p > 0 else np.nan)
                        if 'Champion %' in display_df.columns:
                             display_df['Champion %'] = display_df['Champion %'].apply(lambda p: 1/p if p > 0 else np.nan)
                        st.dataframe(display_df.style.format("{:.2f}", na_rep="-"))
                    else: # Probabilities
                        format_dict = {'Expected Pts': '{:.2f}'}
                        if 'Champion %' in results.columns:
                            format_dict['Champion %'] = '{:.1%}'
                        st.dataframe(results.style.background_gradient(cmap='Greens', subset=position_cols).format("{:.1%}").format(format_dict))
                else:
                    st.warning("Please select/confirm teams in the customization section before running.")
            else:
                st.warning("League simulation is only available for 'Current Ratings' mode.")
    
    with tab3:
        st.header("Custom Tournament Simulation")
        
        tournament_format = st.selectbox("Tournament Format:", ["Single Elimination Knockout", "Custom Knockout", "Single Round Robin", "Double Round Robin"])
        
        selected_teams = st.multiselect("Select Teams:", options=all_club_names)
        
        num_knockout_sims = st.number_input("Number of tournament simulations:", min_value=10, max_value=10000, value=1000, step=100)

        if tournament_format == "Custom Knockout":
            st.subheader("Build Your Custom Bracket")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add Round"):
                    st.session_state.custom_rounds.append([])
                    st.rerun()
            with col2:
                if st.button("Clear Bracket"):
                    st.session_state.custom_rounds = []
                    st.rerun()

            for i in range(len(st.session_state.custom_rounds)):
                st.markdown(f"--- \n#### Round {i+1}")
                
                if st.button(f"Add Matchup to Round {i+1}", key=f"add_match_r{i}"):
                    st.session_state.custom_rounds[i].append({'team_a': None, 'team_b': None, 'is_two_legged': False, 'leg1_a': 0, 'leg1_b': 0})
                    st.rerun()
                
                options = selected_teams.copy()
                if i > 0:
                    for r_idx, r_matchups_prev in enumerate(st.session_state.custom_rounds[:i]):
                        for m_idx in range(len(r_matchups_prev)):
                            options.append(f"Winner of M{r_idx+1}.{m_idx+1}")
                            options.append(f"Loser of M{r_idx+1}.{m_idx+1}")

                options_with_placeholder = [None] + sorted(list(set(options)))

                for j in range(len(st.session_state.custom_rounds[i])):
                    matchup_details = st.session_state.custom_rounds[i][j]
                    cols = st.columns([3, 3, 2])
                    
                    with cols[0]:
                        current_val_a = matchup_details.get('team_a')
                        index_a = options_with_placeholder.index(current_val_a) if current_val_a in options_with_placeholder else 0
                        st.session_state.custom_rounds[i][j]['team_a'] = st.selectbox(f"Match {j+1} - Team A", options=options_with_placeholder, key=f"r{i}_m{j}_tA", index=index_a, format_func=lambda x: "Select Team A" if x is None else x)

                    with cols[1]:
                        current_val_b = matchup_details.get('team_b')
                        index_b = options_with_placeholder.index(current_val_b) if current_val_b in options_with_placeholder else 0
                        st.session_state.custom_rounds[i][j]['team_b'] = st.selectbox(f"Match {j+1} - Team B", options=options_with_placeholder, key=f"r{i}_m{j}_tB", index=index_b, format_func=lambda x: "Select Team B" if x is None else x)
                    
                    with cols[2]:
                        st.write("") 
                        st.write("") 
                        st.session_state.custom_rounds[i][j]['is_two_legged'] = st.checkbox("2-Legged Tie?", key=f"r{i}_m{j}_2leg", value=matchup_details.get('is_two_legged', False))

                    if st.session_state.custom_rounds[i][j]['is_two_legged']:
                        score_cols = st.columns([1,1,4])
                        with score_cols[0]:
                            st.session_state.custom_rounds[i][j]['leg1_a'] = st.number_input("Leg 1 Score A", min_value=0, step=1, key=f"r{i}_m{j}_sA", value=matchup_details.get('leg1_a', 0))
                        with score_cols[1]:
                            st.session_state.custom_rounds[i][j]['leg1_b'] = st.number_input("Leg 1 Score B", min_value=0, step=1, key=f"r{i}_m{j}_sB", value=matchup_details.get('leg1_b', 0))
                    st.markdown("---")


        first_round_matchups = []
        manual_matchups_valid = True
        if tournament_format == "Single Elimination Knockout":
            num_teams = len(selected_teams)
            if num_teams > 1 and (num_teams & (num_teams - 1) == 0):
                manual_matchups = st.checkbox("Manually set first-round matchups")
                if manual_matchups:
                    st.write("Define First Round Matchups:")
                    for i in range(num_teams // 2):
                        cols = st.columns(2)
                        team1 = cols[0].selectbox(f"Match {i+1} - Team A", options=selected_teams, key=f"m{i}_tA", index=None)
                        team2 = cols[1].selectbox(f"Match {i+1} - Team B", options=selected_teams, key=f"m{i}_tB", index=None)
                        if team1 and team2:
                            first_round_matchups.append((team1, team2))
                    
                    all_matchup_teams = [team for pair in first_round_matchups for team in pair]
                    if len(all_matchup_teams) != num_teams and len(first_round_matchups) == num_teams // 2:
                         st.warning("Please assign all selected teams to a matchup.")
                         manual_matchups_valid = False
                    elif len(set(all_matchup_teams)) != len(all_matchup_teams):
                        counts = Counter(all_matchup_teams)
                        duplicates = [team for team, count in counts.items() if count > 1]
                        st.error(f"Each team can only be in one matchup. Duplicates found: {', '.join(duplicates)}")
                        manual_matchups_valid = False

        if st.button("Run Custom Simulation"):
            if not selected_teams:
                st.warning("Please select at least two teams.")

            elif tournament_format == "Custom Knockout":
                is_valid_bracket = all(
                    matchup['team_a'] is not None and matchup['team_b'] is not None
                    for round_matchups in st.session_state.custom_rounds
                    for matchup in round_matchups
                )

                if st.session_state.custom_rounds and is_valid_bracket:
                    
                    all_teams_in_bracket = set()
                    for round_matchups in st.session_state.custom_rounds:
                        for matchup in round_matchups:
                            if ' of ' not in matchup['team_a']: all_teams_in_bracket.add(matchup['team_a'])
                            if ' of ' not in matchup['team_b']: all_teams_in_bracket.add(matchup['team_b'])
                    
                    if not all(team in current_elo_df.index for team in all_teams_in_bracket if team is not None):
                        st.error("One or more teams in the bracket could not be found in the Elo database. Please check the names.")
                    else:
                        teams_to_fetch = [team for team in all_teams_in_bracket if team is not None]
                        knockout_teams_df = current_elo_df.loc[list(teams_to_fetch)][['elo']].copy()
                        custom_knockout_results = run_custom_knockout_simulation(knockout_teams_df, num_knockout_sims, st.session_state.custom_rounds, hfa_to_apply)
                        
                        st.subheader("Tournament Win Probability")
                        if display_format == "Decimal Odds":
                            custom_knockout_results = custom_knockout_results.apply(lambda p: 1/p if p > 0 else np.nan)
                            st.dataframe(custom_knockout_results.to_frame(name="Decimal Odds").style.format("{:.2f}", na_rep="-"))
                        else:
                            st.dataframe(custom_knockout_results.to_frame(name="Win Probability").style.format("{:.1%}"))
                        st.bar_chart(custom_knockout_results)
                else:
                    st.error("Please ensure all matchups in your custom bracket are filled before running the simulation.")

            elif tournament_format == "Single Elimination Knockout":
                num_teams = len(selected_teams)
                if num_teams > 1 and (num_teams & (num_teams - 1) == 0):
                    if not manual_matchups:
                        shuffled_teams = selected_teams.copy()
                        np.random.shuffle(shuffled_teams)
                        first_round_matchups = [(shuffled_teams[j], shuffled_teams[j+1]) for j in range(0, num_teams, 2)]

                    if manual_matchups_valid:
                        knockout_teams_df = current_elo_df.loc[selected_teams][['elo']].copy()
                        knockout_results = run_knockout_simulation(knockout_teams_df, num_knockout_sims, first_round_matchups)
                        
                        st.subheader("Tournament Win Probability")
                        if display_format == "Decimal Odds":
                            knockout_results = knockout_results.apply(lambda p: 1/p if p > 0 else np.nan)
                            st.dataframe(knockout_results.to_frame(name="Decimal Odds").style.format("{:.2f}", na_rep="-"))
                        else:
                            st.dataframe(knockout_results.to_frame(name="Win Probability").style.format("{:.1%}"))
                        st.bar_chart(knockout_results)
                else:
                    st.error("Please select a number of teams that is a power of 2 (e.g., 4, 8, 16, 32) for a knockout tournament.")
            
            else: # Round Robin formats
                robin_format = "Single" if tournament_format == "Single Round Robin" else "Double"
                round_robin_teams_df = current_elo_df.loc[selected_teams][['elo']].copy()
                round_robin_results = run_round_robin_simulation(round_robin_teams_df, num_knockout_sims, hfa_to_apply, robin_format)
                
                st.subheader("Simulated Round Robin Results")
                st.dataframe(round_robin_results.to_frame(name="Average Points").style.format("{:.2f}"))


else:
    st.warning("Could not load Elo data. The application cannot proceed.")

st.sidebar.markdown("---")
st.sidebar.info("Data from ClubElo.com via `soccerdata`.")

