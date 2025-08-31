import streamlit as st
import soccerdata as sd
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import math
from collections import Counter
import io # Potrebno za čitanje uploadovanog fajla

# --- Konfiguracija stranice ---
st.set_page_config(
    page_title="Football Elo Odds Calculator",
    page_icon="⚽",
    layout="wide"
)

# --- Keširanje učitavanja podataka ---
@st.cache_data(ttl=3600)  # Keširanje podataka na 1 sat
def load_current_elo_data():
    """
    Preuzima najnovije klupske Elo rejtinge.
    """
    status_message = st.empty()
    try:
        status_message.info("Preuzimanje najnovijih Elo rejtinga sa ClubElo.com...")
        elo = sd.ClubElo()
        df = elo.read_by_date()
        status_message.empty()
        return df
    except Exception as e:
        status_message.error(f"Neuspešno učitavanje podataka. Greška: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_historical_snapshot(year):
    """
    Preuzima Elo rejtinge za sve klubove na 1. januar izabrane godine.
    """
    with st.spinner(f"Preuzimanje istorijskih podataka za {year}..."):
        try:
            elo = sd.ClubElo()
            df = elo.read_by_date(f"{year}-01-01")
            return df
        except Exception as e:
            st.error(f"Neuspešno učitavanje istorijskih podataka za {year}. Moguće je da podaci za taj datum ne postoje. Greška: {e}")
            return None

@st.cache_data(ttl=3600)
def load_team_history(team_name):
    """
    Preuzima kompletnu istoriju Elo rejtinga za jedan klub.
    """
    with st.spinner(f"Preuzimanje istorije za tim {team_name}..."):
        try:
            elo = sd.ClubElo()
            df = elo.read_team_history(team_name)
            return df
        except Exception as e:
            st.error(f"Neuspešno učitavanje istorijskih podataka za {team_name}. Greška: {e}")
            return None

# --- Funkcije za kalkulaciju Elo i simulacije ---
def calculate_single_match_probs(elo_home, elo_away, hfa_value=0):
    """
    Izračunava verovatnoće za pobedu domaćina, remi ili pobedu gosta za jedan meč.
    """
    elo_home_adj = elo_home + hfa_value
    p_home_win_no_draw = 1 / (1 + 10**((elo_away - elo_home_adj) / 400))
    D = 0.60
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
    """Ažurira Elo rejtinge na osnovu ishoda meča."""
    K_FACTOR = 20
    new_elo_home = elo_home + K_FACTOR * (actual_home_score - expected_home_score)
    new_elo_away = elo_away + K_FACTOR * ((1 - actual_home_score) - (1 - expected_home_score))
    return new_elo_home, new_elo_away

def run_league_simulation(league_teams_df, num_simulations, hfa_value, league_format="Standard Round Robin", playoff_teams=4, split_teams=6, post_split_format="Double Round Robin"):
    """Vrši kompletnu simulaciju lige za određeni broj sezona."""
    teams = league_teams_df.index.tolist()
    num_teams = len(teams)

    initial_elos = league_teams_df['elo'].to_dict()
    position_counts = {team: {pos: 0 for pos in range(1, num_teams + 1)} for team in teams}
    total_points = {team: 0 for team in teams}
    playoff_champions = {team: 0 for team in teams}

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Pokrenuta simulacija {i + 1}/{num_simulations}...")

        sim_elos = initial_elos.copy()
        sim_points = {team: 0 for team in teams}

        if league_format == "Three Round Robin + Split":
            initial_fixtures = list(itertools.permutations(teams, 2)) * 3
        else:
            initial_fixtures = list(itertools.permutations(teams, 2))

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
            else:
                sim_points[away_team] += 3
                actual_home_score = 0.0

            expected_home_score = 1 / (1 + 10**((elo_away - (elo_home + hfa_value)) / 400))
            new_elo_home, new_elo_away = update_elo(elo_home, elo_away, actual_home_score, expected_home_score)
            sim_elos[home_team] = new_elo_home
            sim_elos[away_team] = new_elo_away

        if "Split" in league_format:
            standings_before_split = pd.Series(sim_points).sort_values(ascending=False)
            championship_group = standings_before_split.head(split_teams).index.tolist()
            relegation_group = standings_before_split.tail(num_teams - split_teams).index.tolist()

            for group in [championship_group, relegation_group]:
                if post_split_format == "Double Round Robin":
                    group_fixtures = list(itertools.permutations(group, 2))
                else:
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

    status_text.success("Simulacija završena!")

    results_df = pd.DataFrame.from_dict(position_counts, orient='index')
    results_df['Expected Pts'] = [total_points[team] / num_simulations for team in teams]

    for pos in range(1, num_teams + 1):
        results_df[pos] = results_df[pos] / num_simulations

    if league_format == "Round Robin with Playoff":
        results_df['Champion %'] = pd.Series(playoff_champions) / num_simulations

    results_df.columns = [f"{col}" for col in results_df.columns]
    return results_df.sort_values(by='Expected Pts', ascending=False)

def run_knockout_simulation(teams_df, num_simulations, first_round_matchups):
    """Simulira nokaut turnir sa definisanim parovima prve runde."""
    teams = teams_df.index.tolist()
    initial_elos = teams_df['elo'].to_dict()
    winners = {team: 0 for team in teams}

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Pokrenuta simulacija {i + 1}/{num_simulations}...")

        next_round_teams = []
        for team1, team2 in first_round_matchups:
            prob1_wins, _, prob2_wins = calculate_single_match_probs(initial_elos[team1], initial_elos[team2], hfa_value=0)
            winner = np.random.choice([team1, team2], p=[prob1_wins / (prob1_wins + prob2_wins), prob2_wins / (prob1_wins + prob2_wins)])
            next_round_teams.append(winner)

        round_teams = next_round_teams

        while len(round_teams) > 1:
            np.random.shuffle(round_teams)
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

    status_text.success("Simulacija završena!")

    results = pd.Series(winners) / num_simulations
    return results.sort_values(ascending=False)

def run_round_robin_simulation(teams_df, num_simulations, hfa_value, robin_format="Double"):
    """Simulira round-robin turnir."""
    teams = teams_df.index.tolist()
    initial_elos = teams_df['elo'].to_dict()
    total_points = {team: 0 for team in teams}

    if robin_format == "Double":
        fixtures = list(itertools.permutations(teams, 2))
    else:
        fixtures_tuples = list(itertools.combinations(teams, 2))
        fixtures = []
        for t1, t2 in fixtures_tuples:
            home, away = np.random.choice([t1, t2], 2, replace=False)
            fixtures.append((home, away))

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Pokrenuta simulacija {i + 1}/{num_simulations}...")
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

    status_text.success("Simulacija završena!")

    avg_points = pd.Series({team: total_points[team] / num_simulations for team in teams})
    return avg_points.sort_values(ascending=False)

# --- NOVO: Funkcija za UEFA simulaciju ---
def run_uefa_simulation(teams_df, fixtures, num_simulations, hfa_value):
    """
    Vrši simulaciju UEFA takmičenja po novom formatu.
    """
    initial_elos = teams_df['elo'].to_dict()
    all_teams = list(initial_elos.keys())
    
    # Rezultati koje pratimo
    final_standings_sum = pd.DataFrame(index=all_teams, columns=['total_pts', 'rank_sum'] + [f'pos_{i}' for i in range(1, len(all_teams) + 1)], data=0.0)
    tournament_winners = {team: 0 for team in all_teams}

    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(num_simulations):
        status_text.text(f"Pokrenuta simulacija {i + 1}/{num_simulations}...")
        
        sim_elos = initial_elos.copy()
        sim_points = {team: 0 for team in all_teams}

        # --- Faza 1: Ligaški deo ---
        for home_team, away_team in fixtures:
            elo_home, elo_away = sim_elos.get(home_team, 1500), sim_elos.get(away_team, 1500)
            p_home, p_draw, p_away = calculate_single_match_probs(elo_home, elo_away, hfa_value)
            result = np.random.choice(['H', 'D', 'A'], p=[p_home, p_draw, p_away])
            
            if result == 'H': sim_points[home_team] += 3
            elif result == 'D': sim_points[home_team] += 1; sim_points[away_team] += 1
            else: sim_points[away_team] += 3
        
        # Sortiranje tabele
        league_standings = pd.Series(sim_points).sort_values(ascending=False)
        
        # Ažuriranje statistike
        for rank, team in enumerate(league_standings.index, 1):
            final_standings_sum.loc[team, 'total_pts'] += league_standings[team]
            final_standings_sum.loc[team, 'rank_sum'] += rank
            final_standings_sum.loc[team, f'pos_{rank}'] += 1

        # --- Faza 2: Baraž i Nokaut ---
        top_8 = league_standings.head(8).index.tolist()
        playoff_teams = league_standings.iloc[8:24].index.tolist()
        
        # Simulacija baraža (dvomeč)
        playoff_winners = []
        # Parovi: 9 vs 24, 10 vs 23, itd.
        num_playoff_pairs = len(playoff_teams) // 2
        for j in range(num_playoff_pairs):
            team_a = playoff_teams[j]
            team_b = playoff_teams[len(playoff_teams) - 1 - j]
            
            p_a_wins_leg1, _, _ = calculate_single_match_probs(sim_elos[team_a], sim_elos[team_b], hfa_value)
            p_b_wins_leg2, _, _ = calculate_single_match_probs(sim_elos[team_b], sim_elos[team_a], hfa_value)
            
            # Jednostavnija simulacija dvomeča - ko ima veće šanse da prođe
            prob_a_prog = (p_a_wins_leg1 * (1 - p_b_wins_leg2)) + ((1 - p_a_wins_leg1 - p_b_wins_leg2) * 0.5)
            
            if np.random.rand() < prob_a_prog:
                playoff_winners.append(team_a)
            else:
                playoff_winners.append(team_b)
        
        # --- Faza 3: Nokaut (Top 8 + Pobednici baraža) ---
        knockout_teams = top_8 + playoff_winners
        
        round_teams = knockout_teams
        while len(round_teams) > 1:
            np.random.shuffle(round_teams)
            next_round_teams = []
            for k in range(0, len(round_teams), 2):
                team1, team2 = round_teams[k], round_teams[k+1]
                prob1_wins, _, _ = calculate_single_match_probs(sim_elos[team1], sim_elos[team2], hfa_value=0) # Neutralni teren
                winner = np.random.choice([team1, team2], p=[prob1_wins, 1-prob1_wins])
                next_round_teams.append(winner)
            round_teams = next_round_teams
            
        if round_teams:
            tournament_winners[round_teams[0]] += 1
            
        progress_bar.progress((i + 1) / num_simulations)

    status_text.success("Simulacija završena!")

    # Finalna obrada rezultata
    final_standings_sum['avg_pts'] = final_standings_sum['total_pts'] / num_simulations
    final_standings_sum['avg_rank'] = final_standings_sum['rank_sum'] / num_simulations
    for i in range(1, len(all_teams) + 1):
        final_standings_sum[f'pos_{i}'] /= num_simulations
    
    win_prob = pd.Series(tournament_winners) / num_simulations
    final_standings_sum['win_prob'] = win_prob
    
    return final_standings_sum.sort_values(by='avg_rank')


# --- Pomoćne funkcije za UI ---
def format_value(prob, display_format):
    """Formatuje verovatnoću kao procenat ili decimalnu kvotu."""
    if display_format == "Probabilities":
        return f"{prob:.1%}"
    else:
        return f"{(1/prob):.2f}" if prob > 0 else "—"

def display_outcome_cards(team1_name, team2_name, prob1, prob_draw, prob2, elo1, elo2, display_format="Probabilities", draw_label="Draw"):
    """Pomoćna funkcija za prikaz kartica sa ishodima."""
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

# --- Glavna UI aplikacija ---
st.title("⚽ Football Elo Odds Calculator")

if 'selected_team' not in st.session_state: st.session_state.selected_team = None
if 'corrections' not in st.session_state: st.session_state.corrections = {}


current_elo_df = load_current_elo_data()

if not current_elo_df.empty:
    # --- Sidebar Setup ---
    st.sidebar.header("Calculation Mode")
    calculation_mode = st.sidebar.radio("Use:", ("Current Ratings", "Historical Ratings"), horizontal=True, help="Historical ratings are not available for UEFA simulations.")
    
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

    # --- Glavni tabovi ---
    tab1, tab2, tab4, tab3 = st.tabs(["Match Prediction", "League Simulation", "UEFA Competition Simulation", "Custom Tournament"])

    with tab1:
        st.header("Team Selection")
        col1, col2 = st.columns(2)
        with col1:
            team_a_name = st.selectbox("Select Home Team (Leg 1)", options=all_club_names, index=all_club_names.index("Man City") if "Man City" in all_club_names else 0)
        with col2:
            team_b_name = st.selectbox("Select Away Team (Leg 1)", options=all_club_names, index=all_club_names.index("Real Madrid") if "Real Madrid" in all_club_names else 1)

        if team_a_name and team_b_name:
            #... (Ovaj deo ostaje nepromenjen)
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
                    p_a_wins_leg1, p_draw_leg1, p_b_wins_leg1 = calculate_single_match_probs(elo_a, elo_b, hfa_to_apply)
                    p_b_wins_leg2, p_draw_leg2, p_a_wins_leg2 = calculate_single_match_probs(elo_b, elo_a, hfa_to_apply)
                    prob_a_prog_outright = (p_a_wins_leg1 * p_a_wins_leg2) + (p_a_wins_leg1 * p_draw_leg2) + (p_draw_leg1 * p_a_wins_leg2)
                    prob_b_prog_outright = (p_b_wins_leg1 * p_b_wins_leg2) + (p_b_wins_leg1 * p_draw_leg2) + (p_draw_leg1 * p_b_wins_leg2)
                    prob_extra_time = 1 - prob_a_prog_outright - prob_b_prog_outright
                    final_prob_a = prob_a_prog_outright + (prob_extra_time * 0.5)
                    final_prob_b = prob_b_prog_outright + (prob_extra_time * 0.5)
                    st.subheader("Pre-Match Progression Probability")
                    # ... ostatak koda za dvomeč ostaje isti

    with tab2:
        # ... (Ovaj tab ostaje nepromenjen)
        st.header(f"League Simulation: {selected_league}")
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            num_sims = st.number_input("Number of simulations:", min_value=10, max_value=10000, value=100, step=10, key="league_sims")
        # ... ostatak koda za simulaciju lige ostaje isti

    # --- NOVI TAB ZA UEFA SIMULACIJU ---
    with tab4:
        st.header("UEFA Competition Simulation (New Format)")
        st.info("Ova simulacija koristi 'Current Ratings' i fiksne parove koje uploadujete.")

        num_uefa_sims = st.number_input("Number of simulations:", min_value=10, max_value=10000, value=100, step=10, key="uefa_sims")
        
        uploaded_file = st.file_uploader("Upload CSV file with fixtures", type="csv")
        
        if uploaded_file is not None:
            try:
                # Koristimo io.StringIO da pandas čita fajl kao tekst
                fixtures_df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
                if 'home_team' not in fixtures_df.columns or 'away_team' not in fixtures_df.columns:
                     st.error("CSV file must contain 'home_team' and 'away_team' columns.")
                else:
                    csv_fixtures = list(fixtures_df.itertuples(index=False, name=None))
                    
                    # Validacija imena timova
                    all_csv_teams = set(fixtures_df['home_team'].unique()) | set(fixtures_df['away_team'].unique())
                    unmatched_names = sorted([name for name in all_csv_teams if name not in all_club_names])
                    
                    if unmatched_names:
                        st.warning("Some team names from your file could not be found. Please map them to the official names from ClubElo:")
                        
                        cols = st.columns(3)
                        col_idx = 0
                        
                        # Resetuj korekcije ako se fajl promeni
                        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
                            st.session_state.corrections = {name: None for name in unmatched_names}
                            st.session_state.last_uploaded_file = uploaded_file.name

                        for name in unmatched_names:
                            with cols[col_idx % 3]:
                                selected_match = st.selectbox(
                                    f"Map '{name}':", 
                                    options=[None] + all_club_names, 
                                    key=f"map_{name}",
                                    index=0 # Default na None
                                )
                                if selected_match:
                                    st.session_state.corrections[name] = selected_match
                            col_idx += 1

                        all_mapped = all(st.session_state.corrections.get(name) is not None for name in unmatched_names)
                        
                        if all_mapped:
                            st.success("All names have been mapped!")
                            if st.button("Run Simulation with Mapped Names"):
                                corrected_fixtures = [
                                    (st.session_state.corrections.get(h, h), st.session_state.corrections.get(a, a))
                                    for h, a in csv_fixtures
                                ]
                                simulation_teams = set(h for h, a in corrected_fixtures) | set(a for h, a in corrected_fixtures)
                                sim_teams_df = current_elo_df.loc[list(simulation_teams)][['elo']].copy()
                                
                                results = run_uefa_simulation(sim_teams_df, corrected_fixtures, num_uefa_sims, hfa_to_apply)
                                st.dataframe(results[['avg_rank', 'avg_pts', 'win_prob']].style.format({'avg_rank': '{:.2f}', 'avg_pts': '{:.2f}', 'win_prob': '{:.2%}'}))

                    else: # Sva imena se poklapaju
                        st.success("All team names from the file are valid.")
                        if st.button("Run Simulation"):
                            simulation_teams = all_csv_teams
                            sim_teams_df = current_elo_df.loc[list(simulation_teams)][['elo']].copy()
                            results = run_uefa_simulation(sim_teams_df, csv_fixtures, num_uefa_sims, hfa_to_apply)
                            st.dataframe(results[['avg_rank', 'avg_pts', 'win_prob']].style.format({'avg_rank': '{:.2f}', 'avg_pts': '{:.2f}', 'win_prob': '{:.2%}'}))

            except Exception as e:
                st.error(f"Error processing file: {e}")


    with tab3:
        # ... (Ovaj tab ostaje nepromenjen)
        st.header("Custom Tournament Simulation")
        # ... ostatak koda za custom turnir ostaje isti


else:
    st.warning("Could not load Elo data. The application cannot proceed.")

st.sidebar.markdown("---")
st.sidebar.info("Data from ClubElo.com via `soccerdata`.")
