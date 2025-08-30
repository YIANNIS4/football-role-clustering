import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 Layout enhancements
st.markdown("""<style>
    .stPlotlyChart, .stPyplot {
        margin-left: auto;
        margin-right: auto;
    }
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_name):
    """
    Loads and cleans a specific dataset.
    This function will load the CSV file and handle potential errors.
    """
    base_path = Path(__file__).resolve().parent
    file_path = base_path / "data" / file_name

    # Check if the file is in a 'data' subfolder, if not, check the current directory
    if not file_path.exists():
        file_path = base_path / file_name

    if not file_path.exists():
        st.error(f"❌ File not found at: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Check if the 'Role_Label' column exists
        if 'Role_Label' in df.columns:
            # Check if we are processing a forwards file
            if "forwards" in file_name.lower():
                # --- CORRECTED LINE: Matching the capitalization 'Off-Ball Disruptor' ---
                df['Role_Label'] = df['Role_Label'].replace('Off-Ball Disruptor', 'Channel Runner')

        minutes_aliases = {
          'Minutes Played', 'minutes played', 'Minutes', 'minutes',
          'Min', 'min', 'mins_played', 'Mins', '90s', 'time_minutes'
        }
        # try exact, then case-insensitive
        found_minutes = None
        for c in df.columns:
            if c in minutes_aliases:
                found_minutes = c
                break
        if found_minutes is None:
            lower_map = {c.lower(): c for c in df.columns}
            for alias in [a.lower() for a in minutes_aliases]:
                if alias in lower_map:
                    found_minutes = lower_map[alias]
                    break

        if found_minutes is not None:
            if found_minutes != 'Minutes Played':
                df.rename(columns={found_minutes: 'Minutes Played'}, inplace=True)
        else:
            st.warning("⚠️ 'Minutes Played' not found; creating placeholder (0).")
            df['Minutes Played'] = 0
                
        
        st.success(f"✅ Loaded: {df.shape[0]} players, {df.shape[1]} features from {file_name}")
        return df
    except Exception as e:
        st.error(f"Error loading file {file_name}: {e}")
        return pd.DataFrame()

def find_similar_players(df, target_player, features, top_n=5):
    """
    Finds and returns the top N players most similar to a target player
    based on the cosine similarity of selected features.
    """
    df_copy = df.dropna(subset=features).copy()
    if target_player not in df_copy['Player Name'].values:
        return pd.DataFrame()

    matrix = df_copy[features].values
    target_vector = df_copy[df_copy['Player Name'] == target_player][features].values
    scores = cosine_similarity(target_vector, matrix).flatten()

    df_copy['Similarity'] = scores
    similar_df = df_copy[df_copy['Player Name'] != target_player]
    
    # Return the correct columns based on the available data
    if 'League' in df_copy.columns:
        return similar_df.sort_values(by='Similarity', ascending=False).head(top_n)[['Player Name', 'Club', 'League', 'Similarity']]
    else:
        return similar_df.sort_values(by='Similarity', ascending=False).head(top_n)[['Player Name', 'Club', 'league', 'Similarity']]


def render_defenders_dashboard(df):
    """
    Renders the Streamlit dashboard for defenders.
    """
    if 'minutes' in df.columns:
        df = df.rename(columns={'minutes': 'Minutes Played'})
        df['League'] = df['League'].str.strip().str.title()
        df['Tactical Role'] = df['Tactical Role'].astype(str).str.strip()


    # Explanations for tactical roles and PCA components
    tactical_roles_desc = {
        'Ball-Winning Full-Back': 'High-engagement flank defender focused on duels, pressing, and regaining possession.',
        'Ball-Playing Centre-Back': 'Calm under pressure, excels in structured passing and line-breaking build-up.',
        'Defensive Anchor / Stopper': 'Central protector specializing in aerial duels, clearances, and deep-zone resilience.',
        'Positional Defender': 'Maintains disciplined structure, controls space through smart positioning and low-risk defending.'
    }

    pca_explanations = {
        'Aggressive High Block Engagement': 'Quantifies a defender\'s proactive pressure and tackles high up the pitch, disrupting opponents\' build-up.',
        'Transition Ability (Interceptions-Passing)': 'Measures a defender\'s effectiveness in winning the ball and immediately initiating forward attacks with accurate passes.',
        'Area Clearance & Box Protection': 'Reflects a defender\'s primary role in clearing danger from their own penalty area, often through headers and clearances.',
        'Build-up Ability': 'Describes a defender\'s comfort and accuracy in passing from the back, contributing to their team\'s attacking phases.',
        '1v1 Ability': 'Assesses a defender\'s skill in isolating and winning one-on-one duels against dribbling attackers.',
        'Compactness & Shot Blocks': 'Measures a defender\'s discipline in maintaining a tight defensive shape and blocking shots to prevent scoring opportunities.',
        'Overall Defensive Workload Index': 'An aggregate measure of a defender\'s total defensive activity, encompassing tackles, interceptions, and clearances.',
        'Vertical Passing / Line-Breaking': 'Gauges a defender\'s capability to play penetrative passes that bypass multiple opposition lines.',
        'Low-Tempo Defensive Coverage': 'Highlights a defender\'s ability to maintain a deep, stable defensive shape, controlling space without high-intensity pressing.',
        'Mid-Third Engagements': 'Focuses on a defender\'s proactive actions, such as tackles and interceptions, in the middle of the pitch.',
        'Long Passing Ability - Pressured Duel Success': 'A dual metric combining a defender\'s accuracy in long passing with their success rate in duels while under pressure.',
        'Errors Under Pressure': 'Indicates a defender\'s tendency to make mistakes (e.g., losing possession, misplacing a pass) when facing intense pressure.',
        'Area Variety in Tackling': 'Measures a defender\'s versatility and effectiveness in tackling across different zones of the pitch.',
        'Structured Passing in Tactical Mid-Blocks': 'Describes a defender\'s ability to circulate the ball with purpose within a disciplined mid-block defensive system.',
        'Defensive Stress Indicators & Rescue Actions': 'Quantifies how a defender reacts and performs in high-pressure defensive situations, often involving last-ditch clearances or blocks.',
        'Passing Reliability Across Areas': 'Assesses a defender\'s consistent passing accuracy from various positions on the field, minimizing turnovers.',
        'Defensive Energy & Effort Radar': 'An overarching metric representing a defender\'s intensity and involvement in all defensive actions throughout a match.'
    }

    composite_groups = {
        'High Pressing & Intensity': ['Aggressive High Block Engagement', 'Defensive Energy & Effort Radar'],
        'Mid-Zone Engagement': ['Mid-Third Engagements', 'Area Variety in Tackling', 'Overall Defensive Workload Index'],
        'Build-up & Ball Security': ['Build-up Ability', 'Passing Reliability Across Areas'],
        'Progressive Passing': ['Vertical Passing / Line-Breaking', 'Structured Passing in Tactical Mid-Blocks', 'Long Passing Ability - Pressured Duel Success'],
        '1v1 & Duel Success': ['1v1 Ability', 'Low-Tempo Defensive Coverage'],
        'Box & Area Defense': ['Compactness & Shot Blocks', 'Area Clearance & Box Protection'],
        'Transition Ability': ['Transition Ability (Interceptions-Passing)'],
        'Error Management': ['Errors Under Pressure', 'Defensive Stress Indicators & Rescue Actions']
    }

    composite_descriptions = {
        'High Pressing & Intensity': 'A composite score representing a player\'s proactive defensive actions and high-intensity pressing.',
        'Mid-Zone Engagement': 'A composite score for a player\'s effectiveness in winning the ball in the middle of the pitch and contributing to the overall defensive workload.',
        'Build-up & Ball Security': 'A composite score for a player\'s comfort and accuracy in passing from the back, minimizing turnovers.',
        'Progressive Passing': 'A composite score for a player\'s ability to break opponent lines with penetrative passes and distribute the ball effectively within a tactical system.',
        '1v1 & Duel Success': 'A composite score for a player\'s skill in winning one-on-one duels and maintaining a stable defensive shape.',
        'Box & Area Defense': 'A composite score for a player\'s primary role in clearing danger from their own penalty area and blocking shots.',
        'Transition Ability': 'A metric for a player\'s effectiveness in winning the ball and immediately initiating forward attacks with accurate passes.',
        'Error Management': 'A composite score reflecting a player\'s ability to handle high-pressure defensive situations and avoid making mistakes under pressure.'
    }
    
    # Pre-processing for defenders
    pca_metrics_map = {
        'High Pressing & Disruptive Actions': 'Aggressive High Block Engagement',
        'Recovery & Ball Circulation': 'Transition Ability (Interceptions-Passing)',
        'Deep Block Clearance Profile': 'Area Clearance & Box Protection',
        'Controlled Distribution & Ball Security': 'Build-up Ability',
        '1v1 Retention & Defensive Control': '1v1 Ability',
        'Defensive Zone Coverage & Reactive Blocking': 'Compactness & Shot Blocks',
        'Defensive Volume & Involvement': 'Overall Defensive Workload Index',
        'Progressive Distribution Profile': 'Vertical Passing / Line-Breaking',
        'Passive Disruption & Block Output': 'Low-Tempo Defensive Coverage',
        'Mid-Zone Containment Behavior': 'Mid-Third Engagements',
        'Duel Precision & Long-Range Accuracy': 'Long Passing Ability - Pressured Duel Success',
        'Pressure Reaction & Error Tendency': 'Errors Under Pressure',
        'Tactical Pressing Spectrum': 'Area Variety in Tackling',
        'Mid-Block Distribution Accuracy': 'Structured Passing in Tactical Mid-Blocks',
        'Error Volume & Bailout Actions': 'Defensive Stress Indicators & Rescue Actions',
        'Foundational Passing Backbone': 'Passing Reliability Across Areas',
        'Defensive Interaction Intensity': 'Defensive Energy & Effort Radar'
    }

    df = df.rename(columns=pca_metrics_map)
    pca_metrics = list(pca_metrics_map.values())
    raw_unscaled_metrics = [
        'Goals per 90', 'Assists per 90',
        'Total Pass Accuracy %', 'Tackles Success %', 'Dribblers Tackled',
        'Blocks per 90', 'Shots Blocked per 90', 'Passes Blocked per 90',
        'Clearances per 90', 'Errors per 90'
    ]
     
     # Convert percentage metrics from decimals to proper percentages
    percentage_metrics = ['Total Pass Accuracy %', 'Tackles Success %']
    for metric in percentage_metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            df[metric] = df[metric] * 100

    
    # Scale all raw and PCA metrics
    all_metrics_to_scale = raw_unscaled_metrics + pca_metrics
    for metric in all_metrics_to_scale:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            df[metric + '_scaled'] = df[metric].rank(pct=True, method='dense') * 100

    # Create raw and scaled composite metrics
    df['High Pressing & Intensity'] = df[[
        'Aggressive High Block Engagement', 'Defensive Energy & Effort Radar'
    ]].mean(axis=1)
    df['High Pressing & Intensity_scaled'] = df[[
        'Aggressive High Block Engagement_scaled', 'Defensive Energy & Effort Radar_scaled'
    ]].mean(axis=1)

    df['Mid-Zone Engagement'] = df[[
        'Mid-Third Engagements', 'Area Variety in Tackling', 'Overall Defensive Workload Index'
    ]].mean(axis=1)
    df['Mid-Zone Engagement_scaled'] = df[[
        'Mid-Third Engagements_scaled', 'Area Variety in Tackling_scaled', 'Overall Defensive Workload Index_scaled'
    ]].mean(axis=1)

    df['Build-up & Ball Security'] = df[[
        'Build-up Ability', 'Passing Reliability Across Areas'
    ]].mean(axis=1)
    df['Build-up & Ball Security_scaled'] = df[[
        'Build-up Ability_scaled', 'Passing Reliability Across Areas_scaled'
    ]].mean(axis=1)

    df['Progressive Passing'] = df[[
        'Vertical Passing / Line-Breaking', 'Structured Passing in Tactical Mid-Blocks', 'Long Passing Ability - Pressured Duel Success'
    ]].mean(axis=1)
    df['Progressive Passing_scaled'] = df[[
        'Vertical Passing / Line-Breaking_scaled', 'Structured Passing in Tactical Mid-Blocks_scaled', 'Long Passing Ability - Pressured Duel Success_scaled'
    ]].mean(axis=1)
    
    df['1v1 & Duel Success'] = df[[
        '1v1 Ability', 'Low-Tempo Defensive Coverage'
    ]].mean(axis=1)
    df['1v1 & Duel Success_scaled'] = df[[
        '1v1 Ability_scaled', 'Low-Tempo Defensive Coverage_scaled'
    ]].mean(axis=1)
    
    df['Box & Area Defense'] = df[[
        'Compactness & Shot Blocks', 'Area Clearance & Box Protection'
    ]].mean(axis=1)
    df['Box & Area Defense_scaled'] = df[[
        'Compactness & Shot Blocks_scaled', 'Area Clearance & Box Protection_scaled'
    ]].mean(axis=1)

    df['Transition Ability'] = df[['Transition Ability (Interceptions-Passing)']].mean(axis=1)
    df['Transition Ability_scaled'] = df[['Transition Ability (Interceptions-Passing)_scaled']].mean(axis=1)

    df['Error Management'] = df[['Errors Under Pressure', 'Defensive Stress Indicators & Rescue Actions']].mean(axis=1)
    df['Error Management_scaled'] = df[['Errors Under Pressure_scaled', 'Defensive Stress Indicators & Rescue Actions_scaled']].mean(axis=1)
    
    composite_metrics = list(composite_groups.keys())
    
    roles = df['Tactical Role'].dropna().unique()
    players = df['Player Name'].dropna().unique()

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Defenders Dashboard Filters")
    role_options = ["All"] + sorted(roles)
    selected_roles = st.sidebar.multiselect("Select Role(s)", role_options, default=["All"], key="defenders_role_select")

    if "All" in selected_roles:
        filtered_df = df.copy()
    else:
        filtered_df = df[df['Tactical Role'].isin(selected_roles)]

    selected_players = st.sidebar.multiselect("Compare Players (Radar + Bar)", sorted(players), default=[], key="defenders_player_compare")
    single_player_filter = st.sidebar.selectbox("Filter Table by Player", ["All"] + sorted(players), key="defenders_single_player_filter")

    metric_type_choice = st.sidebar.radio(
        "Choose Metric Type",
        ("Key Performance Areas Groups", "Individual Key Performance Areas", "Raw Unscaled Metrics"),
        key="defenders_metric_type"
    )

    if metric_type_choice == "Key Performance Areas Groups":
        metric_options = composite_metrics
        default_metrics = composite_metrics
    elif metric_type_choice == "Individual Key Performance Areas":
        metric_options = pca_metrics
        default_metrics = pca_metrics
    else: # "Raw Unscaled Metrics"
        metric_options = raw_unscaled_metrics
        default_metrics = raw_unscaled_metrics

    user_metrics = st.sidebar.multiselect(
        "📊 Choose Metrics for Radar & Comparison",
        [m for m in metric_options if m + '_scaled' in df.columns],
        default=default_metrics,
        key="defenders_user_metrics"
    )
    scaled_user_metrics = [m + '_scaled' for m in user_metrics]

    filtered_df = df[df['Tactical Role'].isin(selected_roles)]
    if single_player_filter != "All":
        filtered_df = filtered_df[filtered_df['Player Name'] == single_player_filter]

    st.title("📊 Tactical Dashboard for Defenders")

    with st.expander("ℹ️ Tactical Role & Key Performance Areas Explanations"):
        st.markdown("### 🔑 Role Labels")
        for role, desc in tactical_roles_desc.items():
            st.markdown(f"- **{role}**: {desc}")
        
        st.markdown("### 📊 Key Performance Areas Groups")
        for group, pcas in composite_groups.items():
            st.markdown(f"- **{group}**: {composite_descriptions[group]} (Includes: " + ", ".join(pcas) + ")")
        
        st.markdown("### 📈 Individual Key Performance Areas")
        for pca, desc in pca_explanations.items():
            group_name = next((group for group, pcas in composite_groups.items() if pca in pcas), "N/A")
            st.markdown(f"- **{pca}** (Group: *{group_name}*): {desc}")
            
    st.write("✅ La Liga defenders in filtered_df:", filtered_df[filtered_df["league"] == "La Liga"].shape[0])


    st.subheader("📋 Player Table")

    table_columns = ['Player Name', 'League', 'Club', 'Position', 'Tactical Role']
    if 'Minutes Played' in df.columns:
        table_columns.append('Minutes Played')

    # Only include raw metrics (PCA + raw performance)
    raw_metrics_for_table = table_columns + [col for col in raw_unscaled_metrics + pca_metrics if col in df.columns]

     


    st.dataframe(filtered_df[raw_metrics_for_table].round(2))


    st.subheader("Average Metric Ranks by Role (0-100)")
    st.markdown(
        """
        **What these numbers mean:** This table shows the **average percentile rank** for each metric within a specific role. A percentile rank is a value from **0 to 100**, indicating how a player performs relative to all other players in the dataset.

        **Example:** A value of `29.35` for `Goals per 90` means the average player in this role performs better than only `29.35%` of all players in that metric. This is expected for roles like a Defender, where scoring is not a primary function. A high number, such as `95.00` for a metric like `Tackles Success %`, would show that the average player in this role is among the best in the dataset for that skill.
        """
    )
    all_scaled_metrics_defenders = [m + '_scaled' for m in raw_unscaled_metrics + pca_metrics + composite_metrics]

    all_scaled_metrics_for_table = [col for col in all_scaled_metrics_defenders if col in df.columns]
    if all_scaled_metrics_for_table:
        role_means = filtered_df.groupby('Tactical Role')[all_scaled_metrics_for_table].mean().round(2)
        role_means.columns = [col.replace('_scaled', '') for col in role_means.columns]
        st.dataframe(role_means)
    else:
        st.info("ℹ️ Please select at least one metric to display role averages.")

    st.subheader("🧭 Combined Radar Chart (Percentile)")
    if scaled_user_metrics:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(scaled_user_metrics), endpoint=False).tolist()
        angles += [angles[0]]
        colors = ["#1f77b4", "#ff7f0e"]
        
        for idx, player in enumerate(selected_players):
            player_data = df[df['Player Name'] == player]
            if not player_data.empty:
                player_row = player_data.iloc[0]
                values = player_row[scaled_user_metrics].fillna(0).values.astype(float)
                values = np.concatenate((values, [values[0]]))

                color = colors[idx % len(colors)]
                ax.plot(angles, values, label=player, linewidth=2, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_scaled', '') for m in scaled_user_metrics])
        ax.set_ylim(0, 110)
        ax.set_title("🧭 Tactical Profile Comparison", size=14, pad=20)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select at least one player to generate comparison radar chart.")

    st.subheader("📊 Player Metric Comparison")
    if scaled_user_metrics:
        compare_df = df[df['Player Name'].isin(selected_players)]
        melted = compare_df.melt(id_vars=['Player Name'], value_vars=scaled_user_metrics, var_name='Metric', value_name='Value')
        melted['Metric'] = melted['Metric'].str.replace('_scaled', '', regex=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#1f77b4", "#ff7f0e"]
        sns.barplot(data=melted, x='Value', y='Metric', hue='Player Name', palette=colors[:len(selected_players)], dodge=True, ax=ax)
        ax.set_title("📊 Scaled Metric Comparison", fontsize=14)
        ax.set_xlabel("Value (Percentile)")
        ax.set_ylabel("Metric")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select players to generate comparison bar charts.")

    st.subheader("🧬 Similar Player Finder")
    target_player = st.selectbox("Choose Player to Find Similar Profiles", sorted(players), key="defender_similar_player")
    if target_player and scaled_user_metrics:
        similar_df = find_similar_players(df, target_player, scaled_user_metrics)
        if not similar_df.empty:
            st.markdown(f"_Showing top 5 players most similar to **{target_player}** based on selected metrics._")
            st.dataframe(similar_df.style.format({'Similarity': '{:.2f}'}))
        else:
            st.info("ℹ️ Not enough data to calculate similarity for that player.")
    elif target_player and not scaled_user_metrics:
        st.info("ℹ️ Please select metrics in the sidebar to find similar players.")

def render_midfielders_dashboard(df):
    """
    Renders the Streamlit dashboard for midfielders.
    """
    # Capitalize player names and Clubs
    df['Player Name'] = df['Player Name'].str.title()

    df['Club'] = df['Club'].str.title()
    
    # Explanations for tactical roles and PCA components
    tactical_roles_desc = {
        'Deep-lying Playmaker': 'A midfielder who operates from a deep position, controlling the tempo of the game and initiating attacks with precise passing.',
        'Final-Thrird Playmaker': 'A creative midfielder who operates primarily in advanced zones, just outside or inside the opponent’s penalty area. Their primary function is to unlock defensive blocks through incisive passing, intelligent positioning, and spatial awareness.' ,
        'Positional Anchor': 'A defensive-minded midfielder who stays in a central position to protect the backline and break up opposition attacks.'
    }

    pca_explanations = {
        'Defensive Involvement & Ball Recovery': 'Quantifies a midfielder\'s effort in winning the ball and retrieving possession in all areas of the pitch.',
        'Defensive Involvement & Consistency': 'Measures a player\'s reliability and sustained effort in defensive duties throughout a match.',
        'Final Third/Attacking Contribution': 'Reflects a midfielder\'s impact in the attacking phase, including shots, goals, and passes in dangerous areas.',
        'Defensive Pressing': 'Assesses how proactively a player presses opponents to force turnovers or disrupt their buildup.',
        'Build-Up & Offensive Contribution': 'Describes a player\'s role in progressing the ball from defense to attack, linking play and maintaining possession.',
        'Passing & Technical Quality': 'Highlights a midfielder\'s accuracy and skill in passing, including a variety of pass types and successful dribbles.',
        'Intensity & Ball-Carrying': 'Measures a player\'s work rate and ability to carry the ball forward, driving the team up the pitch.'
    }
    
    raw_metrics = [
        'Goals per 90', 'Shots per 90', 'Assists per 90', 'Duels Won per 90 (%)',
        'Dribbles per 90 (%)', 'Key Passes per 90', 'Total Passes per 90',
        'Pass Accuracy (%)', 'Tackles per 90', 'Interceptions per 90'
    ]
    pca_metrics = [
        'Defensive Involvement & Ball Recovery', 'Defensive Involvement & Consistency',
        'Final Third/Attacking Contribution', 'Defensive Pressing',
        'Build-Up & Offensive Contribution', 'Passing & Technical Quality',
        'Intensity & Ball-Carrying'
    ]
    
    # Scale all raw and PCA metrics
    all_metrics_to_scale = raw_metrics + pca_metrics
    for metric in all_metrics_to_scale:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            df[metric + '_scaled'] = df[metric].rank(pct=True, method='dense') * 100

    all_scaled_metrics_midfielders = [metric + '_scaled' for metric in all_metrics_to_scale if metric + '_scaled' in df.columns]

    
    roles = df['Tactical Role'].dropna().unique()
    players = df['Player Name'].dropna().unique()

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Midfielders Dashboard Filters")
    selected_roles = st.sidebar.multiselect("Select Role(s)", sorted(roles), default=list(roles), key="midfielder_role_select")
    selected_players = st.sidebar.multiselect("Compare Players (Radar + Bar)", sorted(players), default=[], key="midfielder_player_compare")
    single_player_filter = st.sidebar.selectbox("Filter Table by Player", ["All"] + sorted(players), key="midfielder_single_player_filter")
    
    metric_type_choice = st.sidebar.radio(
        "Choose Metric Type",
        ("Individual Key Performance Areas", "Raw Unscaled Metrics"),
        key="midfielders_metric_type"
    )

    if metric_type_choice == "Individual Key Performance Areas":
        metric_options = pca_metrics
        default_metrics = pca_metrics
    else: # "Raw Unscaled Metrics"
        metric_options = raw_metrics
        default_metrics = raw_metrics
    
    user_metrics = st.sidebar.multiselect(
        "📊 Choose Metrics for Radar & Comparison",
        [m for m in metric_options if m + '_scaled' in df.columns],
        default=default_metrics,
        key="midfielders_user_metrics"
    )
    scaled_user_metrics = [m + '_scaled' for m in user_metrics]

    filtered_df = df[df['Tactical Role'].isin(selected_roles)]
    if single_player_filter != "All":
        filtered_df = filtered_df[filtered_df['Player Name'] == single_player_filter]

    st.title("📊 Tactical Dashboard for Midfielders")

    with st.expander("ℹ️ Tactical Role & Key Performance Areas Explanations"):
        st.markdown("### 🔑 Role Labels")
        for role, desc in tactical_roles_desc.items():
            st.markdown(f"- **{role}**: {desc}")
        
        st.markdown("### 📈 Individual Key Performance Areas")
        for pca, desc in pca_explanations.items():
            st.markdown(f"- **{pca}**: {desc}")

    st.subheader("📋 Player Table")
    # All metrics for table display
    
    
    table_columns = ['Player Name', 'League', 'Club', 'Position', 'Tactical Role', 'Minutes Played']

    raw_unscaled_metrics = raw_metrics  # Alias for clarity

    raw_metrics_for_table = table_columns + [col for col in raw_unscaled_metrics + pca_metrics if col in df.columns]

    st.dataframe(filtered_df[raw_metrics_for_table].round(2))

    st.subheader("Average Metric Ranks by Role (0-100)")
    st.markdown(
        """
        **What these numbers mean:** This table shows the **average percentile rank** for each metric within a specific role. A percentile rank is a value from **0 to 100**, indicating how a player performs relative to all other players in the dataset.

        **Example:** A value of `29.35` for `Goals per 90` means the average player in this role performs better than only `29.35%` of all players in that metric. This is expected for roles like a Defender, where scoring is not a primary function. A high number, such as `95.00` for a metric like `Tackles Success %`, would show that the average player in this role is among the best in the dataset for that skill.
        """
    )
    
    all_scaled_metrics_for_table = [col for col in all_scaled_metrics_midfielders if col in df.columns]
    if all_scaled_metrics_for_table:
        role_means = filtered_df.groupby('Tactical Role')[all_scaled_metrics_for_table].mean().round(2)
        role_means.columns = [col.replace('_scaled', '') for col in role_means.columns]
        st.dataframe(role_means)
    else:
        st.info("ℹ️ Please select at least one metric to display role averages.")
        
    st.subheader("🧭 Combined Radar Chart (Percentile)")
    if scaled_user_metrics:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(scaled_user_metrics), endpoint=False).tolist()
        angles += [angles[0]]
        colors = ["#1f77b4", "#ff7f0e"]

        for idx, player in enumerate(selected_players):
            player_data = df[df['Player Name'] == player]
            if not player_data.empty:
                player_row = player_data.iloc[0]
                values = player_row[scaled_user_metrics].fillna(0).values.astype(float)
                values = np.concatenate((values, [values[0]]))

                color = colors[idx % len(colors)]
                ax.plot(angles, values, label=player, linewidth=2, color=color)
                ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_scaled', '') for m in scaled_user_metrics])
        ax.set_ylim(0, 110)
        ax.set_title("🧭 Tactical Profile Comparison", size=14, pad=20)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select at least one player to generate comparison radar chart.")
    
    st.subheader("📊 Player Metric Comparison")
    if scaled_user_metrics:
        compare_df = df[df['Player Name'].isin(selected_players)]
        melted = compare_df.melt(id_vars=['Player Name'], value_vars=scaled_user_metrics, var_name='Metric', value_name='Value')
        melted['Metric'] = melted['Metric'].str.replace('_scaled', '', regex=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#1f77b4", "#ff7f0e"]
        sns.barplot(data=melted, x='Value', y='Metric', hue='Player Name', palette=colors[:len(selected_players)], dodge=True, ax=ax)
        ax.set_title("📊 Scaled Metric Comparison", fontsize=14)
        ax.set_xlabel("Value (Percentile)")
        ax.set_ylabel("Metric")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select players to generate comparison bar charts.")
        
    st.subheader("🧬 Similar Player Finder")
    target_player = st.selectbox("Choose Player to Find Similar Profiles", sorted(players), key="midfielder_similar_player")
    if target_player and scaled_user_metrics:
        similar_df = find_similar_players(df, target_player, scaled_user_metrics)
        if not similar_df.empty:
            st.markdown(f"_Showing top 5 players most similar to **{target_player}** based on selected metrics._")
            st.dataframe(similar_df.style.format({'Similarity': '{:.2f}'}))
        else:
            st.info("ℹ️ Not enough data to calculate similarity for that player.")
    elif target_player and not scaled_user_metrics:
        st.info("ℹ️ Please select metrics in the sidebar to find similar players.")

def render_forwards_dashboard(df):
    """
    Renders the Streamlit dashboard for forwards.
    """
    # Pre-processing for forwards
    df = df.rename(columns={
        'name': 'Player Name',
        'league': 'League',
        'team_clean': 'Club',
        'position': 'Position',
        'passes_accuracy_perc': 'Passes accuracy %',
        'Role_Label': 'Tactical Role',
        'PC1': 'Playmaker Influence',
        'PC2': 'Finishing Threat',
        'PC3': 'Technical Agility',
        'PC4': 'Possession Efficiency',
        'PC5': 'Tactical Discipline',
        'PC6': 'Pressured Contribution',
        'PC7': 'Role Engagement'
    })

    # Capitalize player and club names
    df['Player Name'] = df['Player Name'].str.title()
    df['League'] = df['League'].str.title()


    df['Club'] = df['Club'].str.title()

    correction_metrics = ['Dribbles successful %_p90', 'Duels won %', 'Passes accuracy %']
    for metric in correction_metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
           df[metric] = df[metric] / 100

    # Define metrics
    pca_metrics = [
        'Playmaker Influence', 'Finishing Threat', 'Technical Agility',
        'Possession Efficiency', 'Tactical Discipline',
        'Pressured Contribution', 'Role Engagement'
    ]
    raw_metrics = [
        'Goals_p90', 'Assists_p90', 'Shots_p90',
        'Dribbles successful %_p90', 'Duels won %',
        'Key Passes', 'Passes total p90', 'Passes accuracy %'
    ]
    all_metrics_forwards = raw_metrics + pca_metrics

    # Convert percentage metrics from decimals to proper percentages
    percentage_metrics = ['Dribbles successful %_p90', 'Duels won %', 'Passes accuracy %']
    for metric in percentage_metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            df[metric] = df[metric] * 100

    # Scale metrics
    for metric in all_metrics_forwards:
        if metric in df.columns:
            df[metric + '_scaled'] = df[metric].rank(pct=True) * 100

    # Define scaled metric list
    all_scaled_metrics_forwards = [m + '_scaled' for m in all_metrics_forwards if m + '_scaled' in df.columns]

    roles = df['Tactical Role'].dropna().unique()
    players = df['Player Name'].dropna().unique()

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Forwards Dashboard Filters")
    selected_roles = st.sidebar.multiselect("Select Role(s)", sorted(roles), default=list(roles), key="forward_roles_select")
    selected_players = st.sidebar.multiselect("Compare Players (Radar + Bar)", sorted(players), default=[], key="forward_player_compare")
    single_player_filter = st.sidebar.selectbox("Filter Table by Player", ["All"] + sorted(players), key="forward_single_player_filter")

    metric_type_choice = st.sidebar.radio(
        "Choose Metric Type",
        ("Raw Scaled Metrics", "Key Performance Areas"),
        key="forwards_metric_type"
    )

    if metric_type_choice == "Raw Scaled Metrics":
        metric_options = raw_metrics
        default_metrics = raw_metrics
    else:
        metric_options = pca_metrics
        default_metrics = pca_metrics

    user_metrics = st.sidebar.multiselect(
        "📊 Choose Metrics for Radar & Comparison",
        [m for m in metric_options if m + '_scaled' in df.columns],
        default=default_metrics,
        key="forward_user_metrics"
    )
    scaled_user_metrics = [m + '_scaled' for m in user_metrics]

    filtered_df = df[df['Tactical Role'].isin(selected_roles)]
    if single_player_filter != "All":
        filtered_df = filtered_df[filtered_df['Player Name'] == single_player_filter]

    st.title("📊 Tactical Dashboard for Forwards")

    with st.expander("ℹ️ Tactical Role & Key Performance Areas Explanations"):
        st.markdown("""
        ### 🔑 Role Labels
        - **Channel Runner**: Creates depth through movement in narrow or counter systems.
        - **Pure Finisher**: Box predator focused on conversion, low buildup/pass involvement.
        - **Creative Technician**: Links play and breaks defensive blocks through technique.
        - **Disruptive Presser**: Aggressive in pressing, contributing tactically off-ball.

        ### 📈 Key Performance Areas
        - **Playmaker Influence**: Touches + progressive passing in creative zones.
        - **Finishing Threat**: xG, shot frequency, proximity to goal.
        - **Technical Agility**: Quick control, dribble success under pressure.
        - **Possession Efficiency**: Low turnover rate while retaining ball.
        - **Tactical Discipline**: Consistent positional & stylistic traits.
        - **Pressured Contribution**: Output under high pressure/defensive phases.
        - **Role Engagement**: Tactical fit, on-ball involvement.
        """)

    st.subheader("📋 Player Table")
    table_columns = ['Player Name', 'League', 'Club', 'Position', 'Tactical Role', 'Minutes Played']
    raw_metrics_for_table = table_columns + [col for col in raw_metrics + pca_metrics if col in df.columns]
    st.dataframe(filtered_df[raw_metrics_for_table].round(2))

    st.subheader("Average Metric Ranks by Role (0-100)")
    st.markdown("""
        **What these numbers mean:** This table shows the **average percentile rank** for each metric within a specific role. A percentile rank is a value from **0 to 100**, indicating how a player performs relative to all other players in the dataset.
    """)
    all_scaled_metrics_for_table = [col for col in all_scaled_metrics_forwards if col in df.columns]
    if all_scaled_metrics_for_table:
        role_means = filtered_df.groupby('Tactical Role')[all_scaled_metrics_for_table].mean().round(2)
        role_means.columns = [col.replace('_scaled', '') for col in role_means.columns]
        st.dataframe(role_means)
    else:
        st.info("ℹ️ Please select at least one metric to display role averages.")

    st.subheader("🧭 Combined Radar Chart (Percentile)")
    preview_players = selected_players if selected_players else players[:2]
    if scaled_user_metrics:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(scaled_user_metrics), endpoint=False).tolist()
        angles += [angles[0]]
        colors = ["#1f77b4", "#ff7f0e"]

        for idx, player in enumerate(preview_players):
            player_data = df[df['Player Name'] == player]
            if player_data.empty:
                st.warning(f"⚠️ Player '{player}' not found.")
                continue

            player_row = player_data.iloc[0]
            values = player_row[scaled_user_metrics].fillna(0).values.astype(float)
            values = np.concatenate((values, [values[0]]))

            color = colors[idx % len(colors)]
            ax.plot(angles, values, label=player, linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_scaled', '').replace('_p90', '').replace('%', '').title() for m in scaled_user_metrics])
        ax.set_ylim(0, 110)
        ax.set_title("🧭 Tactical Profile Comparison", size=14, pad=20)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select at least one player to generate comparison radar chart.")

    st.subheader("📊 Player Metric Comparison")
    if scaled_user_metrics:
        compare_df = df[df['Player Name'].isin(preview_players)]
        melted = compare_df.melt(id_vars=['Player Name'], value_vars=scaled_user_metrics, var_name='Metric', value_name='Value')
        melted['Metric'] = melted['Metric'].str.replace('_scaled', '', regex=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=melted, x='Value', y='Metric', hue='Player Name', palette=colors[:len(preview_players)], dodge=True, ax=ax)
        ax.set_title("📊 Scaled Metric Comparison", fontsize=14)
        ax.set_xlabel("Value (Percentile)")
        ax.set_ylabel("Metric")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select players to generate comparison bar charts.")

    st.subheader("🧬 Similar Player Finder")
    target_player = st.selectbox("Choose Player to Find Similar Profiles", sorted(players), key="forward_similar_player")
    if target_player and scaled_user_metrics:
        similar_df = find_similar_players(df, target_player, scaled_user_metrics)
        if not similar_df.empty:
            st.markdown(f"_Showing top 5 players most similar to **{target_player}** based on selected metrics._")
            st.dataframe(similar_df.style.format({'Similarity': '{:.2f}'}))
        else:
                    
            st.info("ℹ️ Not enough data to calculate similarity for that player.")
    elif target_player and not scaled_user_metrics:
        st.info("ℹ️ Please select metrics in the sidebar to find similar players.")




def main():
    st.sidebar.title("Player Dashboard")
    dashboard_choice = st.sidebar.radio("Select Dashboard", ["Defenders", "Midfielders", "Forwards"])

    if dashboard_choice == "Defenders":
        df = load_data("defenders_processed_dataset_for_app_ready.csv")
        if not df.empty:
            render_defenders_dashboard(df)
    elif dashboard_choice == "Midfielders":
        df = load_data("global_midfielders_2023_24_profiled_final.csv")
        if not df.empty:
            render_midfielders_dashboard(df)
    else: # Forwards
        df = load_data("global_forwards_2023_24_app_ready_with_raw_metrics.csv")
        if not df.empty:
            render_forwards_dashboard(df)

if __name__ == "__main__":
    main()



