# defenders_app.py

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

# 📥 Load and clean dataset
def load_data():
    """
    Loads and cleans the defenders dataset.
    This function will load the CSV file and rename columns for clarity.
    """
    base_path = Path(__file__).resolve().parent
    file_path = base_path / "data" / "defenders_processed_dataset_for_app_ready.csv"

    if not file_path.exists():
        st.error(f"❌ File not found at: {file_path}")
        return pd.DataFrame(), [], []

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

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
    radar_metrics = [
        'Goals per 90', 'Shots', 'Assists per 90', 'Dribble Success %',
        'Total Pass Accuracy %', 'Tackles Success %', 'Dribblers Tackled',
        'Blocks per 90', 'Shots Blocked per 90', 'Passes Blocked per 90',
        'Clearances per 90', 'Errors per 90'
    ] + pca_metrics

    for metric in radar_metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            df[metric + '_scaled'] = df[metric].rank(pct=True, method='dense') * 100
    
    # 🆕 Grouping PCA metrics into 8 composite categories for a more detailed but still clean visualization
    df['High Pressing & Intensity_scaled'] = df[[
        'Aggressive High Block Engagement_scaled',
        'Defensive Energy & Effort Radar_scaled'
    ]].mean(axis=1)

    df['Mid-Zone Engagement_scaled'] = df[[
        'Mid-Third Engagements_scaled',
        'Area Variety in Tackling_scaled',
        'Overall Defensive Workload Index_scaled'
    ]].mean(axis=1)

    df['Build-up & Ball Security_scaled'] = df[[
        'Build-up Ability_scaled',
        'Passing Reliability Across Areas_scaled'
    ]].mean(axis=1)

    df['Progressive Passing_scaled'] = df[[
        'Vertical Passing / Line-Breaking_scaled',
        'Structured Passing in Tactical Mid-Blocks_scaled',
        'Long Passing Ability - Pressured Duel Success_scaled'
    ]].mean(axis=1)
    
    df['1v1 & Duel Success_scaled'] = df[[
        '1v1 Ability_scaled',
        'Low-Tempo Defensive Coverage_scaled'
    ]].mean(axis=1)
    
    df['Box & Area Defense_scaled'] = df[[
        'Compactness & Shot Blocks_scaled',
        'Area Clearance & Box Protection_scaled'
    ]].mean(axis=1)

    df['Transition Ability_scaled'] = df[[
        'Transition Ability (Interceptions-Passing)_scaled'
    ]].mean(axis=1)

    df['Error Management_scaled'] = df[[
        'Errors Under Pressure_scaled',
        'Defensive Stress Indicators & Rescue Actions_scaled'
    ]].mean(axis=1)
    
    composite_metrics = list(composite_groups.keys())

    st.success(f"✅ Loaded: {df.shape[0]} players, {df.shape[1]} features")
    return df, radar_metrics + composite_metrics, pca_metrics + composite_metrics

df, radar_metrics, pca_metrics = load_data()

if not df.empty:
    st.sidebar.header("🎛️ Filters")
    roles = df['Tactical Role'].dropna().unique()
    players = df['Player Name'].dropna().unique()

    selected_roles = st.sidebar.multiselect("Select Role(s)", sorted(roles), default=list(roles))
    selected_players = st.sidebar.multiselect("Compare Players (Radar + Bar)", sorted(players), default=[])
    single_player_filter = st.sidebar.selectbox("Filter Table by Player", ["All"] + sorted(players))
    
    composite_metrics_list = list(composite_groups.keys())

    user_metrics = st.sidebar.multiselect(
        "📊 Choose Metrics for Radar & Comparison",
        [m for m in radar_metrics if m + '_scaled' in df.columns],
        default=composite_metrics_list
    )
    scaled_user_metrics = [m + '_scaled' for m in user_metrics]

    filtered_df = df[df['Tactical Role'].isin(selected_roles)]
    if single_player_filter != "All":
        filtered_df = filtered_df[df['Player Name'] == single_player_filter]

    st.title("📊 Tactical Dashboard for Defenders")

    # 🆕 Updated the title of the expander section
    with st.expander("ℹ️ Tactical Role & Key Performance Areas Explanations"):
        st.markdown("### 🔑 Role Labels")
        for role, desc in tactical_roles_desc.items():
            st.markdown(f"- **{role}**: {desc}")
        
        st.markdown("### 📊 Key Performance Areas Groups")
        for group, pcas in composite_groups.items():
            st.markdown(f"- **{group}**: {composite_descriptions[group]} (Includes: " + ", ".join(pcas) + ")")
        
        st.markdown("### 📈 Individual Key Performance Areas")
        for pca, desc in pca_explanations.items():
            # Find the group for the current PCA
            group_name = next((group for group, pcas in composite_groups.items() if pca in pcas), "N/A")
            st.markdown(f"- **{pca}** (Group: *{group_name}*): {desc}")

    st.subheader("📋 Player Table")
    raw_columns = ['Player Name', 'League', 'Club', 'Position', 'Tactical Role'] + [col for col in radar_metrics if col in df.columns]
    
    valid_table_cols = [col for col in raw_columns if col in df.columns]
    st.dataframe(filtered_df[valid_table_cols].round(2))

    st.subheader("📈 Role-Level Scaled Metric Averages")
    st.markdown("_This table summarizes how each tactical role performs across the selected metrics, helping reveal typical strengths and stylistic patterns._")
    role_means = filtered_df.groupby('Tactical Role')[scaled_user_metrics].mean().round(2)
    role_means.columns = [col.replace('_scaled', '') for col in role_means.columns]
    st.dataframe(role_means)
    
    st.subheader("🧭 Combined Radar Chart (Percentile)")
    if selected_players:
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
    if selected_players:
        compare_df = df[df['Player Name'].isin(selected_players)]
        melted = compare_df.melt(
            id_vars=['Player Name'],
            value_vars=scaled_user_metrics,
            var_name='Metric',
            value_name='Value'
        )
        melted['Metric'] = melted['Metric'].str.replace('_scaled', '', regex=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#1f77b4", "#ff7f0e"] 
        sns.barplot(
            data=melted,
            x='Value',
            y='Metric',
            hue='Player Name',
            palette=colors[:len(selected_players)],
            dodge=True,
            ax=ax
        )
        ax.set_title("📊 Scaled Metric Comparison", fontsize=14)
        ax.set_xlabel("Value (Percentile)")
        ax.set_ylabel("Metric")
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select players to generate comparison bar charts.")

    st.subheader("🧬 Similar Player Finder")

    def find_similar_players(df, target_player, features, top_n=5):
        df_copy = df.dropna(subset=features).copy()
        if target_player not in df_copy['Player Name'].values:
            return pd.DataFrame()

        matrix = df_copy[features].values
        target_vector = df_copy[df_copy['Player Name'] == target_player][features].values
        scores = cosine_similarity(target_vector, matrix).flatten()

        df_copy['Similarity'] = scores
        similar_df = df_copy[df_copy['Player Name'] != target_player]
        return similar_df.sort_values(by='Similarity', ascending=False).head(top_n)[['Player Name', 'Club', 'League', 'Similarity']]

    target_player = st.selectbox("Choose Player to Find Similar Profiles", sorted(players))

    if target_player:
        similar_df = find_similar_players(df, target_player, scaled_user_metrics)
        if not similar_df.empty:
            st.markdown(f"_Showing top 5 players most similar to **{target_player}** based on selected metrics._")
            st.dataframe(similar_df.style.format({'Similarity': '{:.2f}'}))
        else:
            st.info("ℹ️ Not enough data to calculate similarity for that player.")
