# midfielders_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.metrics.pairwise import cosine_similarity
import io

# 🎨 Layout enhancements
st.markdown("""<style>
    .stPlotlyChart, .stPyplot {
        margin-left: auto;
        margin-right: auto;
    }
</style>""", unsafe_allow_html=True)


# Explanations for tactical roles and PCA components (Midfielder-specific)
tactical_roles_desc = {
    'Deep-lying Playmaker': 'These players thrive in possession-heavy roles: high passing volume, solid accuracy, and above-average physical metrics like duels and dribble efficiency. They\'re the midfield engines that keep circulation flowing — think holding pivots, build-up anchors, or low-block metronomes who rarely push into attacking zones but control tempo from deep.',
    'Positional Anchor': 'While they show minimal involvement across attacking, possession, or physical metrics, these players are likely deployed to hold structure, absorb transitions, and offer tactical balance. They\'re not progressors or disruptors — more like defensive shadows or safety nets, embedded in system shape rather than high-touch responsibilities.',
    'Final-Third Playmaker': 'High output across goals, assists, shot volume, and key passes. These are the creative hubs operating near the opposition box — roaming No.10s, wide linkers, or free-moving interiors who specialize in unlocking space and delivering end-product.'
}

pca_explanations = {
    'Defensive Involvement & Ball Recovery': 'Driven by interceptions, duels, tackles, and passing workload. This axis reveals players who consistently disrupt opposition play and recover possession. It represents a clear Ball-Winning Disruptor profile, tactically relevant for holding or hybrid midfielders.',
    'Defensive Involvement & Consistency': 'Combines consistent tackling, interception, and dueling metrics with match time. Strong signal of defensive effort and positional responsibility. It helps capture general engagement and complements Ball-Winning Disruptor without redundancy.',
    'Final Third/Attacking Contribution': 'Features shots, goals, attacking duels, and dribble attempts. Highlights forward-oriented midfielders. It\'s essential for identifying creators, attacking eights, and goal-threat midfielders.',
    'Defensive Pressing': 'Focused on per 90 metrics for tackles and interceptions. Signals reactive and high-frequency pressers. It’s a precise axis for profiling energetic midfielders in pressing or transition roles.',
    'Build-Up & Offensive Contribution': 'Correlates with passing volume, shot creation, goals, and dribble success. Blends progression with finishing actions. It offers a composite signal for offensive distributors and chance creators.',
    'Passing & Technical Quality': 'Driven by key passes, passing accuracy, goals, and dribble success. Strong marker of playmakers and deep distributors. It\'s valuable for stylistic separation between creative midfield roles.',
    'Intensity & Ball-Carrying': 'Includes duels, tackles, dribbles, and pass volume. Flags box-to-box midfielders and progressive carriers. It is tactically meaningful and useful for scouting high-workload profiles.'
}

# 📥 Load and clean dataset
def load_data():
    """
    Loads and cleans the midfielders dataset.
    This function will load the CSV file and rename columns for clarity.
    """
    # The file is expected to be in a 'data' sub-folder relative to the script
    base_path = Path(__file__).resolve().parent
    file_path = base_path / "data" / "global_midfielders_2023_24_profiled_final.csv"

    if not file_path.exists():
        st.error(f"❌ File not found at: {file_path}")
        return pd.DataFrame(), [], []

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Midfielder-specific metrics
    pca_metrics = list(pca_explanations.keys())
    raw_metrics = [
        'Goals per 90', 'Goals Total', 'Shots per 90', 'Shots Total', 'Assists per 90', 'Assists Total',
        'Duels Won per 90 (%)', 'Dribbles per 90 (%)', 'Key Passes per 90', 'Total Passes per 90',
        'Pass Accuracy (%)', 'Tackles per 90', 'Tackles Total', 'Interceptions per 90', 'Interceptions Total'
    ]
    
    radar_metrics = raw_metrics + pca_metrics
    
    for metric in radar_metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            df[metric + '_scaled'] = df[metric].rank(pct=True, method='dense') * 100
    
    st.success(f"✅ Loaded: {df.shape[0]} players, {df.shape[1]} features")
    return df, radar_metrics, pca_metrics

df, radar_metrics, pca_metrics = load_data()

if not df.empty:
    st.sidebar.header("🎛️ Filters")
    roles = df['Tactical Role'].dropna().unique()
    players = df['Player Name'].dropna().unique()

    selected_roles = st.sidebar.multiselect("Select Role(s)", sorted(roles), default=list(roles))
    selected_players = st.sidebar.multiselect("Compare Players (Radar + Bar)", sorted(players), default=[])
    single_player_filter = st.sidebar.selectbox("Filter Table by Player", ["All"] + sorted(players))
    
    user_metrics = st.sidebar.multiselect(
        "📊 Choose Metrics for Radar & Comparison",
        [m for m in radar_metrics if m + '_scaled' in df.columns],
        default=pca_metrics
    )
    scaled_user_metrics = [m + '_scaled' for m in user_metrics]

    filtered_df = df[df['Tactical Role'].isin(selected_roles)]
    if single_player_filter != "All":
        filtered_df = filtered_df[df['Player Name'] == single_player_filter]

    st.title("📊 Tactical Dashboard for Midfielders")

    with st.expander("ℹ️ Tactical Role & Key Performance Areas Explanations"):
        st.markdown("### 🔑 Role Labels")
        for role, desc in tactical_roles_desc.items():
            st.markdown(f"- **{role}**: {desc}")
        
        st.markdown("### 📈 Individual Key Performance Areas")
        for pca, desc in pca_explanations.items():
            st.markdown(f"- **{pca}**: {desc}")

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

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        
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
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
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
            st.info("ℹ️ Not enough data to calculate similarity for that player with the selected metrics.")
