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

# 📥 Load and clean dataset
def load_data():
    base_path = Path(__file__).resolve().parent
    file_path = base_path / "data" / "global_forwards_2023_24_app_ready_with_raw_metrics.csv"

    if not file_path.exists():
        st.error(f"❌ File not found at: {file_path}")
        return pd.DataFrame(), []

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        'name': 'Player Name',
        'team_clean': 'Club',
        'passes_accuracy_perc': 'Passes accuracy %',
        'Role_Label': 'Tactical role',
        'PC1': 'Playmaker Influence',
        'PC2': 'Finishing Threat',
        'PC3': 'Technical Agility',
        'PC4': 'Possession Efficiency',
        'PC5': 'Tactical Discipline',
        'PC6': 'Pressured Contribution',
        'PC7': 'Role Engagement'
    })

    radar_metrics = [
        'Goals_p90', 'Assists_p90', 'Shots_p90',
        'Dribbles successful %_p90', 'Duels won %_p90',
        'Key Passes', 'Passes total p90', 'Passes accuracy %',
        'Playmaker Influence', 'Finishing Threat', 'Technical Agility',
        'Possession Efficiency', 'Tactical Discipline',
        'Pressured Contribution', 'Role Engagement'
    ]

    for metric in radar_metrics:
        if metric in df.columns:
            df[metric + '_scaled'] = df[metric].rank(pct=True) * 100

    st.success(f"✅ Loaded: {df.shape[0]} players, {df.shape[1]} features")
    return df, radar_metrics

df, radar_metrics = load_data()

if not df.empty:
    st.sidebar.header("🎛️ Filters")
    roles = df['Tactical role'].dropna().unique()
    players = df['Player Name'].dropna().unique()

    selected_roles = st.sidebar.multiselect("Select Role(s)", sorted(roles), default=list(roles))
    selected_players = st.sidebar.multiselect("Compare Players (Radar + Bar)", sorted(players), default=[])
    single_player_filter = st.sidebar.selectbox("Filter Table by Player", ["All"] + sorted(players))

    user_metrics = st.sidebar.multiselect(
        "📊 Choose Metrics for Radar & Comparison",
        [m for m in radar_metrics if m + '_scaled' in df.columns],
        default=radar_metrics[:8]
    )
    scaled_user_metrics = [m + '_scaled' for m in user_metrics]

    filtered_df = df[df['Tactical role'].isin(selected_roles)]
    if single_player_filter != "All":
        filtered_df = filtered_df[df['Player Name'] == single_player_filter]

    st.title("📊 Tactical Dashboard for Forwards")

    with st.expander("ℹ️ Tactical Role & Key Performance Areas Explanations"):
        st.markdown("""
        ### 🔑 Role Labels
        - **Off-ball Disruptor**: Creates depth through movement in narrow or counter systems.
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
    table_columns = [
        'Player Name', 'league', 'Club', 'position', 'Tactical role',
        'Goals', 'Goals_p90', 'Duels won %', 'Duels won %_p90',
        'Shots', 'Shots_p90', 'Assists', 'Assists_p90',
        'Dribbles successful %', 'Dribbles successful %_p90',
        'Key Passes', 'Passes total p90', 'Passes accuracy %',
        'Playmaker Influence', 'Finishing Threat', 'Technical Agility',
        'Possession Efficiency', 'Tactical Discipline',
        'Pressured Contribution', 'Role Engagement'
    ]
    valid_table_cols = [col for col in table_columns if col in df.columns]
    st.dataframe(filtered_df[valid_table_cols].round(2))

    st.subheader("📈 Role-Level Scaled Metric Averages")
    st.markdown("_This table summarizes how each tactical role performs across the selected metrics, helping reveal typical strengths and stylistic patterns._")
    role_means = filtered_df.groupby('Tactical role')[scaled_user_metrics].mean().round(2)
    role_means.columns = [col.replace('_scaled', '') for col in role_means.columns]
    st.dataframe(role_means)

    # 🧭 Combined Radar Chart with Blue & Orange
    st.subheader("🧭 Combined Radar Chart (Percentile)")
    preview_players = selected_players if selected_players else players[:2]

    if len(preview_players) > 0:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(scaled_user_metrics), endpoint=False).tolist()
        angles += [angles[0]]

        colors = ["#1f77b4", "#ff7f0e"]  # Blue, Orange

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
        ax.set_xticklabels(user_metrics)
        ax.set_ylim(0, 110)
        ax.set_title("🧭 Tactical Profile Comparison", size=14, pad=20)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)
    else:
        st.info("ℹ️ Select at least one player to generate comparison radar chart.")

    # 📊 Bar Chart Comparison
    st.subheader("📊 Player Metric Comparison")
    compare_df = df[df['Player Name'].isin(preview_players)]
    if compare_df.empty:
        st.info("ℹ️ Select players to generate comparison bar charts.")
    else:
        melted = compare_df.melt(
            id_vars=['Player Name'],
            value_vars=scaled_user_metrics,
            var_name='Metric',
            value_name='Value'
        )
        melted['Metric'] = melted['Metric'].str.replace('_scaled', '', regex=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=melted,
            x='Value',
            y='Metric',
            hue='Player Name',
            palette=colors[:len(preview_players)],
            dodge=True,
            ax=ax
        )
        ax.set_title("📊 Scaled Metric Comparison", fontsize=14)
        st.pyplot(fig)

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
        return similar_df.sort_values(by='Similarity', ascending=False).head(top_n)[['Player Name', 'Club', 'league', 'Similarity']]

    target_player = st.selectbox("Choose Player to Find Similar Profiles",  sorted(players))

    if target_player:
        similar_df = find_similar_players(df, target_player, scaled_user_metrics)
        if not similar_df.empty:
            st.markdown(f"_Showing top 5 players most similar to **{target_player}** based on selected metrics._")
            st.dataframe(similar_df.style.format({'Similarity': '{:.2f}'}))
        else:
            st.info("ℹ️ Not enough data to calculate similarity for that player.")
