"""
app.py — WC2026 ML Forecast Dashboard
A clean, professional dashboard for exploring the 2026 FIFA World Cup model.
"""
from __future__ import annotations
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.utils import flag_img_html, flag_label, get_pmb, load_forecast
from core.config import WC2026_TEAMS, WC2026_GROUPS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC2026 Forecast",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit padding */
    .block-container { padding-top: 1.5rem; }

    /* Metric card style */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #2d2d4e;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .metric-card .label {
        font-size: 0.78em;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 1.7em;
        font-weight: 700;
        color: #fff;
        line-height: 1.1;
    }
    .metric-card .sub {
        font-size: 0.82em;
        color: #4ade80;
        margin-top: 4px;
    }

    /* Match predictor team row */
    .team-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 20px;
        background: #16213e;
        border-radius: 10px;
        margin: 6px 0;
    }
    .prob-bar-wrap { flex: 1; background: #0d0d1a; border-radius: 6px; height: 8px; }
    .prob-bar { height: 8px; border-radius: 6px; }

    /* Section divider */
    .section-title {
        font-size: 1.1em;
        font-weight: 600;
        color: #a0a0c0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 20px 0 10px 0;
    }

    /* Recommended badge */
    .badge-rec {
        display: inline-block;
        background: #14532d;
        color: #4ade80;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75em;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    .badge-draw {
        display: inline-block;
        background: #1e293b;
        color: #94a3b8;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75em;
        margin-left: 8px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _get_pmb():
    return get_pmb()

pmb = _get_pmb()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'{flag_img_html("USA", 28)}{flag_img_html("Canada", 28)}{flag_img_html("Mexico", 28)}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.title("WC 2026")
    st.caption("Machine-learning forecast")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Overview", "Match Predictor", "Championship Odds", "Team Explorer", "Dark Horse Analysis", "Methodology"],
    )

    st.markdown("---")
    method = st.selectbox(
        "Resolution method",
        ["coinflip", "proportional", "historical_hybrid", "hybrid_57_5"],
        help="How 90-min draws are resolved in the knockout stage simulator.",
    )
    st.caption(f"50,000 Monte Carlo simulations")

df = load_forecast(method)


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "", width: str = "100%") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card" style="width:{width}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )

def flag_team_html(team: str, size: int = 24) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:8px;">'
        f'{flag_img_html(team, size)}'
        f'<span style="font-weight:600">{team}</span>'
        f'</span>'
    )

def pct_bar_html(pct: float, color: str = "#3b82f6") -> str:
    return (
        f'<div class="prob-bar-wrap">'
        f'<div class="prob-bar" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## 2026 FIFA World Cup Forecast")
    st.markdown("Machine-learning powered tournament simulation. 48 teams · 12 groups · 50,000 simulations.")
    st.markdown("---")

    if df.empty:
        st.error("Forecast data not found. Run `main.py` first.")
    else:
        top1 = df.iloc[0]
        top2 = df.iloc[1]
        top3_sum = df.head(3)["champion_pct"].sum()

        # ── Hero metrics row ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card(
                "Tournament Favourite",
                f'{flag_img_html(top1["team"], 28)} {top1["team"]}',
                f'{top1["champion_pct"]:.1f}% to win the World Cup',
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card(
                "Most Likely Final",
                f'{top1["team"]} vs {top2["team"]}',
                f'{top1["finalist_pct"]:.0f}% / {top2["finalist_pct"]:.0f}% final probability',
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card(
                "Top-3 Combined Win %",
                f'{top3_sum:.1f}%',
                f'{top1["team"]}, {top2["team"]}, {df.iloc[2]["team"]}',
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card(
                "Simulations Run",
                "50,000",
                f'Method: {method}',
            ), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Top 12 Contenders")

        top12 = df.head(12).copy()

        # Flag column (inline HTML rendered separately; Altair uses text label)
        top12["Team"] = top12["team"]

        chart = (
            alt.Chart(top12)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("champion_pct:Q", title="Championship Probability (%)"),
                y=alt.Y("Team:N", sort="-x", title=None),
                color=alt.Color(
                    "champion_pct:Q",
                    scale=alt.Scale(scheme="blues", domain=[0, top12["champion_pct"].max()]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Team", title="Team"),
                    alt.Tooltip("champion_pct", title="Win %", format=".2f"),
                    alt.Tooltip("finalist_pct", title="Final %", format=".2f"),
                    alt.Tooltip("semifinal_pct", title="SF %", format=".2f"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

        # Team cards row under chart
        st.markdown("### Quick Stats")
        cols = st.columns(6)
        for i, row in df.head(6).iterrows():
            with cols[i]:
                st.markdown(
                    f'<div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:10px;padding:14px;text-align:center">'
                    f'{flag_img_html(row["team"], 40)}'
                    f'<div style="font-weight:700;margin-top:6px;font-size:0.95em">{row["team"]}</div>'
                    f'<div style="color:#3b82f6;font-size:1.2em;font-weight:700;margin-top:4px">{row["champion_pct"]:.1f}%</div>'
                    f'<div style="color:#888;font-size:0.7em">to win</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Match Predictor":
    st.markdown("## Match Predictor")
    st.markdown("90-minute outcome probabilities from the XGBoost model. Assumes neutral venue.")
    st.markdown("---")

    sorted_teams = sorted(WC2026_TEAMS)
    c1, _, c2 = st.columns([2, 1, 2])
    with c1:
        team_a = st.selectbox("Team A", sorted_teams, index=sorted_teams.index("Argentina"))
    with c2:
        team_b = st.selectbox("Team B", sorted_teams, index=sorted_teams.index("France"))

    if team_a == team_b:
        st.warning("Select two different teams.")
    else:
        ia, ib = pmb.team_idx[team_a], pmb.team_idx[team_b]
        pw  = float(pmb.P_win[ia, ib])  * 100
        pd_ = float(pmb.P_draw[ia, ib]) * 100
        pl  = float(pmb.P_loss[ia, ib]) * 100
        pko = float(pmb.P_ko[ia, ib])   * 100
        recommended = team_a if pko >= 50 else team_b

        st.markdown("---")

        # ── Match header ──────────────────────────────────────────────────────
        h1, h2, h3 = st.columns([3, 1, 3])
        with h1:
            st.markdown(
                f'<div style="text-align:center; padding:20px;">'
                f'{flag_img_html(team_a, 64)}'
                f'<div style="font-size:1.4em;font-weight:700;margin-top:8px">{team_a}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                '<div style="text-align:center;padding:30px 0;font-size:1.6em;color:#555;font-weight:300">vs</div>',
                unsafe_allow_html=True,
            )
        with h3:
            st.markdown(
                f'<div style="text-align:center; padding:20px;">'
                f'{flag_img_html(team_b, 64)}'
                f'<div style="font-size:1.4em;font-weight:700;margin-top:8px">{team_b}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability metrics ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{team_a} Win",  f"{pw:.1f}%")
        m2.metric("Draw",            f"{pd_:.1f}%")
        m3.metric(f"{team_b} Win",  f"{pl:.1f}%")
        m4.metric("Recommended",    recommended, f"KO win: {pko:.1f}%")

        st.markdown("---")

        # ── Probability bar chart ─────────────────────────────────────────────
        bar_df = pd.DataFrame({
            "Outcome": [f"{team_a} Win", "Draw", f"{team_b} Win"],
            "Probability (%)": [pw, pd_, pl],
            "Color": ["Team A", "Draw", "Team B"],
        })
        chart = (
            alt.Chart(bar_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Outcome:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13)),
                y=alt.Y("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "Color:N",
                    scale=alt.Scale(
                        domain=["Team A", "Draw", "Team B"],
                        range=["#3b82f6", "#64748b", "#f97316"],
                    ),
                    legend=None,
                ),
                tooltip=["Outcome", alt.Tooltip("Probability (%)", format=".1f")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        # ── Knockout context ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Knockout Stage Context")
        st.info(
            f"If this match were a knockout fixture, the draw probability ({pd_:.1f}%) would be "
            f"resolved via the **{method}** method.\n\n"
            f"**{team_a}** knockout win probability: **{pko:.1f}%** · "
            f"**{team_b}**: **{100-pko:.1f}%**"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAMPIONSHIP ODDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Championship Odds":
    st.markdown(f"## Championship Odds  ·  *{method}*")
    st.markdown("Probability of winning the 2026 FIFA World Cup based on 50,000 simulations.")
    st.markdown("---")

    if df.empty:
        st.error("No forecast data found.")
    else:
        top20 = df.head(20).copy()
        top20["Team"] = top20["team"]

        chart = (
            alt.Chart(top20)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("champion_pct:Q", title="Championship Probability (%)"),
                y=alt.Y("Team:N", sort="-x", title=None),
                color=alt.Color(
                    "champion_pct:Q",
                    scale=alt.Scale(scheme="blues"),
                    legend=None,
                ),
                tooltip=[
                    "Team",
                    alt.Tooltip("champion_pct",    title="Win %",     format=".2f"),
                    alt.Tooltip("finalist_pct",    title="Final %",   format=".2f"),
                    alt.Tooltip("semifinal_pct",   title="SF %",      format=".2f"),
                    alt.Tooltip("quarterfinal_pct",title="QF %",      format=".2f"),
                ],
            )
            .properties(height=620)
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.markdown("### Full Table")

        # Build display table with flag HTML per row
        rows = []
        for _, row in df.head(20).iterrows():
            rows.append({
                "Rank": int(row["rank"]),
                "Team": row["team"],
                "Win %": f"{row['champion_pct']:.2f}%",
                "Final %": f"{row['finalist_pct']:.2f}%",
                "SF %": f"{row['semifinal_pct']:.2f}%",
                "QF %": f"{row['quarterfinal_pct']:.2f}%",
                "Group Exit %": f"{row['group_exit_pct']:.1f}%",
            })

        tbl = pd.DataFrame(rows)

        # Flag column as HTML
        flag_col = [
            flag_img_html(r["team"], 20)
            for _, r in df.head(20).iterrows()
        ]

        st.dataframe(
            tbl.style
            .background_gradient(cmap="Blues", subset=["Rank"])
            .set_properties(**{"text-align": "left"}),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TEAM EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Team Explorer":
    st.markdown("## Team Explorer")
    st.markdown("Deep dive into any team's tournament probabilities.")
    st.markdown("---")

    if df.empty:
        st.error("No forecast data.")
    else:
        team = st.selectbox("Select a team", sorted(df["team"].tolist()))
        row = df[df["team"] == team].iloc[0]

        # ── Team header ────────────────────────────────────────────────────────
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:20px;padding:20px;'
            f'background:#1a1a2e;border-radius:12px;margin-bottom:16px;">'
            f'{flag_img_html(team, 72)}'
            f'<div>'
            f'<div style="font-size:2em;font-weight:700">{team}</div>'
            f'<div style="color:#888;font-size:0.9em">Rank #{int(row["rank"])} in WC2026 forecast  ·  Method: {method}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Six metric cards ───────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Champion %",    f'{row["champion_pct"]:.2f}%'), unsafe_allow_html=True)
            st.markdown(metric_card("Quarter-Final %", f'{row["quarterfinal_pct"]:.2f}%'), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Final %",       f'{row["finalist_pct"]:.2f}%'), unsafe_allow_html=True)
            st.markdown(metric_card("Round of 16 %", f'{row["r16_pct"]:.2f}%'), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Semi-Final %",  f'{row["semifinal_pct"]:.2f}%'), unsafe_allow_html=True)
            st.markdown(metric_card("Group Exit %",  f'{row["group_exit_pct"]:.1f}%', "Probability of not advancing"), unsafe_allow_html=True)

        # ── Progression funnel chart ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Progression Funnel")

        prog = pd.DataFrame({
            "Stage": ["Advance R32", "Advance R16", "Advance QF", "Advance SF", "Reach Final", "Win Final"],
            "Probability (%)": [
                row["r32_pct"],
                row["r16_pct"],
                row["quarterfinal_pct"],
                row["semifinal_pct"],
                row["finalist_pct"],
                row["champion_pct"],
            ],
        })
        prog["Stage"] = pd.Categorical(prog["Stage"], categories=prog["Stage"].tolist(), ordered=True)

        funnel = (
            alt.Chart(prog)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Stage:O", sort=None, title=None),
                color=alt.Color(
                    "Probability (%):Q",
                    scale=alt.Scale(scheme="greens"),
                    legend=None,
                ),
                tooltip=["Stage", alt.Tooltip("Probability (%)", format=".2f")],
            )
            .properties(height=280)
        )
        st.altair_chart(funnel, use_container_width=True)

        # ── Peer comparison ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Peer Comparison")
        st.caption("How does this team compare to the full field?")

        peer_df = df[["team", "champion_pct", "finalist_pct", "semifinal_pct"]].copy()
        peer_df["highlight"] = peer_df["team"] == team
        peer_df["Team"] = peer_df["team"]

        scatter = (
            alt.Chart(peer_df)
            .mark_circle(size=80, opacity=0.8)
            .encode(
                x=alt.X("finalist_pct:Q", title="Final Probability (%)"),
                y=alt.Y("champion_pct:Q", title="Champion Probability (%)"),
                color=alt.condition(
                    alt.datum.highlight,
                    alt.value("#f97316"),   # highlight color
                    alt.value("#3b82f6"),
                ),
                size=alt.condition(
                    alt.datum.highlight,
                    alt.value(200),
                    alt.value(60),
                ),
                tooltip=["Team", alt.Tooltip("champion_pct", format=".2f"), alt.Tooltip("finalist_pct", format=".2f")],
            )
            .properties(height=340)
        )
        text = (
            alt.Chart(peer_df[peer_df["highlight"]])
            .mark_text(align="left", dx=10, dy=-4, fontWeight="bold", color="#f97316")
            .encode(
                x="finalist_pct:Q",
                y="champion_pct:Q",
                text="Team:N",
            )
        )
        st.altair_chart(scatter + text, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DARK HORSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dark Horse Analysis":
    st.markdown("## Dark Horse Analysis")
    st.markdown("Teams whose tournament potential exceeds their championship probability ranking.")
    st.markdown("---")

    if df.empty:
        st.error("No forecast data.")
    else:
        # Dark horses: high QF% but low rank
        dark = df[(df["rank"] > 8) & (df["quarterfinal_pct"] > 12)].copy()
        dark["Team"] = dark["team"]

        if dark.empty:
            st.info("No dark horses detected under this method and threshold.")
        else:
            st.markdown(f"**{len(dark)} teams** qualify as dark horses (ranked outside Top 8, QF probability > 12%).")
            st.markdown("---")

            # Cards
            cols = st.columns(min(3, len(dark)))
            for i, (_, row) in enumerate(dark.iterrows()):
                with cols[i % 3]:
                    st.markdown(
                        f'<div style="background:#1a1a2e;border:1px solid #2d2d4e;border-radius:12px;'
                        f'padding:20px;margin-bottom:12px;text-align:center;">'
                        f'{flag_img_html(row["team"], 52)}'
                        f'<div style="font-size:1.1em;font-weight:700;margin-top:10px">{row["team"]}</div>'
                        f'<div style="color:#888;font-size:0.75em;margin-bottom:10px">Ranked #{int(row["rank"])}</div>'
                        f'<div style="display:flex;justify-content:space-between;margin:4px 0">'
                        f'<span style="color:#aaa;font-size:0.8em">QF%</span>'
                        f'<span style="font-weight:700;color:#f97316">{row["quarterfinal_pct"]:.1f}%</span>'
                        f'</div>'
                        f'<div style="display:flex;justify-content:space-between;margin:4px 0">'
                        f'<span style="color:#aaa;font-size:0.8em">SF%</span>'
                        f'<span style="font-weight:700;color:#f59e0b">{row["semifinal_pct"]:.1f}%</span>'
                        f'</div>'
                        f'<div style="display:flex;justify-content:space-between;margin:4px 0">'
                        f'<span style="color:#aaa;font-size:0.8em">Win%</span>'
                        f'<span style="font-weight:700;color:#3b82f6">{row["champion_pct"]:.1f}%</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Method sensitivity ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Method Sensitivity")
        st.markdown("How much does champion probability shift across knockout resolution methods?")

        dfs = {}
        for m in ["coinflip", "proportional", "historical_hybrid"]:
            d = load_forecast(m)
            if not d.empty:
                dfs[m] = d[["team", "champion_pct"]].rename(columns={"champion_pct": m})

        if len(dfs) >= 2:
            merged = list(dfs.values())[0]
            for d in list(dfs.values())[1:]:
                merged = merged.merge(d, on="team")

            methods_available = [m for m in ["coinflip", "proportional", "historical_hybrid"] if m in merged.columns]
            melted = merged.head(10).melt(
                id_vars=["team"],
                value_vars=methods_available,
                var_name="Method",
                value_name="Champion %",
            )
            melted["Team"] = melted["team"]

            chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    x=alt.X("Method:N", title=None),
                    y=alt.Y("Champion %:Q"),
                    color=alt.Color("Method:N", scale=alt.Scale(scheme="set2")),
                    column=alt.Column("Team:N", sort=alt.SortField("Champion %", order="descending")),
                    tooltip=["Team", "Method", alt.Tooltip("Champion %", format=".2f")],
                )
                .properties(width=70, height=220)
            )
            st.altair_chart(chart)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.markdown("## Model Methodology")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### Model Architecture")
        st.markdown("""
**Algorithm:** XGBoost multi-class classifier (3 outputs: Home Win / Draw / Away Win)

**Training data:** All international matches 2010–2026 from the martj42 dataset (~12,000 matches)

**Time-aware CV:** 5-fold expanding window. Each fold trains on all past data and evaluates on future matches only. No look-ahead leakage.

**13 features (production model):**
- `elo_diff` — Elo rating differential
- `opp_adj_form_diff` — opponent-adjusted 5-match form score
- `match_type_enc` — tournament importance weight
- `neutral_venue` — home advantage correction
- 9 additional engineered features (squad value, travel fatigue, etc.)

**Key hyperparameters:**
- `n_estimators: 800` · `max_depth: 2` · `learning_rate: 0.10`
- `subsample: 0.60` · `min_child_weight: 10`
        """)

        st.markdown("---")
        st.markdown("### Knockout Draw Resolution")
        st.markdown("""
Because knockout matches cannot end in a draw, the simulator must resolve 90-minute draw probabilities into a binary winner. Four methods were tested:

| Method | Formula | Notes |
|---|---|---|
| **coinflip** *(default)* | `p_ko = p_win + p_draw × 0.50` | Closest to empirical ET/Pen data |
| **proportional** | `p_ko = p_win / (p_win + p_loss)` | Strongly biases top teams (+7.7pp for Spain) |
| **historical_hybrid** | `p_ko = p_win + p_draw × 0.412` | Derived from 34 real WC ET/Pen matches (1986–2022) |
| **hybrid_57_5** | `p_ko = p_win + p_draw × 0.575` | Conservative favourite edge |

**Empirical finding:** In 34 World Cup draws (1986–2022), the pre-match favourite won only **41.2%** of the time in ET/Penalties — *worse* than a 50/50 coinflip.
        """)

    with col2:
        st.markdown("### Resolution Method Impact")
        df_c = load_forecast("coinflip")
        df_p = load_forecast("proportional")

        if not df_c.empty and not df_p.empty:
            merged = df_c[["team", "champion_pct"]].merge(
                df_p[["team", "champion_pct"]], on="team", suffixes=("_coinflip", "_proportional")
            )
            merged["Delta (pp)"] = merged["champion_pct_coinflip"] - merged["champion_pct_proportional"]
            merged = merged.sort_values("Delta (pp)")

            chart = (
                alt.Chart(merged.head(12))
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Delta (pp):Q", title="Coinflip − Proportional (pp)"),
                    y=alt.Y("team:N", sort="x", title=None),
                    color=alt.condition(
                        alt.datum["Delta (pp)"] < 0,
                        alt.value("#ef4444"),
                        alt.value("#22c55e"),
                    ),
                    tooltip=["team", alt.Tooltip("Delta (pp)", format="+.2f")],
                )
                .properties(
                    height=340,
                    title="Champion % change: Proportional → Coinflip",
                )
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.markdown("### Performance Metrics")
        st.markdown("""
| Metric | Value |
|---|---|
| CV Log-Loss | *see cv_metrics.json* |
| Brier Score | *see model_performance.txt* |
| OOS WC Accuracy | *see oos_backtest.txt* |
| Simulations/sec | ~3,000 |
        """)
        st.caption("Run `main.py` to see full training metrics in the console.")