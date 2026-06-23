"""sim_bracket.py — Step 3: Interactive knockout bracket (R32 → Final)."""
from __future__ import annotations
import numpy as np
import streamlit as st
import altair as alt
import pandas as pd

from dashboard.utils import flag_img_html, get_pmb, load_forecast

ROUNDS = [
    ("r32",   "Round of 32",   16),
    ("r16",   "Round of 16",   8),
    ("qf",    "Quarter-Finals", 4),
    ("sf",    "Semi-Finals",   2),
    ("final", "Final",         1),
]

ROUND_KEYS   = [r[0] for r in ROUNDS]
ROUND_LABELS = {r[0]: r[1] for r in ROUNDS}
ROUND_GAMES  = {r[0]: r[2] for r in ROUNDS}


def _ko_probs(pmb, team_a: str, team_b: str):
    """Return (p_win_a, p_draw_a, p_loss_a) from the model matrices."""
    try:
        ia, ib = pmb.team_idx[team_a], pmb.team_idx[team_b]
        pw = float(pmb.P_win[ia, ib])
        pd_ = float(pmb.P_draw[ia, ib])
        pl = float(pmb.P_loss[ia, ib])
        pko = float(pmb.P_ko[ia, ib])
        recommended = team_a if pko >= 0.5 else team_b
        return pw, pd_, pl, pko, recommended
    except Exception:
        return 0.5, 0.0, 0.5, 0.5, team_a


def _simulate_match(pmb, team_a: str, team_b: str) -> str:
    """Use P_ko to pick a winner probabilistically."""
    try:
        ia, ib = pmb.team_idx[team_a], pmb.team_idx[team_b]
        pko = float(pmb.P_ko[ia, ib])
    except Exception:
        pko = 0.5
    rng = np.random.default_rng()
    return team_a if rng.random() < pko else team_b


def _advance_round(current: str, winners: list[str]):
    """Given winners of the current round, set them as participants in the next."""
    idx = ROUND_KEYS.index(current)
    if idx + 1 >= len(ROUND_KEYS):
        return
    next_round = ROUND_KEYS[idx + 1]
    n_next = ROUND_GAMES[next_round]

    # Pair consecutive winners: winner[0] vs winner[1], winner[2] vs winner[3], ...
    bracket_state = st.session_state.bracket_state
    if next_round == "final":
        bracket_state["final"] = None   # will be set when SF is complete
        # Store finalists separately so the final match can be rendered
        st.session_state["finalists"] = winners
    else:
        bracket_state[next_round] = [None] * (n_next)
        st.session_state["next_participants_" + next_round] = winners


def _get_participants(round_key: str) -> list[tuple[str, str]]:
    """Return list of (team_a, team_b) tuples for the given round."""
    if round_key == "r32":
        bracket = st.session_state.r32_bracket
        return [(bracket[2*i], bracket[2*i+1]) for i in range(16)]

    # For subsequent rounds, participants come from previous round winners
    prev_round = ROUND_KEYS[ROUND_KEYS.index(round_key) - 1]
    prev_winners_key = f"next_participants_{round_key}"
    if prev_winners_key in st.session_state:
        winners = st.session_state[prev_winners_key]
        return [(winners[2*i], winners[2*i+1]) for i in range(len(winners)//2)]
    return []


def _run_mc_from_bracket(pmb, n_sims: int = 10_000):
    """Run MC simulation from the current bracket state forward."""
    rng = np.random.default_rng(42)
    n_teams = len(pmb.teams)
    champ_cnt = np.zeros(n_teams, dtype=np.int32)
    final_cnt = np.zeros(n_teams, dtype=np.int32)
    sf_cnt = np.zeros(n_teams, dtype=np.int32)

    # Determine the starting round and participants
    bs = st.session_state.bracket_state
    cur_round = st.session_state.bracket_round

    # Build participant list for the current round
    cur_idx = ROUND_KEYS.index(cur_round)
    participants_key = f"next_participants_{cur_round}"

    if cur_round == "r32":
        bracket = st.session_state.r32_bracket
        cur_teams = bracket  # 32 flat names
    elif participants_key in st.session_state:
        cur_teams_list = st.session_state[participants_key]
        # flatten back to interleaved pairs
        cur_teams = []
        for i in range(0, len(cur_teams_list), 2):
            cur_teams.append(cur_teams_list[i])
            if i+1 < len(cur_teams_list):
                cur_teams.append(cur_teams_list[i+1])
    else:
        st.warning("Cannot determine current participants.")
        return None

    # Build IDs array for current participants
    try:
        cur_ids = np.array([pmb.team_idx[t] for t in cur_teams], dtype=np.int32)
    except KeyError as e:
        st.error(f"Unknown team in bracket: {e}")
        return None

    n_matchups = len(cur_ids) // 2
    # (n_sims, 2*n_matchups) tiled
    r = np.tile(cur_ids, (n_sims, 1))

    def ko_round(matchups):
        n = matchups.shape[0]
        k = matchups.shape[1] // 2
        winners = np.empty((n, k), dtype=np.int32)
        for m in range(k):
            a = matchups[:, 2*m]
            b = matchups[:, 2*m+1]
            p_a = pmb.P_ko[a, b]
            u = rng.random(n, dtype=np.float32)
            winners[:, m] = np.where(u < p_a, a, b)
        return winners

    # Simulate forward through all remaining rounds
    current = r
    for rnd_key in ROUND_KEYS[cur_idx:]:
        winners = ko_round(current)
        if rnd_key == "sf":
            for col in range(2):
                np.add.at(sf_cnt, winners[:, col], 1)
        elif rnd_key == "final":
            for col in range(1):
                np.add.at(final_cnt, winners[:, col], 1)
                np.add.at(final_cnt, winners[:, 1 if col == 0 else 0], 1)
            np.add.at(champ_cnt, winners[:, 0], 1)
            break
        if winners.shape[1] == 1:
            np.add.at(champ_cnt, winners[:, 0], 1)
            break
        # Pair winners for next round
        next_ids = winners.reshape(n_sims, -1)
        # interleave for next round
        current = np.empty((n_sims, next_ids.shape[1]), dtype=np.int32)
        current[:] = next_ids

    df = pd.DataFrame({
        "team": pmb.teams,
        "champion_pct": champ_cnt / n_sims * 100,
        "finalist_pct": final_cnt / n_sims * 100,
        "sf_pct": sf_cnt / n_sims * 100,
    }).sort_values("champion_pct", ascending=False).head(10).reset_index(drop=True)
    return df


def _match_card(pmb, match_idx: int, team_a: str, team_b: str, round_key: str):
    """Render a single match card with probabilities and advance buttons."""
    bs = st.session_state.bracket_state
    current_winner = bs.get(round_key, [None] * 20)
    if isinstance(current_winner, list):
        winner = current_winner[match_idx] if match_idx < len(current_winner) else None
    else:
        winner = current_winner  # final is scalar

    pw, pd_, pl, pko, recommended = _ko_probs(pmb, team_a, team_b)

    # Card header
    st.markdown(
        f'<div style="border:1px solid #333; border-radius:8px; padding:10px; '
        f'margin-bottom:6px; background:#16213e;">'
        f'<div style="font-size:0.75em; color:#888; margin-bottom:4px;">Match {match_idx+1}</div>',
        unsafe_allow_html=True,
    )

    # Team rows with flags and probabilities
    for team, p_win in [(team_a, pw*100), (team_b, pl*100)]:
        is_winner = (team == winner)
        is_loser  = (winner is not None and team != winner)
        style = ""
        if is_winner:
            style = "background:#1a4a2a; border-radius:6px; padding:2px 6px;"
        elif is_loser:
            style = "opacity:0.4; padding:2px 6px;"
        else:
            style = "padding:2px 6px;"
        rec_badge = " ★" if team == recommended and winner is None else ""
        st.markdown(
            f'<div style="{style}; display:flex; justify-content:space-between; align-items:center;">'
            f'<span>{flag_img_html(team, 20)}<b>{team}{rec_badge}</b></span>'
            f'<span style="font-size:0.8em; color:#aaa;">{p_win:.0f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Draw probability
    st.markdown(
        f'<div style="text-align:center; font-size:0.75em; color:#888; margin:4px 0;">Draw: {pd_*100:.0f}%</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if winner is None:
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            if st.button(f"▶ {team_a}", key=f"btn_{round_key}_{match_idx}_a"):
                _set_winner(round_key, match_idx, team_a)
                st.rerun()
        with c2:
            if st.button("🎲", key=f"btn_{round_key}_{match_idx}_sim", help="Simulate match"):
                w = _simulate_match(pmb, team_a, team_b)
                _set_winner(round_key, match_idx, w)
                st.rerun()
        with c3:
            if st.button(f"▶ {team_b}", key=f"btn_{round_key}_{match_idx}_b"):
                _set_winner(round_key, match_idx, team_b)
                st.rerun()
    else:
        st.caption(f"✅ **{winner}** advances")


def _set_winner(round_key: str, match_idx: int, winner: str):
    bs = st.session_state.bracket_state
    if round_key == "final":
        bs["final"] = winner
    else:
        if bs.get(round_key) is None:
            bs[round_key] = [None] * ROUND_GAMES[round_key]
        bs[round_key][match_idx] = winner

    # Check if all matches in this round are done
    _check_round_complete(round_key, bs)


def _check_round_complete(round_key: str, bs: dict):
    if round_key == "final":
        return  # Champion already set
    winners_list = bs.get(round_key, [])
    if all(w is not None for w in winners_list):
        # Advance
        next_idx = ROUND_KEYS.index(round_key) + 1
        if next_idx < len(ROUND_KEYS):
            next_round = ROUND_KEYS[next_idx]
            n_next = ROUND_GAMES[next_round]
            # Store winners as participants for next round
            st.session_state[f"next_participants_{next_round}"] = winners_list
            bs[next_round] = [None] * n_next
            st.session_state.bracket_round = next_round


def render():
    st.header("Step 3 — Knockout Bracket")
    pmb = get_pmb()

    bracket = st.session_state.get("r32_bracket")
    if not bracket:
        st.error("No bracket generated. Please complete Group Stage and Best Third first.")
        return

    bs = st.session_state.bracket_state
    cur_round = st.session_state.bracket_round

    # ── Progress bar ──────────────────────────────────────────────────────────
    round_labels_ordered = [ROUND_LABELS[r] for r in ROUND_KEYS]
    cur_idx = ROUND_KEYS.index(cur_round)
    st.progress((cur_idx) / (len(ROUND_KEYS) - 1), text=f"Current round: {ROUND_LABELS[cur_round]}")

    # ── Champion announcement ─────────────────────────────────────────────────
    if bs.get("final") is not None:
        champion = bs["final"]
        st.success(f"🏆 Champion: {champion}!")
        st.markdown(
            f'<div style="text-align:center; margin:20px 0;">'
            f'{flag_img_html(champion, 80)}'
            f'<h2 style="margin-top:8px;">{champion}</h2>'
            f'<p>2026 FIFA World Cup Champion</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Simulate from current bracket ────────────────────────────────────────
    n_mc = st.selectbox("Simulations", [10_000, 25_000, 50_000], index=0, key="mc_n")
    if st.button("🔮 Run Monte Carlo from current bracket state", type="primary"):
        with st.spinner(f"Running {n_mc:,} simulations…"):
            sim_df = _run_mc_from_bracket(pmb, n_sims=n_mc)
        if sim_df is not None:
            st.session_state["bracket_mc_results"] = sim_df

    if "bracket_mc_results" in st.session_state and st.session_state["bracket_mc_results"] is not None:
        df = st.session_state["bracket_mc_results"]
        st.subheader("Monte Carlo Results — Top 10")
        df["Team"] = df["team"]
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("champion_pct:Q", title="Champion %"),
            y=alt.Y("Team:N", sort="-x", title=None),
            color=alt.Color("champion_pct:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Team", "champion_pct", "finalist_pct", "sf_pct"],
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # ── Render all rounds ─────────────────────────────────────────────────────
    for rnd_key, rnd_label, n_games in ROUNDS:
        rnd_idx = ROUND_KEYS.index(rnd_key)
        if rnd_idx > cur_idx + 1:
            break  # Don't render future rounds

        st.subheader(rnd_label)

        participants = _get_participants(rnd_key)

        if not participants:
            st.info("Waiting for previous round to complete.")
            break

        # Render match cards in columns
        n_cols = min(4, len(participants))
        cols = st.columns(n_cols)
        for m_idx, (ta, tb) in enumerate(participants):
            with cols[m_idx % n_cols]:
                _match_card(pmb, m_idx, ta, tb, rnd_key)

        # Count complete
        if rnd_key != "final":
            rnd_winners = bs.get(rnd_key, [])
            done = sum(1 for w in rnd_winners if w is not None)
            total = len(rnd_winners)
            st.caption(f"{done}/{total} matches decided")

        st.markdown("---")

    # ── Navigation ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Best Third"):
            st.session_state.sim_step = 2
            st.rerun()
    with c2:
        if st.button("↺ Restart Simulator"):
            from dashboard.utils import reset_simulator
            reset_simulator()
            st.rerun()
