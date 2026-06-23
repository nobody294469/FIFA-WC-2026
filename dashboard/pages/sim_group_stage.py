"""sim_group_stage.py — Step 1: Group Stage ranking with drag-and-drop."""
from __future__ import annotations
import copy
import streamlit as st
from streamlit_sortables import sort_items

from core.config import WC2026_GROUPS
from dashboard.utils import flag_img_html, flag_label, get_pmb


def _group_win_pct(pmb, teams: list[str]) -> dict[str, float]:
    """Rough group-win probability for each team using pairwise P_ko."""
    out = {}
    for t in teams:
        if t not in pmb.team_idx:
            out[t] = 0.0
            continue
        p = 1.0
        for opp in teams:
            if opp == t or opp not in pmb.team_idx:
                continue
            i, j = pmb.team_idx[t], pmb.team_idx[opp]
            p *= float(pmb.P_ko[i, j])
        out[t] = round(p * 100, 1)
    return out


def render():
    st.header("Step 1 — Group Stage")
    st.markdown(
        "Drag teams within each group to set final standings. "
        "The **top two** advance; the **third-place** team enters the best-thirds pool."
    )

    pmb = get_pmb()

    # Initialise ordered groups from config if not yet set
    if st.session_state.get("groups_ordered") is None:
        st.session_state.groups_ordered = copy.deepcopy(WC2026_GROUPS)

    groups = st.session_state.groups_ordered

    # Render 3 columns × 4 rows of group cards
    group_names = sorted(groups.keys())
    cols = st.columns(3)

    new_orders: dict[str, list[str]] = {}

    for idx, g in enumerate(group_names):
        col = cols[idx % 3]
        with col:
            st.markdown(f"#### Group {g}")
            teams = groups[g]
            gwp = _group_win_pct(pmb, teams)

            # Build label list for sortable (plain text — sortable needs strings)
            labels = [f"{t}" for t in teams]
            sorted_labels = sort_items(labels, key=f"group_{g}", direction="vertical")

            # Re-derive team names from sorted labels
            # labels are plain team names so direct map works
            new_orders[g] = sorted_labels

            # Display probability badges below the sortable
            for pos, team in enumerate(sorted_labels):
                pos_label = {0: "🥇 1st", 1: "🥈 2nd", 2: "🥉 3rd", 3: "4th"}[pos]
                pct = gwp.get(team, 0.0)
                st.markdown(
                    f'{flag_img_html(team, 18)}'
                    f'<span style="font-size:0.85em;vertical-align:middle">'
                    f'{pos_label} · <b>{team}</b> · group-win {pct:.0f}%</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")

    # Persist order immediately on every render
    st.session_state.groups_ordered = new_orders

    st.markdown("---")
    if st.button("Lock Group Stage & Continue →", type="primary"):
        st.session_state.groups_locked = True
        st.session_state.sim_step = 2
        st.rerun()
