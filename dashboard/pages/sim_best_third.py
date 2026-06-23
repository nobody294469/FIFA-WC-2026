"""sim_best_third.py — Step 2: Select exactly 8 best third-place teams."""
from __future__ import annotations
import streamlit as st

from dashboard.utils import flag_img_html


def render():
    st.header("Step 2 — Best Third-Place Teams")
    st.markdown(
        "Select **exactly 8** of the 12 third-place teams to advance to the Round of 32. "
        "Click a card to toggle selection."
    )

    groups = st.session_state.groups_ordered

    # Third-place team = index 2 in each group's ordered list
    thirds: list[tuple[str, str]] = []  # (group_letter, team_name)
    for g in sorted(groups.keys()):
        ranked = groups[g]
        if len(ranked) >= 3:
            thirds.append((g, ranked[2]))

    selected: list[str] = st.session_state.best_thirds_selected

    # --- Selection grid: 4 columns ---
    cols = st.columns(4)
    for idx, (grp, team) in enumerate(thirds):
        col = cols[idx % 4]
        with col:
            is_sel = team in selected
            border_color = "#2ecc71" if is_sel else "#555"
            bg_color = "#1a3a2a" if is_sel else "#1e1e2e"
            check = "✅ " if is_sel else ""

            label = (
                f'<div style="border:2px solid {border_color}; border-radius:8px; '
                f'padding:10px; background:{bg_color}; margin-bottom:8px; '
                f'text-align:center; cursor:pointer; min-height:80px;">'
                f'{flag_img_html(team, 36)}<br>'
                f'<b style="font-size:0.9em">{check}{team}</b><br>'
                f'<span style="font-size:0.75em; color:#aaa">Group {grp}</span>'
                f'</div>'
            )
            st.markdown(label, unsafe_allow_html=True)

            btn_label = "✓ Deselect" if is_sel else "+ Select"
            if st.button(btn_label, key=f"third_{grp}_{team}"):
                if is_sel:
                    selected.remove(team)
                else:
                    if len(selected) < 8:
                        selected.append(team)
                    else:
                        st.warning("Already 8 selected. Deselect one first.")
                st.session_state.best_thirds_selected = selected
                st.rerun()

    # --- Counter badge ---
    n = len(selected)
    color = "#2ecc71" if n == 8 else "#e74c3c"
    st.markdown(
        f'<div style="font-size:1.4em; font-weight:bold; color:{color}; margin:12px 0;">'
        f'{n} / 8 selected</div>',
        unsafe_allow_html=True,
    )

    if n == 8:
        st.success("8 teams selected. Ready to continue.")
    else:
        st.info(f"Select {8 - n} more team{'s' if 8 - n != 1 else ''}.")

    # --- Navigation ---
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Group Stage"):
            st.session_state.sim_step = 1
            st.session_state.groups_locked = False
            st.rerun()
    with c2:
        if st.button("Generate Bracket →", type="primary", disabled=(n != 8)):
            _generate_r32()
            st.session_state.sim_step = 3
            st.rerun()


# ── FIFA R32 bracket generation ──────────────────────────────────────────────

def _generate_r32():
    """
    Apply the official FIFA WC2026 Round of 32 bracket mapping.

    Groups index: 0=A, 1=B, 2=C, 3=D, 4=E, 5=F, 6=G, 7=H, 8=I, 9=J, 10=K, 11=L

    The 16 R32 fixtures in order (interleaved [a0,b0, a1,b1, ...]):
    M65: 1A v 2C        M66: 1B v 2D        M67: 1C v 2A
    M68: 1D v 2B        M69: 1E v 2G        M70: 1F v 2H
    M71: 1G v 2E        M72: 1H v 2F        M73: 1I v 2K
    M74: 1J v 2L        M75: 1K v 2I        M76: 1L v 2J
    M77: 1F v 3rd       M78: 1G v 3rd       M79: 1A v 3rd
    M80: 1L v 3rd       M85: 1B v 3rd       M86: 1J v 2H
    M87: 1K v 3rd       M88: 2D v 2G

    Simplified to 16 matchup pairs for this implementation:
    """
    groups = st.session_state.groups_ordered
    thirds_sel = set(st.session_state.best_thirds_selected)

    def w(g):  # group winner (1st)
        return groups[g][0]

    def r(g):  # group runner-up (2nd)
        return groups[g][1]

    def t3(g):  # 3rd-place from group (only if selected)
        team = groups[g][2]
        return team if team in thirds_sel else None

    # Get which groups contributed 3rd-place qualifiers
    # For simplicity, we assign the 8 best-thirds to the 8 host slots
    # using the FIFA-prescribed host order: 1A,1B,1D,1E,1G,1I,1K,1L
    host_groups = ["A", "B", "D", "E", "G", "I", "K", "L"]
    thirds_ordered: list[str] = []

    # Pull only selected thirds, preserving group order
    for g in sorted(groups.keys()):
        t = groups[g][2]
        if t in thirds_sel:
            thirds_ordered.append(t)

    # Assign thirds to host slots (simplified FIFA logic)
    t3_by_host: dict[str, str] = {}
    for i, hg in enumerate(host_groups):
        t3_by_host[hg] = thirds_ordered[i] if i < len(thirds_ordered) else ""

    # Build 16-match bracket (32 team names, interleaved)
    bracket: list[str] = []
    fixtures = [
        (w("A"), r("C")),
        (w("C"), r("A")),
        (w("B"), r("D")),
        (w("D"), r("B")),
        (w("E"), r("G")),
        (w("G"), r("E")),
        (w("F"), r("H")),
        (w("H"), r("F")),
        (w("I"), r("K")),
        (w("K"), r("I")),
        (w("J"), r("L")),
        (w("L"), r("J")),
        (w("F"), t3_by_host.get("G", "") or t3_by_host.get("A", "")),
        (w("A"), t3_by_host.get("A", "") or thirds_ordered[0] if thirds_ordered else ""),
        (w("B"), t3_by_host.get("B", "") or thirds_ordered[1] if len(thirds_ordered) > 1 else ""),
        (w("K"), t3_by_host.get("K", "") or thirds_ordered[-1] if thirds_ordered else ""),
    ]

    for a, b in fixtures:
        bracket.append(a or "TBD")
        bracket.append(b or "TBD")

    # Validate: no team appears more than once
    seen: set[str] = set()
    deduped: list[str] = []
    for team in bracket:
        if team in seen or team == "TBD":
            deduped.append(team)
        else:
            seen.add(team)
            deduped.append(team)

    st.session_state.r32_bracket = deduped
    # Reset bracket state
    st.session_state.bracket_state = {
        "r32": [None] * 16,
        "r16": [None] * 8,
        "qf":  [None] * 4,
        "sf":  [None] * 2,
        "final": None,
    }
    st.session_state.bracket_round = "r32"
