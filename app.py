"""Dalmia AI Intelligence Hub - Streamlit entrypoint (navigation, global controls, styling).

Run with:  streamlit run app.py
"""
from __future__ import annotations

from datetime import timedelta

import streamlit as st

from src.config import (APP_NAME, APP_SUBTITLE, AS_OF_DATE, DEMO_BANNER, MAX_DEALERS, MAX_WEEKS, MIN_DEALERS,
                        MIN_WEEKS, POSITIONING_NOTE, REGIONS, DataSettings)
from src.ui import state
from src.ui.components import inject_css

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="expanded")

# Scenario assumptions persist across pages as objects in session state (e.g. "freight_scenario");
# page widgets take their defaults from those objects.

PAGES = [
    st.Page("pages/1_Executive_Dashboard.py", title="Executive Intelligence Dashboard", default=True),
    st.Page("pages/2_Dealer_Intelligence.py", title="Dealer Demand and Replenishment"),
    st.Page("pages/3_Freight_Optimisation.py", title="Freight and Dispatch Optimisation"),
    st.Page("pages/4_Tender_Intelligence.py", title="Tender and Institutional Sales"),
    st.Page("pages/5_Production_Inventory.py", title="Production and Inventory Planning"),
    st.Page("pages/6_Credit_Margin.py", title="Credit Risk and Profitable Orders"),
    st.Page("pages/7_Product_Compliance.py", title="Technical Compliance Copilot"),
    st.Page("pages/8_Pricing_Intelligence.py", title="Pricing and Discount Intelligence"),
    st.Page("pages/9_Offtake_Early_Warning.py", title="Project Offtake Early Warning"),
    st.Page("pages/10_Sales_Knowledge_Assistant.py", title="Sales Knowledge Assistant"),
]
pg = st.navigation(PAGES, position="hidden")
inject_css()

with st.sidebar:
    st.markdown(f'<div class="dh-side-title">{APP_NAME}</div><div class="dh-side-sub">{APP_SUBTITLE}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="dh-banner">{DEMO_BANNER}</div>', unsafe_allow_html=True)
    st.caption("Navigation")
    for p in PAGES:
        st.page_link(p, label=p.title)
    st.divider()

    st.caption("Global filters")
    as_of = AS_OF_DATE
    st.date_input("Date range (history)", value=(as_of - timedelta(days=365), as_of - timedelta(days=1)),
                  min_value=as_of - timedelta(days=7 * MAX_WEEKS), max_value=as_of, key="sb_date_range",
                  format="DD/MM/YYYY")
    st.multiselect("Region", REGIONS, default=[], key="sb_regions", placeholder="All regions")
    st.caption(f"Simulation as-of date: {as_of.strftime('%d-%b-%Y')}")

    with st.expander("Data generation controls"):
        cur = state.settings()
        seed = st.number_input("Random seed", 0, 99999, cur.seed, key="sb_seed")
        n_dealers = st.slider("Interactive dealer subset", MIN_DEALERS, MAX_DEALERS, cur.n_dealers, 50, key="sb_dealers")
        n_weeks = st.slider("Weeks of demand history", MIN_WEEKS, MAX_WEEKS, cur.n_weeks, 4, key="sb_weeks")
        n_inst = st.slider("Institutional customers", 80, 600, cur.n_institutional, 10, key="sb_inst")
        st.caption("A separate vectorised simulation represents the full 49,300-partner network.")
        if st.button("Regenerate synthetic data", key="sb_regen", width="stretch"):
            st.session_state["data_settings"] = DataSettings(int(seed), int(n_dealers), int(n_weeks), int(n_inst))
            st.cache_data.clear()
            st.rerun()
    if st.button("Reset scenario assumptions", key="sb_reset", width="stretch"):
        state.reset_scenarios()
        st.rerun()
    st.caption(POSITIONING_NOTE)

pg.run()
