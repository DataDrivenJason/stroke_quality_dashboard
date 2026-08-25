"""
Stroke Quality Intelligence
===========================

A Streamlit dashboard for stroke service quality improvement: audit
indicators on a switchable SSNAP / Irish national standard, a full
statistical process control toolkit with NHS "Making Data Count"
variation and assurance icons, and a therapy layer built for the PT, OT
and SLT services rather than bolted on to a medical dashboard.

    streamlit run app.py

Layout
------
    app.py                  this router
    core/
        standards.py        indicator registry + the switchable target layer
        metrics.py          patient records -> numerators and denominators
        spc.py              the control chart engine
        mdc.py              Making Data Count classification and icons
        synth.py            synthetic cohort with planted special causes
        loaders.py          schema contract, validation, caching
        viz.py              Plotly builders and the palette
        ui.py               shared Streamlit furniture
    views/                  one module per page
    tests/                  SPC arithmetic + planted-signal ground truth

All data shipped with this app is simulated. See METHODS.md for the
statistics and README.md for how to point it at a real extract.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Stroke Quality Intelligence",
    page_icon="◔",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    st.Page("views/overview.py", title="Service overview", icon=":material/dashboard:",
            default=True),
    st.Page("views/hyperacute.py", title="Hyperacute pathway", icon=":material/timer:"),
    st.Page("views/control_charts.py", title="Control charts",
            icon=":material/monitoring:"),
    st.Page("views/therapy.py", title="Therapy dose", icon=":material/exercise:"),
    st.Page("views/caseload.py", title="Caseload & capacity", icon=":material/groups:"),
    st.Page("views/explorer.py", title="Patient explorer", icon=":material/search:"),
    st.Page("views/data_quality.py", title="Data quality", icon=":material/rule:"),
]

st.navigation(PAGES).run()
