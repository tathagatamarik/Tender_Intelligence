# Dalmia AI Intelligence Hub

*AI-powered commercial, supply-chain and profitability intelligence for the cement industry.*

> **DEMO ENVIRONMENT — SYNTHETIC DATA — NOT CONNECTED TO LIVE COMPANY SYSTEMS**
>
> This is a proof of concept for management decision support. It is not an official Dalmia Bharat
> product, and it does not replace ERP, sales, transport, finance or technical approval systems. All data is
> synthetic. Every saving, probability and recommendation is a simulated, illustrative result,
> not an achieved or verified business outcome.

A multi-page Streamlit application. Predictive ML, deterministic generative AI (document
extraction, retrieval and templated briefings) and mathematical optimisation work on **one coherent
synthetic business environment**: plants, depots, dealers, institutional customers, projects, orders,
dispatches, inventory, invoices and tenders.

---

## 1. Quick start

Requirements: Python 3.11 (3.10 to 3.12 should work), roughly 1 GB RAM, and no internet access after installation.

```bash
git clone https://github.com/tathagatamarik/Tender_Intelligence.git
cd Tender_Intelligence
python -m venv .venv                 # or: conda create -p ./.venv python=3.11
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The first page load generates the data and trains all models in about 10 seconds.
After that, results are cached.

In this workspace the environment is already installed under `.venv/`, so you can run:

```bash
.venv/bin/streamlit run app.py
```

Run the tests (53 tests, about 40 seconds, including an end-to-end render of all 10 pages):

```bash
.venv/bin/python -m pytest
```

### Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with the GitHub account that owns this repository.
2. Create an app with repository `tathagatamarik/Tender_Intelligence`, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.11**. No secrets are required.
4. Deploy. `requirements.txt` installs the Python packages and `packages.txt` installs `libgomp1` (needed by LightGBM).

The first load on Cloud takes longer (around 20-40 seconds) while data is generated and models train; the result is then cached.
The approval log (`data/demo_store.sqlite`) is local to the container and resets when the app restarts.

No API keys, paid services or external LLMs are used. Streamlit usage statistics are disabled in
`.streamlit/config.toml`, so the app makes no outbound network calls.

---

## 2. Pages

| # | Page | What it demonstrates | Core technique |
|---|------|----------------------|----------------|
| 1 | Executive Intelligence Dashboard | 10 KPIs, 8 charts, filters, **AI Recommended Actions**, ROI simulator, data validation, approval queue | Aggregates all modules; recommendation engine |
| 2 | Dealer Demand and Replenishment | 7/14/30-day forecasts, accuracy against baselines, stock-out risk, replenishment, segmentation, visit plan, 49,300-partner network view | LightGBM global forecaster (chronological holdout), safety-stock policy, rules-based segmentation |
| 3 | Freight and Dispatch Optimisation | Plant/depot/vehicle allocation, current-versus-optimised cost, utilisation, on-time delivery, empty km, Green Channel and Stock-on-Wheels, ETA, route anomalies | MILP (SciPy HiGHS), gradient-boosted ETA, Isolation Forest plus GPS rules |
| 4 | Tender and Institutional Sales | Tender pipeline and calendar, regex extraction from TXT/PDF, eligibility checklist, win probability, risk-adjusted EV, bid/no-bid briefing | Logistic regression without post-outcome leakage, deterministic parser |
| 5 | Production and Inventory Planning | Multi-week production, allocation and transfers; disruption and surge simulator; baseline versus optimised | Multi-period LP (HiGHS) |
| 6 | Credit Risk and Profitable Orders | Late-payment risk, ECL, collection priority, exposure after orders, explainable drivers, order-level risk-adjusted contribution | Calibrated logistic and GBM, Kaplan-Meier payment timing, PD x LGD x EAD |
| 7 | Technical Compliance Copilot | Spec extraction, PASS / FAIL / NEEDS REVIEW matrix with evidence, missing documents, consultant approvals, downloadable report | Deterministic rules engine |
| 8 | Pricing and Discount Intelligence | Price and margin trends, elasticity with confidence intervals, discount scenarios within policy, scheme effectiveness, low-margin customers | Log-log OLS with fixed effects (statsmodels) |
| 9 | Project Offtake Early Warning | Alerts, at-risk tonnage, working capital, change points, re-planning comparison | CUSUM, trend, z-scores, forecast against plan |
| 10 | Sales Knowledge Assistant | Offline Q&A with cited sources and role-based permissions | TF-IDF retrieval, template generation, LLM adapter interface |

The sidebar contains the title, demo disclosure, navigation, global date range and region filters,
data-generation controls (seed, dealer subset, history length, institutional customers) and a
**Reset scenario assumptions** button.

---

## 3. Cross-module integration

All pages share one environment (`src/data/synthetic_data.py`) and one definition of order economics
(`src/services/order_economics.py`). Results flow between modules as follows:

- **Dealer forecasts → replenishment → production plan.** Week-1 trade demand uses the larger of the forecast and the recommended dealer orders.
- **Inventory and capacity → dispatch.** Available-to-promise stock and loading capacity constrain the freight MILP.
- **Dispatch → margin.** Page 6 computes order contribution using freight from the current dispatch scenario. Change the diesel price on Page 3 and margins on Page 6 change.
- **Credit risk → profitability and tenders.** The probability of default (PD) drives expected credit loss in order economics and in tender risk-adjusted value.
- **Tenders → production.** Tenders recommended to bid add probability-weighted demand to the plan.
- **Offtake alerts → supply planning.** Project schedules are adjusted by forecast attainment.
- **Everything → dashboard.** The executive dashboard and AI Recommended Actions are computed from the *current* scenario state of every module.

Order economics, with each cost counted once:

```
net revenue                = list price x qty - discounts
contribution margin        = net revenue - production cost - freight (primary + secondary) - handling
risk-adjusted contribution = contribution - expected credit loss (PD x LGD x exposure) - financing cost of collection period
```

---

## 4. Architecture

```
app.py                      entrypoint: st.navigation, sidebar, global controls, styling
pages/                      10 thin page scripts (UI only)
src/config.py               master data and all explicit demo assumptions
src/data/                   synthetic_data.py (generator), geography.py, schemas.py (pydantic), validation.py
src/models/                 demand_forecasting, dealer_segmentation, credit_risk, tender_scoring,
                            eta_prediction, anomaly_detection, pricing_analysis, metrics
src/optimisation/           freight_optimisation (MILP), inventory_optimisation (LP), replenishment
src/services/               order_economics, roi_calculator, recommendation_engine, document_parser,
                            document_assistant, compliance, freight_tariff, persistence (SQLite)
src/ui/                     state.py (caching / session state), components.py, charts.py, formatting.py
tests/                      pytest suite incl. Streamlit AppTest page rendering
docs/                       ARCHITECTURE.md, DEMO_WALKTHROUGH.md, screenshots/
```

The core logic in `src/models`, `src/optimisation`, `src/services` and `src/data` has no Streamlit
dependency and is fully unit-tested. See `docs/ARCHITECTURE.md` for details.

**Performance and state.** The data and trained models are cached as resources keyed by the data settings.
Scenario computations are cached by their parameter values. For example, changing a freight rate re-runs
only the dispatch MILP (about 1.5 s); it never retrains models. Scenario assumptions persist across pages
in session state.

---

## 5. Governance and responsible-AI controls

- Human approval is required throughout. "Submit for approval" buttons record a request in local SQLite (`data/demo_store.sqlite`), and **nothing is executed**. This applies to credit, pricing, tender, dispatch, planning and compliance decisions.
- Uploaded files are TXT or PDF only, limited to 2 MB and 50 pages, and validated. Their content is parsed as text and never executed or treated as instructions.
- Retrieved evidence is shown separately from generated explanations. The assistant cites the exact chunks it used and says when information is unavailable. Finance-restricted content is withheld from other roles.
- Credit models use no protected personal attributes, only commercial payment behaviour.
- Status is never conveyed by colour alone: every status carries a text label (HIGH PRIORITY, NEEDS REVIEW, WITHIN LIMIT and so on). There are no emojis or icons.
- Compliance standard references and limits are labelled **illustrative**. The best possible status is "ALL CHECKS PASSED - PENDING AUTHORISED APPROVAL".
- Competitor prices are used only as a simulated market indicator, and there is no coordinated pricing logic.

---

## 6. Model evaluation

All metrics are **computed at runtime** from holdout predictions; none are hard-coded. Indicative values with the default seed (42):

| Use case | Evaluation | Indicative result |
|---|---|---|
| Dealer demand (weekly, dealer x product) | Last 12 weeks, one-week-ahead | WAPE about 0.52 against 0.53 (4-week moving average) and 0.60 (seasonal naive); depot-week WAPE about 0.07 |
| Late payment (30+ days, next 90 days) | Time-separated snapshot, 30% customer holdout | ROC-AUC about 0.87, with calibration curve |
| Tender win | Chronological split by closing date | Holdout ROC-AUC about 0.74 (in-sample 0.81 shown separately) |
| ETA | Chronological 20% holdout | MAE about 1.7 h against 3.2 h baseline |
| Route anomalies | Against simulated labels | GPS rules: precision about 0.67, recall about 0.92 |
| Offtake slowdown | Against simulated labels | ACTION REQUIRED: precision and recall about 0.96 |
| Freight MILP | Against current-practice baseline | About 11% fulfilment-cost and 36% freight saving on this order book (simple rule-based baseline) |
| Production LP | Against nearest-plant heuristic | About 10% lower total relevant cost, higher fill rate |

Dealer-week demand is intermittent and noisy, so the model's advantage over a moving average is modest at that
level. The page shows this honestly. Optimisation savings are measured against a deliberately simple rule-based
baseline, so real-world savings would be lower and must be validated.

---

## 7. Known simplifications and limitations

- **Scale.** The interactive sample is 500 dealers and 250 institutional accounts. Plant capacities are *allocated to the sample* so that constraints bind; they are not real capacities. The full 49,300-partner network is a separate vectorised simulation used for aggregates. ROI inputs default to editable network-scale assumptions.
- **Freight.** This is an assignment and allocation MILP, not full vehicle routing. Multi-drop routes, driver hours and live re-dispatch are out of scope. The freight tariff is synthetic.
- **Planning.** Weekly buckets, same-week arrival, lost sales instead of backorders, product grades aggregated into categories, and linear costs.
- **Forecast horizons.** 7/14/30-day horizons are derived from weekly forecasts. Prediction intervals are approximate.
- **ECL.** A simplified PD x LGD x EAD demonstration, **not** a regulatory accounting model (for example Ind AS 109).
- **Elasticity.** Estimates are correlational. Scenario results apply them as explicit assumptions.
- **Assistant.** Extractive and template-based. `EnterpriseLLMAdapter` is a placeholder interface for a future approved LLM.
- **Dealer stock snapshot.** This is simulated, standing in for a dealer ordering platform feed.
- **Demonstration assumptions.** Business context figures (channel mix, about 20% freight share, 49,300 partners) are demonstration assumptions, not verified company disclosures.

---

## 8. Further documents

- `docs/ARCHITECTURE.md`: technical architecture and data flow
- `docs/DEMO_WALKTHROUGH.md`: how to demonstrate each use case to business stakeholders
- `docs/screenshots/`: screenshots of every page
