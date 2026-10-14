# Demo Walkthrough for Business Stakeholders

Plan for about 35-45 minutes. Open with the disclosure: *"Everything you will see runs on synthetic data and is
not connected to company systems. Numbers show how the methods work, not results the company has achieved."*

Before the meeting, run `streamlit run app.py` and open the Executive Dashboard once. The first load takes
about 10 seconds while the models train.

---

## 1. Executive Intelligence Dashboard (5 min)

**Message:** one integrated view, built from the same data every other page uses.

1. Walk through the KPI cards: volume, net revenue, contribution margin, freight per tonne, forecast accuracy,
   on-time delivery, inventory, overdue receivables, credit exposure and addressable AI value.
2. Change the sidebar **Region** to *East* and show that every KPI and chart updates.
3. Open **AI Recommended Actions**. Each item shows a reason, an estimated impact, a value type, a confidence
   level and the required action. Point out that the list is generated from module outputs, not typed in.
4. Open **Financial impact and ROI**. Move "Achievable freight saving" between 2% and 6%, then show the
   conservative, base and upside scenarios. Stress that working-capital release is shown separately and is not counted as profit.
5. Open **Data quality and approvals** to show the validation checks and the approval queue.

## 2. Dealer Demand and Replenishment (5 min)

**Questions answered:** Where will demand rise or fall? Which dealers should sales prioritise?

1. Under **Forecast accuracy**, show the model against both baselines on a chronological holdout.
   Explain that dealer-week demand is noisy, and that depot-level accuracy (shown in the card) is what planning uses.
2. Under **Dealer drill-down**, search for a dealer and show the history, the holdout predictions and the forecast with its range.
3. Change **Service level** from 0.95 to 0.99. Safety stock and recommended orders rise.
   Switch the horizon between 7, 14 and 30 days.
4. Under **Replenishment and stock-outs**, filter to LIKELY STOCKOUT and show credit-capped lines.
   Click *Submit recommended orders for sales approval*.
5. Under **Segmentation and visit plan**, pick one sales rep and show their ranked visit list with reasons.
   Then open **Network simulation** for the 49,300-partner view.

## 3. Freight and Dispatch Optimisation (6 min)

**Questions answered:** Which plant or depot should fulfil each order? How do we cut freight cost and empty kilometres?

1. Show the KPIs: fulfilment and freight savings, utilisation, predicted on-time delivery and manual-intervention orders.
2. Open the **Shipment table**, filter to *RE-SOURCED*, and compare the baseline and recommended source, vehicle and ETA.
3. Change the **Diesel price** to 110, mark **PL01** as unavailable, and click *Run dispatch optimisation*.
   The plan re-solves in about 2 seconds.
4. Open **Green Channel and Stock-on-Wheels** and click *Run scenario comparison*.
   Compare delivery time against cost across the three modes.
5. Open **ETA and route anomalies** to show ETA accuracy against the naive baseline, and the flagged deviations and unscheduled halts.
6. Use **Model and constraints** to state the scope limit: this is an assignment model, not full vehicle routing.

## 4. Tender and Institutional Sales (5 min)

**Question answered:** Which tenders deserve attention?

1. Show the pipeline and closing calendar.
2. In **Tender evaluation and briefing**, pick a *NO BID - ELIGIBILITY GAP* tender. It typically fails on an
   expired ISO 14001 certificate in the synthetic repository. Then pick a *BID* tender and show the risk deductions and the generated briefing.
3. Click *Download as PDF*, then upload that PDF in **Upload a tender document**. The fields are re-extracted,
   you can correct them, and the tender is scored.
4. In **Win-probability model**, show the holdout metrics, the calibration curve and the excluded post-outcome fields that prevent leakage.

## 5. Production and Inventory Planning (5 min)

**Question answered:** Which production and inventory decisions minimise avoidable cost?

1. Compare baseline and optimised cost, fill rate, stock-outs and inventory.
2. Set the scenario to **plant shutdown PL04 for weeks 1 to 3** with an **East surge of 25%**, then run.
   The optimiser re-routes supply and uses overtime, while the baseline loses sales.
3. Open **Demand inputs** to show that dealer forecasts, replenishment, project offtake and tenders all feed this plan.

## 6. Credit Risk and Profitable Orders (5 min)

**Questions answered:** Which customers pose risk? Which orders earn the best risk-adjusted contribution?

1. Show the risk distribution, invoice ageing and the collection priority list with key risk drivers.
2. In **Customer drill-down**, show the explainable factor chart for a high-risk account.
3. Open **Order profitability** and walk the waterfall from gross revenue to risk-adjusted contribution.
   Point out orders flagged *WEAK RISK-ADJUSTED CONTRIBUTION*: positive gross margin, but weak after freight, ECL and financing.
4. Set **Credit-limit scenario multiplier** to 0.7 and **payment-term change** to +30 days, then recalculate.
   More customers need escalation and financing cost rises. Changes always require finance approval.

## 7. Technical Compliance Copilot (3 min)

**Question answered:** Does a product meet the project's technical requirements?

1. Choose **CP02 Metro Viaduct**. P02 shows NEEDS REVIEW because the alkali value is missing in its latest test,
   and P03 shows NOT COMPLIANT because the type is not permitted.
2. Choose **CP01** with P04 and walk through the evidence source of each criterion, then download the report.
3. Stress that all limits are illustrative and that approval stays with authorised technical personnel.

## 8. Pricing and Discount Intelligence (3 min)

1. Show the regional price and margin trends, then the elasticity estimates with confidence intervals against the simulated truth.
2. In the **Discount scenario simulator**, raise the discount by Rs 150 per tonne. Volume rises, but margin may
   fall below the policy floor and trigger a POLICY BREACH. Show the suggested range within policy.
3. Make the guardrail explicit: competitor prices are a market indicator only, there is no coordinated pricing,
   and every proposal needs management approval.

## 9. Project Offtake Early Warning (3 min)

1. Show the ACTION REQUIRED projects, the at-risk tonnage and the working-capital implication.
2. Drill into one project and show the plan against actuals, the CUSUM change point and the forecast.
3. Under **Re-planning scenario**, click *Compare plans* to show the planned-tonnage difference with and without the alerts.
   Stress that these are estimates, not confirmed cancellations.

## 10. Sales Knowledge Assistant (3 min)

1. Click the example questions in turn. Each answer cites chunk IDs, and the retrieved excerpts appear below it.
2. Ask "Which customers require a credit review?" as a **Sales Representative**, then as a **Finance Manager**,
   to show role-based withholding.
3. Ask an unrelated question to show the "information not available" response.

---

## Closing message

Predictive ML (forecasting, credit, tender, ETA, anomaly detection), deterministic generative AI (extraction,
retrieval, briefings) and optimisation (dispatch MILP, planning LP) share one data model. Decisions stay with
people. The next step would be a pilot on real, governed data for one region, with baseline measurement before claiming any savings.
