# data/

All application data is **synthetic** and generated in memory at start-up by
`src/data/synthetic_data.py` (fixed seed 42 by default). No real customer, dealer, plant
or company performance data is stored or used.

Files that may appear here at runtime:

| File | Purpose |
|---|---|
| `demo_store.sqlite` | Local log of "Submit for approval" requests (human-in-the-loop record only; nothing is executed). Safe to delete. |

To change the dataset size or seed, use **Data generation controls** in the sidebar.
