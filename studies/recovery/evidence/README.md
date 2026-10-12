# Report visual evidence

Captured by the independent verifier for the Task 3 acceptance recorded in `../handoffs/task3-rereview.md`; inspected again by the coordinator before preservation here.

- `actual-executed-media-plot.png`: actual persisted Plotly figure from independently executing the corrected historical constant-smoke report and switching to `1:ols · C1`. Panels show true/fitted contribution, raw media spend and true/fitted baseline. `valid` describes optimizer/computational status, NOT contribution recovery or scientific adequacy.
- `actual-executed-pattern-table.png`: actual persisted first-method historical-smoke pattern table, including eligible/nonzero, known-zero, evaluable, failed and unresolved denominators, with unrepresented groups shown distinctly. This is tiny smoke evidence, not the 50-world scientific rate table.

These are Chromium headless screenshots of HTML exported from the executed notebook's Plotly MIME/table HTML. They are NOT screenshots of a live Jupyter browser and NOT independently regenerated charts. Selection changes were separately verified through executable notebook assertions against actual saved trajectories, including raw-spend values, baseline/interval traces and hidden null channels.

Original disposable evidence was under `/tmp/recovery-task3-ui-review-gi5k80we/`, including the executed `real-smoke-switches.ipynb` and renderer logs. The two PNGs are preserved here so the PR does not rely on temporary paths. Include/upload these assets when publishing the PR; no PR has yet been created.
