# Visualization

Matplotlib renders for a world — the four figures every bundle contains. Each
takes an [`SCM`](sampling.md#prior_generator.worlds.SCM) and a save path;
matplotlib is imported lazily, and headless callers should select the `Agg`
backend first (`matplotlib.use("Agg")`).

::: prior_generator.viz
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - plot_dag
        - plot_timeseries
        - plot_decomposition
        - plot_channels
