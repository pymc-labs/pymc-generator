# Sampling one world

[`sample_scm`](#prior_generator.worlds.sample_scm) draws a single accepted world
at the config's max sizes and returns an [`SCM`](#prior_generator.worlds.SCM)
object carrying the observable series, the active DAG blocks, the drawn
parameters, and the exact decomposition truth.

::: prior_generator.worlds.sample_scm

::: prior_generator.worlds.SCM

## Graph introspection helpers

Small pure functions over a world's `g` / `params` dicts — used by the
description and plotting layers, and handy for your own analysis.

::: prior_generator.worlds.path_to_y

::: prior_generator.worlds.node_status

::: prior_generator.worlds.edges_with_coeffs

::: prior_generator.worlds.channel_role

::: prior_generator.worlds.mechanism_label
