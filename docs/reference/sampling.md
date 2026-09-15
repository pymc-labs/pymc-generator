# Sampling one world

[`sample_scm`](#pymc_generator.worlds.sample_scm) draws a single accepted world
at the config's max sizes and returns an [`SCM`](#pymc_generator.worlds.SCM)
object carrying the observable series, the active DAG blocks, the drawn
parameters, and the exact decomposition truth.

::: pymc_generator.worlds.sample_scm

::: pymc_generator.worlds.SCM

## Graph introspection helpers

Small pure functions over a world's `g` / `params` dicts — used by the
description and plotting layers, and handy for your own analysis.

::: pymc_generator.worlds.path_to_y

::: pymc_generator.worlds.node_status

::: pymc_generator.worlds.edges_with_coeffs

::: pymc_generator.worlds.channel_role

::: pymc_generator.worlds.mechanism_label
