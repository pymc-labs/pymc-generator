# World model

The world as a PyMC model: *priors are distributions, worlds are draws*.
[`build_world_model`](#pymc_generator.world_model.build_world_model) assembles
one `pm.Model` for a fixed causal structure;
[`build_oracle_model`](#pymc_generator.world_model.build_oracle_model) is the
observed-data variant of the same model, so `pm.sample` yields the
structure-known posterior on any drawn world's dataset — see the
[posterior-oracle guide](../guide/oracle.md).

::: pymc_generator.world_model.build_world_model

::: pymc_generator.world_model.build_oracle_model

::: pymc_generator.world_model.sample_structure

::: pymc_generator.world_model.sample_prior_cond

::: pymc_generator.world_model.draw_worlds
