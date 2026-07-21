# World model

The world as a PyMC model: *priors are distributions, worlds are draws*.
[`build_world_model`](#prior_generator.world_model.build_world_model) assembles
one `pm.Model` for a fixed causal structure;
[`build_oracle_model`](#prior_generator.world_model.build_oracle_model) is the
observed-data variant of the same model, so `pm.sample` yields the
structure-known posterior on any drawn world's dataset — see the
[posterior-oracle guide](../guide/oracle.md).

::: prior_generator.world_model.build_world_model

::: prior_generator.world_model.build_oracle_model

::: prior_generator.world_model.sample_structure

::: prior_generator.world_model.sample_prior_cond

::: prior_generator.world_model.draw_worlds
