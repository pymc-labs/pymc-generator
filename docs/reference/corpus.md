# Corpus generation

Generate an N-world corpus — a dict of stacked numpy arrays over tasks and weeks
— either functionally with
[`sample_prior_predictive`](#prior_generator.sampler.sample_prior_predictive) or
through the [`DataGenerator`](#prior_generator.data_generator.DataGenerator)
facade (which adds batching, schema validation, and save-on-generate).

::: prior_generator.sampler.sample_prior_predictive

::: prior_generator.data_generator.DataGenerator

## Persistence

::: prior_generator.data_generator.save_corpus

::: prior_generator.data_generator.load_corpus
