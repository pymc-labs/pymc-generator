# Signal diagnostics

A corpus can satisfy every schema contract and still be *unlearnable*: if the
true per-channel contribution barely varies, there is no signal to attribute.
This module quantifies, per direct (C→Y) channel, how much signal the generator
actually produced — embedded in every corpus's `diagnostics["signal"]` block and
checkable at generation time.

::: prior_generator.signal_diagnostics
    options:
      show_root_heading: true
      show_root_toc_entry: false
      members:
        - per_channel_signal
        - signal_summary
        - check_signal_gate
        - DEFAULT_GATE
        - METRIC_KEYS
        - FRAC_KEYS
