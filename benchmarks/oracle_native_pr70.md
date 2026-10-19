# Native Oracle pilot: PR #70

## Scope and source

The pilot ran after PR #70 opened. The executed generator commit was
`7b82833c3c8c0b81d8a8b57a402a347d39a26616`. It used three views from the
consumer's recorded `eval-d6e3a8802add`, not newly generated data.

Modal environment: `pfn`. App: `ap-Pz6L5CJbil28Tqmrlwic6T`.
Resources: 8 vCPUs (request and limit), 16 GiB RAM. Each fit used one native
nutpie call with 8 chains, `cores=8`, 1,000 warmup steps, 500 draws per chain,
`adaptation="draw_diag"`, and `target_accept=0.9`.
The generator was installed from the exact Git commit with no dependency
updates in the consumer's locked reference image. The report records all versions.

The configured capacity was 10 channels and 10 controls. The two worlds had
7/5 and 6/2 active channels/controls. The same sampler instance
`8b27aa9f700a437698491cab265fe870` served all three fits.

## Timing and computational health

Initial native sampler compilation: **74.38 seconds**.
Total function time for all three fits, prediction, diagnostics, and saved
artifacts: **1,529.50 seconds (25.5 minutes)**. Image building is not included.

| World | Context | Fit, seconds | Prediction, seconds | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Health |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w000-linear-none | 99 | 387.27 | 48.71 | 1.00780 | 1,203 | 1,665 | 1 | unhealthy |
| w000-linear-none | 207 | 469.04 | 24.39 | 1.00838 | 1,460 | 2,255 | 0 | healthy |
| w001-linear-none | 99 | 484.66 | 14.05 | 1.00657 | 1,353 | 1,504 | 0 | healthy |

Fit time includes binding, sampling, and the library's receipt diagnostics.
The first prediction includes compilation of the separate prediction kernel.
All fits had zero tree-depth hits. Minimum BFMI was 0.752, 0.861, and 0.899.
The consumer computed the same parameter and predictive-window diagnostics as
its original Oracle benchmark. It used R-hat <= 1.01, bulk/tail ESS >= 400,
and zero divergences. It did not calculate nested R-hat for the new sampler's
independent starts. Every fit remained in this report.

## Comparison limits

The previous callback/process pilot on w000 at context 99 used the same draw
and tuning counts and target acceptance. Its complete job took 622.33 seconds;
its slowest chain sampled for 476.79 seconds. The new fit took 387.27 seconds,
but this is not a controlled estimate of a backend-only speed difference:

- The new model samples prior-only capacity parameters.
- Native nutpie manages chain initialisation and RNG streams. They are not the
  old runner's separately seeded chains with shared superchain starts.
- Modal hardware and timing boundaries can differ.
- The native fit includes receipt diagnostics; the old chain time does not.

Two fits passed the checks. The third failed because it had one divergence.
Do not interpret these three selected views as full-benchmark convergence or
as evidence that the divergence can be ignored. No full-set launch occurred.

## Artifacts

The consumer repository contains the executed script and downloaded report:

- `data/oracle_recovery_check/native_pr70_modal.py`
- `data/oracle_recovery_check/native-pr70-report.json`

Modal volume `mfm-ops` contains the protocol, full posterior and private state,
paired predictions, per-fit receipts, and summary report under:

```text
/experiments/native-oracle-pr70/ee06e14bddf6/
```

The first image-build attempt failed before sampling because a build step was
added after a local source mount. The successful script installs the PR before
adding local source. Original benchmark artifacts and dependency pins remain
unchanged.
