# 04 - Replace symbolic dimension names with descriptive count names

**Status:** to-do
**Priority:** compatibility-sensitive cleanup

## Goal

Use descriptive count names consistently at public boundaries, in persisted
corpus schemas, in documentation, and eventually in internal implementation
code. Prefer:

- `n_tasks` instead of `N`
- `n_time_steps` instead of `T`
- `n_treatments` instead of `K`
- `n_covariates` instead of `M`
- `n_latent` instead of `J`
- `n_treatments_active`, `n_covariates_active`, and `n_latent_active` instead
  of `K_active`, `M_active`, and `J_active`

## Current Inconsistencies

- Internal tensors and shape bookkeeping use `K`, `M`, `J`, `T`, and `N`.
- The public `build_world_model` function accepts `T`.
- `SCMPrior` exposes `T`.
- Persisted corpus fields include `K_active`, `M_active`, and `J_active`.
- Shape documentation uses forms such as `(N, T, K)`.
- `SlotLayout` exposes `K`, `M`, and `J`.

The factory and most user-facing graph-size configuration already use
`n_treatments`, `n_covariates`, and `n_latent`; this task completes that
convention across the rest of the project.

## Deliverables

1. Inventory every public signature, dataclass field, persisted key, shape
   description, diagnostic field, test fixture, and downstream consumer that
   uses the symbolic dimension names.
2. Define one canonical vocabulary for task count, time count, treatments,
   covariates, latent variables, and active counts.
3. Rename internal variables and tensor-shape parameters to the canonical
   vocabulary where doing so improves clarity.
4. Rename public parameters and `SlotLayout` fields using a deliberate API
   migration.
5. Version the corpus schema before renaming `K_active`, `M_active`, and
   `J_active`; provide an explicit reader migration for previously persisted
   shards and coordinate the change with known consumers.
6. Update documentation, examples, diagnostics, tests, and type annotations so
   symbolic names remain only in mathematical explanations where they are
   useful and explicitly defined.

## Invariants

1. Existing persisted shards remain readable through a documented schema
   migration path.
2. New shards contain only the canonical persisted field names.
3. Public APIs do not expose single-letter dimension-count parameters.
4. Shape documentation defines dimensions descriptively rather than relying on
   unexplained `(N, T, K, M, J)` notation.
5. Generation remains byte-identical apart from deliberately renamed metadata
   keys; no RNG order, tensor value, graph structure, or model behavior changes.
6. Generation and oracle models continue to share the same dimensions, priors,
   and PyMC-Marketing-backed tensor transforms.

## Out Of Scope

- Changing SCM semantics, tensor axis order, mechanism families, or generated
  numerical values.
- Removing mathematical symbols from derivations where they are locally and
  clearly defined.
