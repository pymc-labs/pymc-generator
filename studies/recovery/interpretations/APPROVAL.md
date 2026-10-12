# Interpretation approval record

The user approved `tv50.md` and `bounded50.md` together in Plannotator session **pn-2a768e**, then explicitly supplied the reviewer name **Teemu Säilynoja**. Approval concerns the interpretations bound to the report IDs below, not PR publication or new inference.

| Interpretation | Report run ID | Sealed review ID |
|---|---|---|
| `tv50.md` | `f2dccfb682b95c9e8da4ffdc0d3f8c5f4134098468d54ef4f8c8e974cfbbffb3` | `15a972d9798d4f1c7631264c77fc7fa4da317a6fb9d8658ff52db8334e405cea` |
| `bounded50.md` | `041a2b4bfc17d53631ad524508fe0a6243c8bdf50df47ec3ee567a7c202562de` | `031ec6a9b96dfa8e7fdd2821cc25eabcef482c87a36082c2d62e81dc30f7939b` |

Canonical sealed attachments under `studies/recovery/report-artifacts/`:

- `tv50-profile-v2-review/`
- `bounded50-profile-v2-review/`

Approved executed report copies in that same directory:

- `tv50-profile-v2-reviewed-executed.ipynb`
- `bounded50-profile-v2-reviewed-executed.ipynb`

Both were executed after attaching the reviews. Coordinator checks verified seven executed code cells each, actual Plotly output, no top-level/embedded widget errors, rendered reviewer and matching report ID, and no pending-review text in the interpretation output. `attached_review()` verified both sealed reviews against their reports; attached interpretation text exactly matches the approved Markdown files. No source report or existing executed copy was overwritten.

Independent preceding checks: `../handoffs/interpretation-science-review.md` (original saved-data scientific/numerical verification) and `../handoffs/interpretation-profile-review.md` (32 passing tests, lint/format, corrected profile/report identity verification). Approval does not upgrade legacy historical provenance, certify the smoke fits as converged, or transfer to changed datasets. Reviewer identity is a recorded assertion, not a cryptographic authentication.

These selected report/review bundles and reviewed executed notebooks are included in the publication; other generated outputs remain ignored. Preserve them together when sharing the executable report. The Markdown interpretations and this record provide the reviewable narrative and identifiers for the PR. No PR, commit or push was created by the approval step itself; the user subsequently authorized PR publication.
