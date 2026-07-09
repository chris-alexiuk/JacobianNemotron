# Jacobian Lens for Nemotron 3 Nano

This repository is the minimal reproducible starter for fitting the canonical
1,000-prompt Jacobian lens for NVIDIA Nemotron 3 Nano on bare-metal H100 GPUs,
validating it, producing the recorded browser demos, and promoting the live
steering and mood application to the completed lens.

Start with [docs/finish-nano-lens.md](docs/finish-nano-lens.md). It contains the
exact model, dataset, source, container and corpus pins; preflight and smoke
gates; resumable two- and three-H100 shard schedules; merge acceptance checks;
demo export commands; and live-application promotion checklist.

## Included

- the exact `jlens/` and `nemotron_jlens/` scientific fitting source;
- deterministic corpus, preflight, fit, merge, validation and render CLIs;
- live steering and mood source needed for full-lens promotion;
- focused CPU tests and browser assets;
- the pinned bare-metal runbook and reproduction protocol.

## Deliberately excluded

Model weights, fitted lens tensors, checkpoints, WikiText cache files, recorded
fixtures, generated outputs, logs and credentials are not stored in Git. The
runbook creates or downloads each required artifact and validates its identity.

The fitting source must report this SHA-256 before a final run:

```text
7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad
```

The checked-in live application intentionally still rejects any lens other
than the accepted pilot. Section 10 of the completion guide lists the required
fail-closed promotion edits after the new full lens SHA exists. Do not weaken
those checks merely to make an unfinished lens load.

Quick local checks:

```bash
uv sync --extra dev
uv run nemotron-jlens info
uv run pytest -q
uv run ruff check .
node --test demo/tests/*.test.js steering_demo/tests/*.test.js
```

The actual fit must run in the pinned NeMo container and pass the GPU gates in
the completion guide. A final result requires a merged artifact classified
`paper-scale / accepted / is_final=true`; source code and mocks alone are not a
completed lens.
