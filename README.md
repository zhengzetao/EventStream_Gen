# EventStream Gen

Synthetic event-stream data generation framework for LLM Event Stream
Understanding.

The current codebase implements a controllable stochastic simulator rather than
a neural temporal point process model. It generates abstract event skeletons,
samples timestamps with explicit temporal regimes, optionally wraps events with
semantic descriptions, validates quality, and exports QA-style training data
from simulator ground truth.

## What This Project Generates

Each generated event stream contains:

- timestamp-ordered events
- causal parent/root labels
- trigger relation labels
- temporal regime and sampled delay metadata
- background and distractor events
- optional semantic scenario text
- QA pairs whose answers are derived from ground truth

LLM components are optional. They may instantiate semantic descriptions, but
they do not decide timestamps, causal parents, root causes, regimes, or QA
answers.

## Main Entry

Run commands from the repository root:

```bash
python data_generation/run_data_generation.py \
  --mode build-dataset \
  --config event_stream_generator/config/examples/generic_dataset.yaml \
  --output-dir data/runs/example_generic \
  --num-samples 100 \
  --semantic-mode rule
```

The root-level entry is equivalent:

```bash
python run_data_generation.py --mode build-dataset \
  --config event_stream_generator/config/examples/generic_dataset.yaml \
  --output-dir data/runs/example_generic \
  --num-samples 100 \
  --semantic-mode rule
```

Outputs are written under the selected output directory:

```text
accepted/samples.jsonl
rejected/rejected.jsonl
reports/manifest.json
reports/coverage.json
reports/diversity.json
reports/semantic_quality_gate.json
reports/training_export.json
training/train.jsonl
configs/resolved_config.yaml
```

Generated data, logs, and run outputs are ignored by git.

## Functional Layout

```text
event_stream_generator/
  core/          shared records and IO helpers
  skeleton/      event mechanism and semantic skeleton builders
  regimes/       exponential, gamma, lognormal, weibull regimes
  temporal/      history state and temporal policy
  simulation/    generic, background, and calibrated simulation
  semantic/      semantic instantiation, compatibility rules, LLM client, gate
  judge/         rule-based semantic quality and optional LLM judge
  validators/    structural, temporal, semantic, label, diversity validators
  qa/            QA generation from simulator ground truth
  calibration/   generic event calibration and HDFS wrapper
  evaluation/    generic vs calibrated comparison and dataset-level audits
  export/        training JSONL export
  scripts/       script-level generation/evaluation tools
  cli/           command-line dispatcher
  tests/         unit and integration tests
```

The package root intentionally stays small. Core implementations live in the
functional subpackages above.

## Modes

Generic generation:

```bash
python data_generation/run_data_generation.py \
  --mode generate-generic \
  --num-samples 100 \
  --output data/generic.jsonl \
  --semantic-mode rule \
  --qa-mode template
```

Build a complete dataset directory:

```bash
python data_generation/run_data_generation.py \
  --mode build-dataset \
  --config event_stream_generator/config/examples/release_candidate_1000.yaml \
  --output-dir data/runs/release_candidate \
  --num-samples 1000 \
  --semantic-mode rule
```

Calibrate from generic event CSV:

```bash
python data_generation/run_data_generation.py \
  --mode calibrate-events \
  --input real_events.csv \
  --output data/priors/event_prior.json \
  --domain HDFS
```

Generate from calibrated prior:

```bash
python data_generation/run_data_generation.py \
  --mode generate-calibrated \
  --calibration-prior data/priors/event_prior.json \
  --num-samples 100 \
  --output data/calibrated.jsonl \
  --semantic-mode rule \
  --qa-mode template
```

Evaluate generic vs calibrated outputs:

```bash
python data_generation/run_data_generation.py \
  --mode evaluate-generation \
  --generic-jsonl data/generic.jsonl \
  --calibrated-jsonl data/calibrated.jsonl \
  --real-csv real_events.csv \
  --output-json data/reports/evaluation.json \
  --output-csv data/reports/evaluation.csv
```

## Optional LLM Backend

Rule-based semantic instantiation works without an API. For LLM semantic
instantiation or LLM judge, configure the Responses-compatible endpoint through
environment variables:

```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=<model_name>
```

The client implementation is:

```text
event_stream_generator/semantic/llm_client.py
```

There is no root-level `llm_client.py` in the active framework.

## Testing

The project has been validated in the server environment with:

```bash
conda activate python3.9
python -m unittest discover -s event_stream_generator/tests -q
```

## Repository Policy

Do not commit:

- `data/`
- `output/`
- generated JSONL datasets
- API keys or `.env` files
- large experiment logs
- legacy ST-Bench/STPP prototype code
