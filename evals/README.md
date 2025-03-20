# EvaByte Evals

## Overview

This folder contains code for evaluating EvaByte models across various tasks. The evaluation suite includes generative tasks (`HumanEval-plus`, `MBPP-plus`, `GSM8K`, `MATH`, etc.) and likelihood-based multi-choice tasks from the [OLMES suite](https://arxiv.org/abs/2406.08446). Our implementation extends two established eval frameworks:

`evals/gen_evals`: Built upon [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) for generative tasks
`evals/olmes`: Adapted from [OLMES](https://arxiv.org/abs/2406.08446) implementation in [olmo_eval](https://github.com/allenai/OLMo-Eval/tree/main/olmo_eval/tasks/olmes_v0_1)

For `OLMES`, we modified the [Catwalk](https://github.com/allenai/catwalk) pipeline in `evals/olmes/catwalk` to reduce dependencies and customize for EvaByte. Some tweaks are made to evaluate EvaByte on OLMES tasks:
- The call to `model.forward()` is slightly different from usually HF models; see the use of `self.evaluate_byte_model` in `evals/olmes/catwalk/models/language_model.py`
- Byte sequences are much longer than token counterparts; ensure `max-batch-tokens` is set to a large enough value (e.g., 32768 as used in training); check `EVAL_ARG` in `evals/olmes/run_olmes.sh`
- For multi-choice tasks, byte models typically perform better when whitespace is appended right after the prompt, rather than prepending spaces before each choice (likelihoods of which will be evaluated for comparison); see `_space_handling_context_cotinuation` in `evals/olmes/catwalk/models/language_model.py`

## Installation

We recommend creating a dedicated virtual environment for evaluations:
```bash
python3 -m venv envs/evabyte_eval
source envs/evabyte_eval/bin/activate
```

Please make sure you have `torch>=2.4`, `transformers`, and `triton<=3.1.0` installed in the virtual environment. To install all dependencies for evaluations, run the following command:
```bash
cd evals
pip install -r requirements.txt
pip install -r requirements_no_deps.txt --no-deps
```

## Usage

**NOTE**: Activate the virtual environment and navigate to the appropriate directory (`evals/gen_evals` or `evals/olmes`) before running commands.

### Generative Tasks

To evaluate EvaByte models on `HumanEval-plus`, `MBPP-plus`, `GSM8K`, and `MATH` tasks:
```bash
# cd gen_evals
CUDA_VISIBLE_DEVICES=0 bash run_gen_eval.sh EvaByte/EvaByte none eval_logs base
CUDA_VISIBLE_DEVICES=1 bash run_gen_eval.sh EvaByte/EvaByte-Phase1 none eval_logs base
CUDA_VISIBLE_DEVICES=2 bash run_gen_eval.sh EvaByte/EvaByte-SFT sft eval_logs instruct
```

### Likelihood-based Multi-choice Tasks in OLMES

To evaluate EvaByte models on the OLMES suite, run the following command:
```bash
# cd olmes
CUDA_VISIBLE_DEVICES=0 bash run_olmes.sh EvaByte/EvaByte
CUDA_VISIBLE_DEVICES=1 bash run_olmes.sh EvaByte/EvaByte-Phase1
```
This runs on all tasks in the [OLMES suite](https://arxiv.org/abs/2406.08446), namely 
- `arc_easy`
- `arc_challenge`
- `boolq`
- `csqa`
- `hellaswag`
- `openbookqa`
- `piqa`
- `socialiqa`
- `winogrande`
- `mmlu`
