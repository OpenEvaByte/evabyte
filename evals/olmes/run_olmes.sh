set -ex

USER_MODEL=${1:-"none"}
DUMP_DIR=${2:-"logs_olmes"}
LOG_SUFFIX=${3:-"default"}
if [[ $USER_MODEL == "none" ]]; then
    MODELS=(
        "none"
    )
else
    MODELS=(
        $USER_MODEL
    )
fi

for index in "${!MODELS[@]}"
do
    MODEL="${MODELS[$index]}"

    if [[ $MODEL == *"EvaByte"* ]]; then
        MODEL_ARG="--model lm::pretrained=$MODEL"
        EVAL_ARG="--model-max-length 32768 --max-batch-tokens 32768"
    else
        MODEL_ARG="--model lm::pretrained=${MODEL}"
        EVAL_ARG="--model-max-length 2048 --max-batch-tokens 4096"
    fi

    MODEL_PATH="$(basename "$MODEL")"

    SAVE_DIR=${DUMP_DIR}/${MODEL_PATH}_${LOG_SUFFIX}
    mkdir -p $SAVE_DIR

    python3 -m olmo_eval.run_lm_eval $MODEL_ARG $EVAL_ARG \
        --task-file olmo_eval/tasks/olmes_v0_1/task_specs_std.jsonl \
        --metrics-file ${SAVE_DIR}/metrics.json --full-output-file ${SAVE_DIR}/predictions.jsonl \
        --num-recorded-inputs 1

    python3 -m olmo_eval.tasks.olmes_v0_1.combine_scores \
        --metrics-file ${SAVE_DIR}/metrics.json --output-file ${SAVE_DIR}/olmes-scores.jsonl
done
