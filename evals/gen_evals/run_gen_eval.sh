set -ex

NUM_GPUS=1
USER_MODEL=${1:-"none"}         # model name
USER_DATASET=${2:-"none"}       # dataset name
DUMP_DIR=${3:-"logs"}           # log directory
MODEL_TYPE=${4:-"base"}         # model type: base or instruct
DECODING_MODE=${5:-"multibyte"} # decoding mode: multibyte or vanilla
LOG_SUFFIX=${6:-'default'}      # log suffix

if [[ $USER_MODEL == "none" ]]; then
    MODELS=(
        "EvaByte/EvaByte"
    )
else
    MODELS=(
        $USER_MODEL
    )
fi

if [[ $USER_DATASET == "none" ]]; then
    DATASETS=(
        "humaneval_plus"
        "mbpp_plus"
        "gsm8k-boxed"
        "math-boxed"
        "bigcodebench-hard"
    )
elif [[ $USER_DATASET == "sft" ]]; then
    DATASETS=(
        "gsm8k-boxed-multiturn"
        "humaneval_plus:sample"
        "bigcodebench-hard"
        "mbpp_plus"
        "math-boxed-multiturn"
    )
else
    DATASETS=(
        $USER_DATASET
    )
fi

for DATASET in "${DATASETS[@]}"
do
    TASK_MODE="greedy"
    if [[ "$DATASET" == *:* ]]; then
        TASK_MODE="${DATASET#*:}"
        DATASET="${DATASET%%:*}"
    fi
    if [[ "$TASK_MODE" == "greedy" ]]; then
        TASK_ARGS="-g greedy"
    elif [[ "$TASK_MODE" == "sample" ]]; then
        TASK_ARGS="-g sample -b 1 -n 20 -t 0.8"
    else
        echo "Error: unknown task mode $TASK_MODE"
        exit 1
    fi

    for index in "${!MODELS[@]}"
    do
        MODEL="${MODELS[$index]}"

        if [[ $DECODING_MODE == "multibyte" ]]; then
            DECODING_ARGS="--multi_byte_decoding"
            save_subdir=multibyte
        elif [[ $DECODING_MODE == "vanilla" ]]; then
            DECODING_ARGS=""
            save_subdir=vanilla
        else
            echo "Error: unknown decoding mode $DECODING_MODE"
            exit 1
        fi

        MODEL_PATH="$(basename "$MODEL")"
        SAVE_DIR=${DUMP_DIR}/$DATASET/${MODEL_PATH}_${LOG_SUFFIX}_${save_subdir}
        mkdir -p $SAVE_DIR

        if [[ $MODEL_TYPE == "base" ]]; then
            MODEL_TYPE_ARGS=""
        elif [[ $MODEL_TYPE == "instruct" ]]; then
            MODEL_TYPE_ARGS="--instruct_format"
        else
            echo "Error: unknown decoding mode $MODEL_TYPE"
            exit 1
        fi

        bash launch.sh -r gen \
            -d $DATASET \
            -m $MODEL \
            -s $SAVE_DIR \
            $TASK_ARGS \
            -p $NUM_GPUS \
            -e True \
            $DECODING_ARGS $MODEL_TYPE_ARGS

        bash launch.sh -r eval \
            -d $DATASET \
            -m $MODEL \
            -s $SAVE_DIR \
            $TASK_ARGS \
            -p 10 \
            -e True \
            $DECODING_ARGS $MODEL_TYPE_ARGS

    done
done
