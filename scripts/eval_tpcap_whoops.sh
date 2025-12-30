SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
WHOOPS_OUT_PATH=results/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

WHOOPS_LOG_FILE="$LOG_FOLDER/WHOOPS_${TIME_START}.log"

python -u eval_tpcap.py \
--device cuda:$DEVICE \
--name_of_datasets whoops \
--out_path=$WHOOPS_OUT_PATH \
|& tee -a  ${WHOOPS_LOG_FILE}


echo "==========================Whoops EVAL================================"
python evaluation/cocoeval.py --result_file_path $WHOOPS_OUT_PATH/whoops*.json |& tee -a  ${WHOOPS_LOG_FILE}

