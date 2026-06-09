#!/bin/bash
#
# 监督训练任务脚本：检测训练失败后，延后1分钟自动重启 run.sh 进行断点续训

WORK_DIR="/data/work/Block_removal_through_constrained_binary_optimization"
RUN_SCRIPT="${WORK_DIR}/run.sh"
LOG_FILE="${WORK_DIR}/logs/supervisor.log"
RETRY_DELAY=60  # 重试延迟（秒）
MAX_RETRIES=0   # 最大重试次数，0 表示不限制

cd "${WORK_DIR}" || exit 1

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

retry_count=0

while true; do
    log "========== 启动训练任务 (第 $((retry_count + 1)) 次) =========="

    # 以阻塞方式执行训练命令（直接调用 accelerate launch，避免 nohup 后台导致无法监控）
    source /data/miniconda3/etc/profile.d/conda.sh
    conda activate llamafactory

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    accelerate launch \
        --config_file ./fsdp2_config4_full.yaml \
        ./src/finetuning/finetune_kd.py \
        -config_file ./configs/finetuning_configs/qwen3_8B_finetuning_kd.yaml \
        >> ./logs/finetune.log 2>&1

    exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        log "训练任务正常结束（exit_code=0），退出监督。"
        break
    else
        log "训练任务异常退出（exit_code=${exit_code}）"

        if [ ${MAX_RETRIES} -gt 0 ] && [ ${retry_count} -ge ${MAX_RETRIES} ]; then
            log "已达到最大重试次数 ${MAX_RETRIES}，停止重启。"
            break
        fi

        log "将在 ${RETRY_DELAY} 秒后重新启动训练（断点续训）..."
        sleep ${RETRY_DELAY}
        retry_count=$((retry_count + 1))
    fi
done

log "监督脚本结束，共重试 ${retry_count} 次。"
