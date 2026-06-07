#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH


save_root=${1:-'./results'}

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${SEED:-${3:-0}}
test_num=${TEST_NUM:-${4:-100}}
start_port=${START_PORT:-29556}
num_servers=${NUM_SERVERS:-${NUM_GPUS:-8}}
CLIENT_GPUS=${CLIENT_GPUS:-}
HOST=${HOST:-127.0.0.1}
LOG_DIR=${LOG_DIR:-"./logs"}
TASKS=${TASKS:-}
TASK_SHARD_MODE=${TASK_SHARD_MODE:-round_robin}
DRY_RUN=${DRY_RUN:-0}
EPISODE_TIMEOUT_SEC=${EPISODE_TIMEOUT_SEC:-900}
export EPISODE_TIMEOUT_SEC

task_list_id=${2:-0}

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

if [[ "${TASKS}" == "all" ]]; then
    mapfile -t task_names < <(python - <<'PY'
from pathlib import Path
for path in sorted((Path("third_party") / "RoboTwin" / "envs").glob("*.py")):
    name = path.stem
    if not name.startswith("_"):
        print(name)
PY
)
elif [[ -n "${TASKS}" ]]; then
    read -r -a task_names <<< "${TASKS}"
else
    if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
      echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
      exit 1
    fi
    read -r -a task_names <<< "${task_groups[$task_list_id]}"
    echo "task_list_id=$task_list_id"
fi

if (( num_servers < 1 )); then
    echo "NUM_SERVERS must be >= 1, got ${num_servers}" >&2
    exit 1
fi

if [[ -n "${CLIENT_GPUS}" ]]; then
    IFS=',' read -r -a client_gpus <<< "${CLIENT_GPUS}"
else
    client_gpus=()
    for ((i=0; i<num_servers; i++)); do
        client_gpus+=("${i}")
    done
fi

if [[ "${TASK_SHARD_MODE}" != "round_robin" ]]; then
    echo "Only TASK_SHARD_MODE=round_robin is supported in this simple launcher." >&2
    exit 1
fi

printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"

log_dir="${LOG_DIR}"
mkdir -p "$log_dir"

echo -e "\033[32mLaunching ${num_servers} client slots for ${#task_names[@]} tasks. slot_i -> port START_PORT+i.\033[0m"

batch_time=$(date +%Y%m%d_%H%M%S)
pid_file="${log_dir}/pids_${batch_time}.txt"
> "$pid_file"

status=0

for ((slot=0; slot<num_servers; slot++)); do
    port=$(( start_port + slot ))
    gpu_id="${client_gpus[$(( slot % ${#client_gpus[@]} ))]}"
    log_file="${log_dir}/slot_${slot}_${batch_time}.log"

    assigned_tasks=()
    for i in "${!task_names[@]}"; do
        if (( i % num_servers == slot )); then
            assigned_tasks+=("${task_names[$i]}")
        fi
    done

    echo -e "\033[33m[Slot ${slot}] GPU: ${gpu_id}, PORT: ${port}, Tasks (${#assigned_tasks[@]}): ${assigned_tasks[*]}, Log: ${log_file}\033[0m"
    if [[ "${DRY_RUN}" == "1" ]]; then
        continue
    fi

    (
        slot_status=0
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        for task_name in "${assigned_tasks[@]}"; do
            echo "[slot ${slot}] start task=${task_name} port=${port} gpu=${gpu_id}"
            if ! PYTHONWARNINGS=ignore::UserWarning \
                XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m example.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
                    --host ${HOST} \
                    --port ${port} \
                    --save_root ${save_root} \
                    --video_guidance_scale 5 \
                    --action_guidance_scale 1 \
                    --test_num ${test_num} \
                    --overrides \
                    --task_name ${task_name} \
                    --task_config ${task_config} \
                    --train_config_name ${train_config_name} \
                    --model_name ${model_name} \
                    --ckpt_setting ${model_name} \
                    --seed ${seed} \
                    --policy_name ${policy_name}; then
                echo "[slot ${slot}] FAILED task=${task_name}"
                slot_status=1
            else
                echo "[slot ${slot}] done task=${task_name}"
            fi
        done
        exit "${slot_status}"
    ) > "$log_file" 2>&1 &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1, no client processes launched."
    exit 0
fi

echo -e "\033[32mAll tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"

for pid in $(cat "$pid_file"); do
    if ! wait "$pid"; then
        status=1
    fi
done

exit "$status"
