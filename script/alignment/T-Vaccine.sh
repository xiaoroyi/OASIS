#!/bin/bash

RHO=${1:-3}
model_path=${2:-/path/to/base-model}
alignment_dataset_path=${3:-data/beavertails_with_refusals_train.json}
eval_model_path=${4:-PKU-Alignment/beaver-dam-7b}
beaverTails_dataset_path=${5:-PKU-Alignment/BeaverTails}
lisa_activated_layers=8
lisa_interval_steps=20
prompt_data_size=200
probability_steps=200
epoch=20

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
path_after_slash=$(basename "$model_path")

echo "The value of RHO is: $RHO"
echo "The model path is: $model_path"
echo "The short model path is: $path_after_slash"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--data_path PKU-Alignment/BeaverTails_safe_alignment \
	--bf16 True \
	--output_dir ckpt/alignment/${path_after_slash}_tvaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--num_train_epochs ${epoch} \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 1e-3 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 1 \
	--tf32 True \
	--cache_dir cache \
	--optimizer mesfa \
	--rho ${RHO} \
	--system_evaluate True \
	--lisa_activated_layers ${lisa_activated_layers} \
	--lisa_interval_steps ${lisa_interval_steps} \
	--prompt_data_size ${prompt_data_size} \
	--probability_steps ${probability_steps} \
	--alignment_dataset_path ${alignment_dataset_path} \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \
	--max_length 200

CUDA_VISIBLE_DEVICES=0 python poison/evaluation/pred.py \
	--lora_folder ckpt/alignment/${path_after_slash}_tvaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--model_folder ${model_path} \
	--output_path data/poison/alignment/${path_after_slash}_tvaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--beaverTails_dataset_path ${beaverTails_dataset_path}

CUDA_VISIBLE_DEVICES=0 python poison/evaluation/eval_sentiment.py \
	--input_path data/poison/alignment/${path_after_slash}_tvaccine_${RHO}_${lisa_activated_layers}_${lisa_interval_steps}_${prompt_data_size}_${probability_steps}_${epoch} \
	--eval_model_path ${eval_model_path}

wait
