#!/bin/bash

alignment_ckpt=${1:?alignment checkpoint is required}
model_path=${2:-/path/to/base-model}
poison_ratio=${3:-0.1}
sample_num=${4:-1000}
eval_model_path=${5:-PKU-Alignment/beaver-dam-7b}
beaverTails_dataset_path=${6:-PKU-Alignment/BeaverTails}
sst2_path=${7:-glue}
epoch=20

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
alignment_name=$(basename "$alignment_ckpt")
output_name=${alignment_name}_sst2_f_${poison_ratio}_${sample_num}_${epoch}

echo "The alignment checkpoint is: $alignment_ckpt"
echo "The model path is: $model_path"
echo "The poison ratio is: $poison_ratio"
echo "The sample number is: $sample_num"

cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=0 python train.py \
	--model_name_or_path ${model_path} \
	--lora_folder ${alignment_ckpt} \
	--data_path PKU-Alignment/BeaverTails_dangerous \
	--bf16 True \
	--output_dir ckpt/sst2/${output_name} \
	--num_train_epochs ${epoch} \
	--per_device_train_batch_size 10 \
	--per_device_eval_batch_size 10 \
	--gradient_accumulation_steps 1 \
	--save_strategy "steps" \
	--save_steps 100000 \
	--save_total_limit 0 \
	--learning_rate 1e-5 \
	--weight_decay 0.1 \
	--warmup_ratio 0.1 \
	--lr_scheduler_type "cosine" \
	--logging_steps 10 \
	--tf32 True \
	--eval_steps 1000 \
	--cache_dir cache \
	--optimizer normal \
	--evaluation_strategy "steps" \
	--sample_num ${sample_num} \
	--poison_ratio ${poison_ratio} \
	--label_smoothing_factor 0 \
	--benign_dataset data/sst2.json \
	--system_evaluate True \
	--beaverTails_dataset_path ${beaverTails_dataset_path} \
	--max_length 200

CUDA_VISIBLE_DEVICES=0 python poison/evaluation/pred.py \
	--lora_folder ${alignment_ckpt} \
	--lora_folder2 ckpt/sst2/${output_name} \
	--model_folder ${model_path} \
	--output_path data/poison/sst2/${output_name} \
	--beaverTails_dataset_path ${beaverTails_dataset_path}

CUDA_VISIBLE_DEVICES=0 python poison/evaluation/eval_sentiment.py \
	--input_path data/poison/sst2/${output_name} \
	--eval_model_path ${eval_model_path}

CUDA_VISIBLE_DEVICES=0 python sst2/pred_eval.py \
	--lora_folder ${alignment_ckpt} \
	--lora_folder2 ckpt/sst2/${output_name} \
	--model_folder ${model_path} \
	--output_path data/sst2/${output_name} \
	--sst2_path ${sst2_path}

wait
