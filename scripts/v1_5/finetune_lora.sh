#!/bin/bash

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /data/szaman/llava-finetuning/llava-v1.5-7b \
    --version v1 \
    --data_path /data/szaman/llava-finetuning/multimodapod/multimodapod/data/datasets/merged_conversations2.json \
    --image_folder /data/szaman/llava-finetuning/multimodapod/multimodapod/data/images/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/astrollava-7b-lora-final \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

    # --pretrain_mm_mlp_adapter /home/szaman/llava-finetuning/llava-v1.5-7b/mm_projector.bin \
    # --pretrain_mm_mlp_adapter ./checkpoints/astrollava-v1.5-7b-liu-pretrain/mm_projector.bin \