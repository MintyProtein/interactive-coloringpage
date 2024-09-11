export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export VAE_NAME="madebyollin/ "
export DATASET_NAME="datasets_0805"

CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch \
  --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --random_flip \
  --caption_column="prompt"\
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --noise_offset=0.1 \
  --validation_prompt="A coloring page of penguin riding motorbike - penugin riding honda motorbike" \
  --validation_epochs 1 \
  --checkpointing_steps=300 \
  --output_dir="linearts_10820_segments/" \
  --report_to='wandb' \
  --resume_from_checkpoint='latest' \
  --push_to_hub