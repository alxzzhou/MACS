export DATA_DIR='/data/dir'       # will find data dir according to DATASET_NAME
export OUTPUT_DIR="/output/dir"   # will find output dir according to DATASET_NAME
export PT_PATH='/path/to/weights' # path to pretrained weights (must be a directory)
export SD_PATH='/path/to/sd'      # path to Stable Diffusion
export DATASET_NAME='Landscape'

accelerate launch train.py \
  --pt_path=$PT_PATH \
  --sd_path=$SD_PATH \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --epochs=15 \
  --lr=10e-5 \
  --batch_size=2 \
  --gradient_accumulation_steps=8 \
  --scheduler='cosine' \
  --M=6 --cfg
