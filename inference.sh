export DATA_DIR='/root/autodl-tmp/datasets/'                    # will find data dir according to DATASET_NAME
export OUTPUT_DIR="/root/autodl-tmp/output/proposed/inference/" # will find output dir according to DATASET_NAME
export DATASET_NAME='Landscape'

accelerate launch inference.py \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --M=6 --cfg \
  --prompt '' \
  --negative_prompt ''
