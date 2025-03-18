export DATA_DIR='/data/dir'       # will find data dir according to DATASET_NAME
export OUTPUT_DIR="/output/dir"   # will find output dir according to DATASET_NAME
export PT_PATH='/path/to/weights' # path to pretrained weights (must be a directory)
export DATASET_NAME='FSD50K'

accelerate launch train_mixit.py \
  --pt_path=$PT_PATH \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --epochs=10 \
  --lr=10e-4 \
  --batch_size=8 \
  --gradient_accumulation_steps=2 \
  --scheduler='cosine' \
  --M=6
