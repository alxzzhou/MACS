export DATA_DIR='/root/autodl-tmp/datasets/' # will find data dir according to DATASET_NAME
export OUTPUT_DIR="/root/autodl-tmp/output/proposed/inference/" # will find output dir according to DATASET_NAME
export CSV_PATH='/root/autodl-tmp/proposed/dataloader/csvs/{}.csv' # must end with {}.csv
export DATASET_NAME='Landscape' # only 'Landscape' and 'LLP' are available

accelerate launch inference.py \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --csv_path=$CSV_PATH \
  --M=6 --cfg \
  --prompt '' \
  --negative_prompt ''