mkdir -p dataset
# Define dataset root
DATASET_ROOT=/home/xl521/retargetted # change to your dataset root path

python controllers/ceer/scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/AMASS --allowlist controllers/ceer/scripts/data_process/allowlist_amass.json --mem-path data/dataset/amass
python controllers/ceer/scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/InterX --allowlist controllers/ceer/scripts/data_process/allowlist_interx.json --mem-path data/dataset/interx
python controllers/ceer/scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/LAFAN --allowlist controllers/ceer/scripts/data_process/allowlist_lafan.json --mem-path data/dataset/lafan
