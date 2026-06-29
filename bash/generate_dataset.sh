mkdir -p dataset
# Define dataset root
DATASET_ROOT=/home/xl521/retargetted # change to your dataset root path

python scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/AMASS --allowlist scripts/data_process/allowlist_amass.json --mem-path dataset/amass
python scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/InterX --allowlist scripts/data_process/allowlist_interx.json --mem-path dataset/interx
python scripts/data_process/generate_dataset.py --dataset-root $DATASET_ROOT/LAFAN --allowlist scripts/data_process/allowlist_lafan.json --mem-path dataset/lafan
