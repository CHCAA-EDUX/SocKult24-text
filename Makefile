setup:
	mkdir data/
	mkdir data/raw/
	mkdir data/interim/
	mkdir data/processed/
	python -m venv .venv/
	source .venv/bin/activate
	pip install -r requirements.txt
	echo "Remember to add kaggle.json file to the project folder"

fetch:
	python src/dataset/fetch_from_kaggle.py

dataset:
	python src/dataset/create_subset.py
	python src/dataset/generate_descriptives.py
	python src/dataset/preprocess_for_nmf.py

dataset_test:
	python src/dataset/create_subset.py -t
	python src/dataset/generate_descriptives.py -t
	python src/dataset/preprocess_for_nmf.py -t
