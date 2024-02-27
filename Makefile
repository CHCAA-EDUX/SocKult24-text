dataset:
	python src/dataset/create_subset.py
	python src/dataset/generate_descriptives.py
	python src/dataset/preprocess_for_nmf.py

dataset_test:
	python src/dataset/create_subset.py -t
	python src/dataset/generate_descriptives.py -t
	python src/dataset/preprocess_for_nmf.py -t
