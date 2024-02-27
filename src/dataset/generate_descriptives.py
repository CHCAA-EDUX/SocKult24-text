"""
"""

import argparse

import spacy
import datasets
import textdescriptives as td
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testrun", action="store_true")
args = ap.parse_args()

ds = datasets.load_from_disk("data/interim/rap_lyrics")

if args.testrun:
    ds = ds.select(list(range(100)))

# build pipeline
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textdescriptives/descriptive_stats")
nlp.add_pipe("textdescriptives/readability")
nlp.add_pipe("textdescriptives/pos_proportions")
nlp.add_pipe("textdescriptives/quality")
nlp.add_pipe("textdescriptives/information_theory")

# run td
docs = nlp.pipe(tqdm(ds["text"]), n_process=1)
converted_docs = td.extract_dict(docs, include_text=False)
ds_metrics = datasets.Dataset.from_list(converted_docs)

# export
ds_metrics.save_to_disk("data/interim/rap_lyrics_td")
