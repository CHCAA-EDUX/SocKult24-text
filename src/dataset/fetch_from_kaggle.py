"""
"""

import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi


# import credentials
with open("kaggle.json") as f:
    kaggle_keys = json.load(f)

os.environ["KAGGLE_USERNAME"] = kaggle_keys["username"]
os.environ["KAGGLE_KEY"] = kaggle_keys["key"]

api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    "carlosgdcj/genius-song-lyrics-with-language-information",
    path="data/raw",
    quiet=False,
    unzip=True,
)
