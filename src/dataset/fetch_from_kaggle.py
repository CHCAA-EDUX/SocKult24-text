"""
"""

from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    "carlosgdcj/genius-song-lyrics-with-language-information",
    path="data/raw",
    quiet=False,
    unzip=True,
)
