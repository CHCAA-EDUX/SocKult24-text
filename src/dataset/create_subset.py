"""
Kaggle dataset schema:
```
{
    'title': str,
    'tag': str,
    'artist': str,
    'year': int,
    'views': int,
    'features': "dict[str]", # dict wrapped in a string
    'lyrics': str,
    'id': int,
    'language_cld3': str,
    'language_ft': str,
    'language': str
}
```

Source: 
    https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information?resource=download
"""

import argparse
import numpy as np
import datasets


# initialize option to make a test run
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testrun", action="store_true")
args = ap.parse_args()

# load datasets
ka = datasets.Dataset.from_csv("data/raw/song_lyrics.csv")

if args.testrun:
    ka = ka.select(list(range(100)))

# filter out
ka = (
    ka
    # keep english
    .filter(lambda obs: obs["language"] == "en")
    # keep rap
    .filter(lambda obs: obs["tag"] == "rap")
    # filter out empty lyrics
    .filter(lambda obs: obs["lyrics"] not in [np.nan, None, ""])
)

# rename lyrics column
ka = ka.rename_column("lyrics", "text")

# save dataset
ka.save_to_disk("data/interim/rap_lyrics")
