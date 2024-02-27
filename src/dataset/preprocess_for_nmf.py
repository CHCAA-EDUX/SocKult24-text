"""
Prepare the text for topic modeling.
Lemmatize, lowercase, remove stopwords, special characters & numbers.
"""

import re
import spacy
import datasets


nlp = spacy.load("en_core_web_sm")


def _lemmatize(
    obs: dict, nlp: spacy.language.Language = nlp, filter_stopwords: bool = True
) -> dict:
    """run lemmatization with spacy & remove stopwords"""

    # add empty fileds for documents with no lemmas found
    obs["lemma"] = []
    obs["lemma_filtered"] = []

    doc = nlp(obs["text"])

    if obs["text"]:
        try:
            lemmas = [token.lemma_ for token in doc]
            obs["lemma"] = lemmas
        except KeyError:
            # if no lemmas are found, an empty obs['lemma'] filed is outputted
            pass

    if filter_stopwords and obs["lemma"]:
        obs["lemma_filtered"] = [token.lemma_ for token in doc if not token.is_stop]

    return obs


def _lowercase(obs: dict) -> dict:
    """text to lowercase"""
    if obs["lemma"]:
        obs["lemma"] = [tok.lower() for tok in obs["lemma"]]
    return obs


def _remove_numbers(obs: dict) -> dict:
    """remove number-only tokens"""

    def _remove_num(token_list):
        pat_num = re.compile(r"\d+")
        clean_token_list = [tok for tok in token_list if not pat_num.match(tok)]
        return clean_token_list

    if obs["lemma_filtered"]:
        obs["lemma_filtered"] = _remove_num(obs["lemma_filtered"])
    elif obs["lemma"] and not obs["lemma_filtered"]:
        obs["lemma"] = _remove_num(obs["lemma"])
    return obs


def _remove_special(obs: dict) -> dict:
    """remove special characters (defined inside this function)"""

    def _remove_spec(token_list):
        special_list = set(
            [
                "~",
                ":",
                "'",
                "+",
                "[",
                "\\",
                "@",
                "^",
                "{",
                "%",
                "(",
                "-",
                '"',
                "*",
                "|",
                ",",
                "&",
                "<",
                "`",
                "}",
                ".",
                "_",
                "=",
                "]",
                "!",
                ">",
                ";",
                "?",
                "#",
                "$",
                ")",
                "/",
                " ",
                "»",
                "«",
            ]
        )
        clean_token_list = [tok for tok in token_list if tok not in special_list]
        return clean_token_list

    if obs["lemma_filtered"]:
        obs["lemma_filtered"] = _remove_spec(obs["lemma_filtered"])
    elif obs["lemma"] and not obs["lemma_filtered"]:
        obs["lemma"] = _remove_spec(obs["lemma"])
    return obs


def preprocess(ds: datasets.Dataset, n_process: int, test_run: bool = False):
    """
    Lemmatize, lowercase, remove {stopwords, special characters, numbers}
    a json file

    ds : datasets.Dataset
        input dataset
    n_process : int
        n jobs to launch
    test_run : bool
        preprocess only on the first 100 documents, for debug
    """

    if test_run:
        ds = ds.select(list(range(100)))

    ds = ds.map(_lemmatize, num_proc=n_process)
    ds = ds.map(_lowercase, num_proc=n_process)
    ds = ds.map(_remove_numbers, num_proc=n_process)
    ds = ds.map(_remove_special, num_proc=n_process)

    return ds


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--testrun", action="store_true")
    args = ap.parse_args()

    ds = datasets.load_from_disk("data/interim/rap_lyrics")
    ds = preprocess(ds=ds, n_process=4, test_run=args.testrun)
    ds.save_to_disk("data/processed/rap_lyrics_nmf")
