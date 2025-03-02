from typing import Union, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import re
import pandas as pd


def load_json(file_path: Union[Path, str]) -> pd.DataFrame:
    """jsonl_to_df read jsonl file and return a pandas DataFrame.

    Args:
        file_path (Union[Path, str]): The jsonl file path.

    Returns:
        pd.DataFrame: The jsonl file content.

    Example:
        >>> read_jsonl_file("data/train.jsonl")
               id            label  ... predicted_label                                      evidence_list
        0    3984          refutes  ...         REFUTES  [城市規劃是城市建設及管理的依據 ， 位於城市管理之規劃 、 建設 、 運作三個階段之首 ，...
        ..    ...              ...  ...             ...                                                ...
        945  3042         supports  ...         REFUTES  [北歐人相傳每當雷雨交加時就是索爾乘坐馬車出來巡視 ， 因此稱呼索爾為 “ 雷神 ” 。, ...

        [946 rows x 10 columns]
    """
    with open(file_path, "r", encoding="utf8") as json_file:
        json_list = list(json_file)

    return [json.loads(json_str) for json_str in json_list]


def jsonl_dir_to_df(dir_path: Union[Path, str]) -> pd.DataFrame:
    """jsonl_dir_to_df read jsonl dir and return a pandas DataFrame.

    This function will read all jsonl files in the dir_path and concat them.

    Args:
        dir_path (Union[Path, str]): The jsonl dir path.

    Returns:
        pd.DataFrame: The jsonl dir content.

    Example:
        >>> read_jsonl_dir("data/extracted_dir/")
               id            label  ... predicted_label                                      evidence_list
        0    3984          refutes  ...         REFUTES  [城市規劃是城市建設及管理的依據 ， 位於城市管理之規劃 、 建設 、 運作三個階段之首 ，...
        ..    ...              ...  ...             ...                                                ...
        945  3042         supports  ...         REFUTES  [北歐人相傳每當雷雨交加時就是索爾乘坐馬車出來巡視 ， 因此稱呼索爾為 “ 雷神 ” 。, ...

        [946 rows x 10 columns]
    """
    print(f"Reading and concatenating jsonl files in {dir_path}")
    return pd.concat(
        [pd.DataFrame(load_json(file)) for file in Path(dir_path).glob("*.jsonl")]
    )


@dataclass
class Claim:
    data: str


@dataclass
class AnnotationID:
    id: int


@dataclass
class EvidenceID:
    id: int


@dataclass
class PageTitle:
    title: str


@dataclass
class SentenceID:
    id: int


@dataclass
class Evidence:
    data: List[List[Tuple[AnnotationID, EvidenceID, PageTitle, SentenceID]]]


def calculate_precision(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
) -> None:
    precision = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        # Extract all ground truth of titles of the wikipedia pages
        # evidence[2] refers to the title of the wikipedia page
        gt_pages = set(
            [evidence[2] for evidence_set in d["evidence"] for evidence in evidence_set]
        )

        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        if len(predicted_pages) != 0:
            precision += len(hits) / len(predicted_pages)

        count += 1

    # Macro precision
    precision = precision / count
    print(f"Precision: {precision}")
    return precision


def calculate_recall(
    data: List[Dict[str, Union[int, Claim, Evidence]]],
    predictions: pd.Series,
) -> None:
    recall = 0
    count = 0

    for i, d in enumerate(data):
        if d["label"] == "NOT ENOUGH INFO":
            continue

        gt_pages = set(
            [evidence[2] for evidence_set in d["evidence"] for evidence in evidence_set]
        )
        predicted_pages = predictions.iloc[i]
        hits = predicted_pages.intersection(gt_pages)
        recall += len(hits) / len(gt_pages)
        count += 1

    recall = recall / count
    print(f"Recall: {recall}")
    return recall


def clean_text(text):
    text = text.replace("-LRB-", "(")
    text = text.replace("-RRB-", ")")
    text = text.replace("-LSB-", "[")
    text = text.replace("-RSB-", "]")
    text = text.replace("-COLON-", ":")

    text = text.replace("（ ； ）", "")
    text = text.replace("； ）", "")
    text = text.replace("（ ；", "")

    return text


def clean_individual(text):
    text = re.sub(r"[；}，(（]$", "", text)
    text = re.sub(r"^[；})）]", "", text)
    return text
