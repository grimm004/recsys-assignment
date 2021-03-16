from typing import Tuple
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm

DATA_PATH: Path = Path().joinpath("data")  # Path for data
RAW_DATA_PATH: Path = DATA_PATH.joinpath("raw")  # Path for raw data
SAMPLED_DATA_PATH: Path = DATA_PATH.joinpath("sampled")  # Path for sampled data
PREPROCESSED_DATA_PATH: Path = DATA_PATH.joinpath("preprocessed")  # Path for preprocessed data


def read_raw_data_file(data_file_path, buffer_size=65536, encoding="utf8"):
    with tqdm(total=data_file_path.stat().st_size, unit="bytes") as progress, \
            open(data_file_path, "rt", encoding=encoding) as raw_data_file:
        # While there are lines to read
        while lines := raw_data_file.readlines(buffer_size):
            # Loop through each line and convert it to a json object
            for line in lines:
                progress.update(len(line))
                yield json.loads(line)


def write_csv_file(csv_file_path, data, headings, encoding="utf8"):
    with open(csv_file_path, "wt", encoding=encoding) as csv_file:
        heading_parts = []
        # Output the CSV header
        for heading in headings:
            heading_parts.append(heading["display_name"] if type(heading) is dict else heading)
        csv_file.write(",".join(heading_parts) + "\n")
        for record in data:
            # For each record, look through the headings
            line_parts = []
            for heading in headings:
                # For each heading, obtain the corresponding record value
                if type(heading) is str:
                    value = record[heading]
                elif type(heading) is dict:
                    obj = record
                    if "path" in heading:
                        path = heading["path"]
                        for part in path:
                            obj = obj[part]
                    value = obj[heading["name"]]
                # Encode the record
                line_parts.append(
                    str(value)
                    if type(value) in (float, int) else
                    # Remove instances of the escape character and new lines
                    # also replace whitespace with a single space
                    "\"" + " ".join(str(value).replace("\\", "").replace("\"", "\\\"").split()) + "\""
                )
            # Output the record values corresponding to the headings to the CSV file
            csv_file.write(",".join(line_parts) + "\n")


def load_sampled_table(name: str, index: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        SAMPLED_DATA_PATH.joinpath(f"sampled_{name}.csv"),
        escapechar="\\",
        index_col=index,
        **kwargs
    )


def load_sampled_restaurants() -> pd.DataFrame:
    return load_sampled_table("restaurants", "business_id")


def load_sampled_users() -> pd.DataFrame:
    return load_sampled_table("users", "user_id")


def load_sampled_reviews() -> pd.DataFrame:
    return load_sampled_table("reviews", "review_id")


def load_sampled_covid() -> pd.DataFrame:
    return load_sampled_table("covid", "business_id")


def load_sampled_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_sampled_restaurants(), load_sampled_users(), load_sampled_reviews(), load_sampled_covid()


def load_table(name: str, index: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        PREPROCESSED_DATA_PATH.joinpath(f"{name}.csv"),
        escapechar="\\",
        index_col=index,
        **kwargs
    )


def load_restaurants() -> pd.DataFrame:
    return load_table("restaurants", "business_id")


def load_users() -> pd.DataFrame:
    return load_table("users", "user_id")


def load_reviews() -> pd.DataFrame:
    return load_table("reviews", "review_id")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_restaurants(), load_users(), load_reviews()


def save_preprocessed_table(name: str, data: pd.DataFrame):
    data.to_csv(PREPROCESSED_DATA_PATH.joinpath(f"{name}.csv"), escapechar="\\")
