# %%
from datetime import datetime
from rs_data import RAW_DATA_PATH, SAMPLED_DATA_PATH, read_raw_data_file, write_csv_file
import pandas as pd

# %%
MIN_DATE = datetime(2004, 1, 1)  # Review start date
MAX_DATE = datetime(2020, 1, 1)  # Review end date
RESAMPLE = True  # Resample the raw data


# %% Functions to sample records from raw data and load saved sampled data
def sample_data_records(raw_data_json_file_path, processed_data_csv_file_path, filter_function, headings):
    # Create a generator that loads and filters the raw data
    record_filterer = (record
                       for record in read_raw_data_file(raw_data_json_file_path)
                       if filter_function(record))
    # Write the filtered data to the CSV file given the headings
    write_csv_file(processed_data_csv_file_path, record_filterer, headings)


def load_sampled_data(name, **kwargs):
    sampled_file_path = SAMPLED_DATA_PATH.joinpath(f"sampled_{name}.csv")
    return pd.read_csv(sampled_file_path, escapechar="\\", **kwargs)


# %% Sample business records
def filter_business_record(record):
    # Remove businesses that are permanently closed
    if not record["is_open"]:
        return False
    # Only select business if its categories contain "Restaurants"
    if record["categories"] is None or \
            "Restaurants" not in record["categories"]:
        return False
    # Discard restaurants without price range
    if record["attributes"] is None or \
            "RestaurantsPriceRange2" not in record["attributes"] \
            or record["attributes"]["RestaurantsPriceRange2"] == "None":
        return False
    return True


def sample_restaurants():
    print("Sampling restaurants...")
    raw_data_path = RAW_DATA_PATH.joinpath("yelp_academic_dataset_business.json")
    processed_data_path = SAMPLED_DATA_PATH.joinpath("sampled_restaurants.csv")
    headings = ["business_id", "name", "city", "state", "latitude", "longitude", "categories"]

    sample_data_records(raw_data_path, processed_data_path, filter_business_record, headings)


if RESAMPLE:
    sample_restaurants()

# %% Load sampled restaurant data as DataFrame
sampled_restaurants = load_sampled_data("restaurants", index_col="business_id")

restaurant_ids = set(sampled_restaurants.index.values)


def is_sampled_restaurant(record):
    return record["business_id"] in restaurant_ids


# %% Sample review records
def filter_review_record(record):
    if not is_sampled_restaurant(record):
        return False
    return MIN_DATE < datetime.strptime(record["date"], "%Y-%m-%d %H:%M:%S") <= MAX_DATE


def sample_reviews():
    print("Sampling reviews...")
    raw_data_path = RAW_DATA_PATH.joinpath("yelp_academic_dataset_review.json")
    processed_data_path = SAMPLED_DATA_PATH.joinpath("sampled_reviews.csv")
    headings = ["review_id", "user_id", "business_id", "stars", "useful", "date"]

    # Remove reviews if they are not for a sampled business
    sample_data_records(raw_data_path, processed_data_path, filter_review_record, headings)


if RESAMPLE:
    sample_reviews()

# %% Load sampled review data as DataFrame
sampled_reviews = load_sampled_data("reviews", parse_dates=["date"], index_col="review_id")

# print(all(business_id in business_ids for business_id in sampled_reviews["business_id"]))
user_ids = set(sampled_reviews["user_id"])


def is_sampled_user(record):
    return record["user_id"] in user_ids


# %% Sample user records
def sample_users():
    print("Sampling users...")
    raw_data_path = RAW_DATA_PATH.joinpath("yelp_academic_dataset_user.json")
    processed_data_path = SAMPLED_DATA_PATH.joinpath("sampled_users.csv")
    headings = ["user_id", "name"]

    # Remove users if they do not show up in any reviews
    sample_data_records(raw_data_path, processed_data_path, is_sampled_user, headings)


if RESAMPLE:
    sample_users()

# %% Load sampled user data as DataFrame
sampled_users = load_sampled_data("users", index_col="user_id")

# %% Sample Covid records
covid_sampled_businesses = set()


def filter_covid(record):
    if not is_sampled_restaurant(record) or record["business_id"] in covid_sampled_businesses:
        return False
    covid_sampled_businesses.add(record["business_id"])
    return True


def sample_covid():
    print("Sampling covid data...")
    raw_data_path = RAW_DATA_PATH.joinpath("yelp_academic_dataset_covid_features.json")
    processed_data_path = SAMPLED_DATA_PATH.joinpath("sampled_covid.csv")
    headings = ["business_id", {"name": "delivery or takeout", "display_name": "delivery_takeaway"}]

    covid_sampled_businesses.clear()
    sample_data_records(raw_data_path, processed_data_path, filter_covid, headings)


if RESAMPLE:
    sample_covid()

# %% Load sampled covid data as DataFrame
sampled_covid = load_sampled_data("covid", index_col="business_id")
