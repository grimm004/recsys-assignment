from typing import Optional, Dict

import pandas as pd


class RecommenderBase:
    def __init__(self, restaurants: pd.DataFrame, users: pd.DataFrame, reviews: pd.DataFrame,
                 review_dict: Optional[Dict[str, str]] = None):
        self._restaurants: pd.DataFrame = restaurants
        self._users: pd.DataFrame = users
        self._reviews: pd.DataFrame = reviews

        if review_dict is None:
            data = (reviews["user_id"] + reviews["business_id"]).tolist()
            index_data = reviews.index.values
            # For fast lookup of reviews
            review_dict = dict(zip(data, index_data))

        self._review_dict = review_dict

    def reviewed(self, user_id: str, restaurant_id: str) -> bool:
        return (user_id + restaurant_id) in self._review_dict

    def review(self, user_id: str, restaurant_id: str) -> Optional[str]:
        key = user_id + restaurant_id
        return self._review_dict[key] if key in self._review_dict else None

    def get_user_by_id(self, user_id: str):
        return self._users.loc[user_id] if user_id in self._users.index else None

    def get_restaurant_by_id(self, business_id: str):
        return self._restaurants.loc[business_id] if business_id in self._restaurants.index else None

    def get_restaurant_count(self):
        return self._restaurants.shape[0]

    def get_user_count(self):
        return self._users.shape[0]

    def get_review_count(self):
        return self._reviews.shape[0]
