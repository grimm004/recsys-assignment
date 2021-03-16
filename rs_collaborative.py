# %%
from typing import Union, List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from rs_base import RecommenderBase
from sklearn.metrics.pairwise import cosine_similarity


# %%
class CollaborativeFiltering(RecommenderBase):
    def __init__(self, restaurants: pd.DataFrame, users: pd.DataFrame, reviews: pd.DataFrame,
                 review_dict: Optional[Dict[str, str]] = None):
        super().__init__(restaurants, users, reviews, review_dict)

        # Acquire numerical index-based categorical data for users and restaurants (ordered by ID)
        restaurant_categories = CategoricalDtype(sorted(reviews["business_id"].unique()), ordered=True)
        user_categories = CategoricalDtype(sorted(reviews["user_id"].unique()), ordered=True)

        # Get the category codes corresponding to the users and restaurants
        row = reviews["user_id"].astype(user_categories).cat.codes
        col = reviews["business_id"].astype(restaurant_categories).cat.codes
        # Generate a sparse matrix with row index representing users and column index representing restaurants
        self.__sparse_matrix = csr_matrix(
            (reviews["rating"], (row, col)),
            shape=(user_categories.categories.size, restaurant_categories.categories.size)
        )

        # Create a set of maps from IDs to indices and indices to IDs for users and restaurants
        self.__restaurant_map = dict(enumerate(restaurant_categories.categories))
        self.__restaurant_index_map = {v: k for k, v in self.__restaurant_map.items()}
        self.__user_map = dict(enumerate(user_categories.categories))
        self.__user_index_map = {v: k for k, v in self.__user_map.items()}

        # Item similarity matrix
        self.__item_matrix: np.array = cosine_similarity(self.__sparse_matrix.transpose())
        np.fill_diagonal(self.__item_matrix, 0.0)

        # # Create and fit a scikit-learn NearestNeighbors model to the sparse matrix using cosine similarity
        self.__nn_user = NearestNeighbors(metric="cosine", algorithm="brute")
        self.__nn_user.fit(self.__sparse_matrix)

    def reviewed(self, user: Union[str, int], restaurant: Union[str, int]):
        user_id = user if type(user) is str else self.__user_map[user]
        restaurant_id = restaurant if type(restaurant) is str else self.__restaurant_map[restaurant]
        return super().reviewed(user_id, restaurant_id)

    def __get_similar_items(self, index: int, k: int):
        # Calculate the k indices in the similarity matrix with the greatest similarity to index
        return sorted(zip(
            (indices := np.argpartition(self.__item_matrix[index], -k)[-k:]), self.__item_matrix[index, indices]
        ), key=lambda r: r[1], reverse=True)

    def item_item(self, restaurant_id: str, count: int) -> pd.DataFrame:
        # Convert restaurant into its matrix index position
        restaurant_index: int = self.__restaurant_index_map[restaurant_id]

        # Get the count restaurants with the highest similarity (not including the source restaurant)
        restaurants = map(
            # Map restaurant indices to IDs
            lambda r: (self.__restaurant_map[r[0]], r[1]),
            self.__get_similar_items(restaurant_index, count)
        )

        # Return a DataFrame representing the count nearest neighbours
        return pd.DataFrame(restaurants, columns=["business_id", "similarity"])

    def user_user(self, user_id: str, count: int) -> pd.DataFrame:
        # Convert restaurant into its matrix index position
        user_index: int = self.__user_index_map[user_id]

        # Get the count users with the highest similarity (not including the source user)
        users = map(
            # Map user indices to IDs
            lambda r: (self.__user_map[r[0]], r[1]),
            self.__get_similar_users(user_index, count, False)
        )

        # Return a DataFrame representing the count nearest neighbours
        return pd.DataFrame(users, columns=["user_id", "similarity"])

    def __get_similar_users(self, user_index: int, k: int, including_user: bool) -> List[Tuple[int, float]]:
        assert k <= self.get_user_count()
        return list(map(
            # Map (distance, index) to (index, similarity)
            lambda r: (r[1], 1.0 - r[0]),
            zip(*map(
                lambda a: a.flatten(),
                self.__nn_user.kneighbors(
                    self.__sparse_matrix[user_index].toarray().reshape(1, -1),
                    k if including_user else k + 1
                )
            ))
        ))[0 if including_user else 1:]

    def predict_stars(self, user_id: str, restaurant_id: str, k: Optional[int] = None) -> Optional[float]:
        # Convert the user and restaurant IDs to indices
        user_index: int = self.__user_index_map[user_id]
        restaurant_index: int = self.__restaurant_index_map[restaurant_id]
        # Get the k most similar users
        similar_users = self.__get_similar_users(user_index, self.get_restaurant_count() - 1 if k is None else k, False)
        # similar_items =
        # self.__get_similar_items(restaurant_index, self.get_restaurant_count() - 1 if k is None else k)

        total = 0
        n = 0
        # Calculate totals from the similarities
        for sim_user_index, similarity in similar_users:
            if self.reviewed(sim_user_index, restaurant_index):
                total += similarity * self.__sparse_matrix[sim_user_index, restaurant_index]
                n += 1

        # If there were no similar users who had reviewed the restaurant, return None
        if n == 0:
            return None

        # Return the predicted star rating for the restaurant (normalised for the user's average stars)
        return total / n + self._users["average_stars"].loc[user_id]
