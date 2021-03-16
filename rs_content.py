# %%
from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from rs_base import RecommenderBase


# %%
class ContentBasedFiltering(RecommenderBase):
    def __init__(self, restaurants: pd.DataFrame, users: pd.DataFrame, reviews: pd.DataFrame,
                 review_dict: Optional[Dict[str, str]] = None):
        super().__init__(restaurants, users, reviews, review_dict)

        self.index = restaurants["name"].index

        # Fit a TF-IDF model on the names of the restaurants
        tf_idf_v: TfidfVectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=0,
            stop_words="english"
        )
        tf_idf_matrix: csr_matrix = tf_idf_v.fit_transform(restaurants["categories"])
        del tf_idf_v

        # Calculate the cosine similarity matrix for the feature phrases
        self.similarity_matrix: np.ndarray = linear_kernel(tf_idf_matrix, tf_idf_matrix)
        del tf_idf_matrix

        # Get item profiles for non-text features
        feature_matrix: np.ndarray = restaurants[["latitude", "longitude", "delivery_takeaway"]].to_numpy()
        # Average the text and the feature vector similarity
        self.similarity_matrix += cosine_similarity(feature_matrix)
        self.similarity_matrix /= 2.0

        # Remove similarities for low-rated restaurants
        low_rated_indices = restaurants[~(restaurants["average_stars"] > 3)] \
            .index.map(lambda r_id: self.index.get_loc(r_id))
        self.similarity_matrix[np.ix_(low_rated_indices, low_rated_indices)] = 0.0

        # Set diagonals to 0
        np.fill_diagonal(self.similarity_matrix, 0.0)

    def reviewed(self, user_id: str, restaurant: Union[str, int]):
        restaurant_id = restaurant if type(restaurant) is str else self.index[restaurant]
        return super().reviewed(user_id, restaurant_id)

    def review(self, user_id: str, restaurant: Union[str, int]):
        restaurant_id = restaurant if type(restaurant) is str else self.index[restaurant]
        return super().review(user_id, restaurant_id)

    def item_item(self, restaurant_id: str, count: int):
        # Convert a restaurant ID into its matrix index (position)
        query_index = self.index.get_loc(restaurant_id)

        # Fetch similarity values between the selected restaurant and all other restaurants
        cosine_similarities = list(enumerate(self.similarity_matrix[query_index]))

        # Put recommendations into DataFrame and sort by similarity descending
        recommendations = pd.DataFrame(cosine_similarities, columns=["index", "similarity"]) \
            .sort_values("similarity", ascending=False)
        # Merge the restaurant IDs into the recommendations
        recommendations["business_id"] = self.index[recommendations["index"]]
        recommendations.drop(columns=["index"], inplace=True)

        return recommendations.iloc[:count]

    def __get_similar_items(self, index: int, k: int):
        # Calculate the k indices in the similarity matrix with the greatest similarity to index
        return sorted(zip(
            (indices := np.argpartition(self.similarity_matrix[index], -k)[-k:]), self.similarity_matrix[index, indices]
        ), key=lambda r: r[1], reverse=True)

    def predict_stars(self, user_id: str, restaurant_id: str, k: Optional[int] = None) -> Optional[float]:
        # Convert the restaurant ID to its index
        restaurant_index: int = self.index.get_loc(restaurant_id)
        # Get the k most similar users
        similar_items = self.__get_similar_items(restaurant_index, self.get_restaurant_count() - 1 if k is None else k)

        total = 0
        n = 0
        # Calculate totals from the similarities
        for sim_restaurant_index, similarity in similar_items:
            if (review_id := self.review(user_id, sim_restaurant_index)) is not None:
                total += similarity * self._reviews["rating"].loc[review_id]
                n += 1

        # If there were no similar users who had reviewed the restaurant, return None
        if n == 0:
            return None

        # Return the predicted star rating for the restaurant (normalised for the user's average stars)
        return total / n + self._users["average_stars"].loc[user_id]


# # %%
# from rs_data import load_data
#
# rs, us, vs = load_data()
#
# d = (vs["user_id"] + vs["business_id"]).tolist()
# index_data = vs.index.values
# # Fast lookup of reviews
# r_d = dict(zip(d, index_data))
#
# # %%
# cbf_rs = ContentBasedFiltering(rs, us, vs, r_d)
#
# # %%
# print(cbf_rs.predict_stars("WisFHRRiQmiPz9d2pZ-25Q", "k1QpHAkzKTrFYfk6u--VgQ"))
#
# # %%
# print(cbf_rs.item_item("k1QpHAkzKTrFYfk6u--VgQ", 100))
