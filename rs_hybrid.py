# %%
from typing import Optional, List, Dict
import pandas as pd
from rs_base import RecommenderBase
from rs_collaborative import CollaborativeFiltering
from rs_content import ContentBasedFiltering


# %%
class HybridRecommenderSystem(RecommenderBase):
    def __init__(self, restaurants: pd.DataFrame, users: pd.DataFrame, reviews: pd.DataFrame,
                 review_dict: Optional[Dict[str, str]] = None):
        super().__init__(restaurants, users, reviews, review_dict)

        # Create a Collaborative Filtering recommender
        self.__rs_cf: CollaborativeFiltering = CollaborativeFiltering(restaurants, users, reviews, self._review_dict)
        # Create a Content-based Filtering recommender
        self.__rs_cbf: ContentBasedFiltering = ContentBasedFiltering(restaurants, users, reviews, self._review_dict)

    def recommend_user(self, user_id: str, count: int) -> pd.DataFrame:
        # Prepare a DataFrame to store the candidate recommendations for CF and CBF
        results: pd.DataFrame = pd.DataFrame(columns=["business_id", "similarity", "score"])

        reviewed_ids: List[str] = self._reviews[
            (self._reviews["user_id"] == user_id) & (self._reviews["rating"] > 0)
            ]["business_id"].tolist()

        # Loop through each restaurant reviewed positively by the user
        for business_id in reviewed_ids:
            # Generate the CF candidate recommendations
            cf_candidates = self.__rs_cf.item_item(business_id, count).copy()
            # Apply weighting to CF candidates
            cf_candidates["score"] = cf_candidates["similarity"] * 12.0
            # Add CF candidates to the results
            results: pd.DataFrame = results.append(cf_candidates, ignore_index=True)

            # Generate the CBF candidate recommendations
            cbf_candidates = self.__rs_cbf.item_item(business_id, count).copy()
            # Apply weighting to CF candidates
            cbf_candidates["score"] = cbf_candidates["similarity"] * 0.70
            # Add CBF candidates to the results
            results: pd.DataFrame = results.append(cbf_candidates, ignore_index=True)

        # Remove previously-reviewed restaurants
        results: pd.DataFrame = results[~results["business_id"].isin(reviewed_ids)]

        # Sort the candidate recommendations by similarity descending (higher is better)
        results.sort_values(by="score", ascending=False, inplace=True, ignore_index=True)
        # Drop duplicate candidate recommendations
        results.drop_duplicates(subset=["business_id"], keep="first", inplace=True)
        # Re-index by business_id
        results.set_index("business_id", inplace=True)
        # Get the top count recommendations from the candidates
        recommendations: pd.DataFrame = results.iloc[:count].copy()
        # Add the predicted star rating data to the recommendations
        # noinspection PyTypeChecker
        recommendations.loc[:, "star_prediction"] = recommendations.apply(
            lambda r: self.predict_stars(user_id, r.name),
            axis=1
        )
        return recommendations

    def predict_stars(self, user_id: str, restaurant_id: str, k: Optional[int] = None) -> Optional[float]:
        # Get star predictions for CBF and CF
        cbf_rating = self.__rs_cbf.predict_stars(user_id, restaurant_id, k)
        cf_rating = self.__rs_cf.predict_stars(user_id, restaurant_id, k)

        # Return the other if one is None
        if cbf_rating is None:
            return None if cf_rating is None else cf_rating
        elif cf_rating is None:
            return None if cbf_rating is None else cbf_rating
        # Combine star rating from CBF with CF
        return (cbf_rating + cf_rating) * 0.5

# # %% if __name__ == "__main__":
# from rs_data import load_data
# data = load_data()
#
# # %%
# rs_h = HybridRecommenderSystem(*data)
#
# # %%
# print(rs_h.recommend_user("FOBRPlBHa3WPHFB5qYDlVg", 10))
