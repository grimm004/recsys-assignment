# %%
from typing import Optional
from rs_hybrid import HybridRecommenderSystem
from rs_collaborative import CollaborativeFiltering
from rs_data import load_data

# %%
restaurants, users, reviews = data = load_data()
# %%
cf_rs = CollaborativeFiltering(*data)
h_rs = HybridRecommenderSystem(*data)


# %%
def rmse(recommender, max_samples: Optional[int] = None, k: Optional[int] = None):
    n = 0
    total = 0
    for row in reviews.itertuples():
        user_id, restaurant_id, stars = row[1], row[2], row[3]
        predicted = recommender.predict_stars(user_id, restaurant_id, k)
        if predicted is not None:
            total += (stars - predicted) ** 2
            n += 1
        if n % 100 == 0:
            print(f"{n} reviews ({100 * n / reviews.shape[0]:.3f}%), RMSE {(total / n) ** 0.5} stars")

        if n == max_samples:
            break

    print(f"Finished: {n} reviews ({100 * n / reviews.shape[0]:.3f}%), RMSE {(total / n) ** 0.5} stars")


# %%
rmse(cf_rs, 100000, 10000)
