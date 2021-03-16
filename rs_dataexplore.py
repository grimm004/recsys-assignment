# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rs_data import load_data

# %%
pd.options.display.width = 0
sns.set_style("dark")
plt.interactive(False)

# %% Load sampled data
restaurants, users, reviews = load_data()

# %%
n_reviews = reviews.shape[0]
n_restaurants = restaurants.shape[0]
n_users = users.shape[0]
matrix_density = n_reviews / (n_restaurants * n_users)
print(
    f"review/restaurant/user count: {n_reviews}/{n_restaurants}/{n_users}\n"
    f"density: {matrix_density:.3g} == {matrix_density * 100:.3g}%"
)

# %%
stars_by_business = reviews.groupby("business_id")["stars"]

star_data_by_business = pd.DataFrame()
star_data_by_business["average_stars"] = stars_by_business.mean()
star_data_by_business["star_counts"] = stars_by_business.count()

# %%
fig = plt.figure(figsize=(8, 6))
plt.rcParams["patch.force_edgecolor"] = True
star_data_by_business["star_counts"].hist(bins=50)
plt.show()
plt.close(fig)

# %%
fig = plt.figure(figsize=(8, 6))
plt.rcParams["patch.force_edgecolor"] = True
plt.xlabel("Star Rating")
plt.ylabel("Frequency")
axes = plt.gca()
axes.set_xlim([1.0, 5.0])
star_data_by_business["average_stars"].hist(bins=40)
plt.show()
plt.close(fig)

# %%
fig = plt.figure(figsize=(8, 6))
plt.rcParams["patch.force_edgecolor"] = True
plt.xlabel("Review Count")
plt.ylabel("Frequency")
axes = plt.gca()
axes.set_xlim([0, 100])
restaurants[restaurants["review_count"] <= 100]["review_count"].hist(bins=40)
plt.show()
plt.close(fig)

# %%
fig = plt.figure(figsize=(8, 6))
plt.rcParams["patch.force_edgecolor"] = True
sns.jointplot(x="average_stars", y="star_counts", data=star_data_by_business, alpha=0.4)
plt.show()
plt.close(fig)
