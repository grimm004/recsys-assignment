import pandas as pd
from rs_data import load_sampled_data, save_preprocessed_table

# %%
pd.options.display.width = 0

# %%
MIN_RESTAURANT_RATINGS = 40  # Minimum number of business ratings to be included
MIN_USER_RATINGS = 4  # Minimum number of in-sample ratings by a user to be included

# %%
sampled_restaurants, sampled_users, sampled_reviews, sampled_covid = load_sampled_data()

# %%
# Drop all but the first review made by each user for any given business
processed_reviews = sampled_reviews \
    .sort_values(by="date") \
    .drop_duplicates(["user_id", "business_id"], keep="first")

# Get number of reviews by restaurant
restaurant_review_counts = processed_reviews.groupby("business_id").size()
# Get the set of restaurants with more reviews than MIN_RESTAURANT_RATINGS
restaurants_to_keep = restaurant_review_counts[restaurant_review_counts >= MIN_RESTAURANT_RATINGS].index
# Update the processed reviews to only contain the selected businesses
processed_reviews = processed_reviews[processed_reviews["business_id"].isin(restaurants_to_keep)]

# Get number of reviews by user
user_review_counts = processed_reviews.groupby("user_id").size()
# Get the set of users with more reviews than MIN_USER_RATINGS
users_to_keep = user_review_counts[user_review_counts >= MIN_USER_RATINGS].index
# Update the processed reviews to only contain the selected users
processed_reviews = processed_reviews[processed_reviews["user_id"].isin(users_to_keep)]

# %%
# Take only the selected restaurants from the sampled ones
processed_restaurants = sampled_restaurants[sampled_restaurants.index.isin(restaurants_to_keep)].copy()
# Regroup the reviews by restaurant
restaurant_review_data = processed_reviews.groupby("business_id")
# Update the restaurant entries with their preprocessed review counts
processed_restaurants["review_count"] = restaurant_review_data.size()
# Update the restaurant entries with their preprocessed average review stars
processed_restaurants["average_stars"] = restaurant_review_data["stars"].mean()
# Merge the covid data into the restaurant data
processed_restaurants = pd.merge(processed_restaurants, sampled_covid, left_index=True, right_index=True)
# Cast Covid-19 delivery_takeaway feature from bool to int
processed_restaurants["delivery_takeaway"] = processed_restaurants["delivery_takeaway"].astype(int)
# Combine name and categories into text feature for TF-IDF
processed_restaurants["categories"] = processed_restaurants["categories"].str.replace("Restaurants", "")

# %%
# Take only the selected users from the sampled ones
processed_users = sampled_users[sampled_users.index.isin(users_to_keep)].copy()
# Regroup the reviews by user
user_review_data = processed_reviews.groupby("user_id")
# Update the user entries with their preprocessed review counts
processed_users["review_count"] = user_review_data.size()
# Update the user entries with their preprocessed average review stars
processed_users["average_stars"] = user_review_data["stars"].mean()

# %%
print("Updating normalised ratings...")
# Calculate normalised review ratings by subtracting the users' average stars from the review stars
processed_reviews["rating"] = processed_reviews.apply(
    lambda x: x["stars"] - processed_users["average_stars"].loc[x["user_id"]],
    axis=1
)
print("Done")

# %%
print("Writing review data...")
save_preprocessed_table("reviews", processed_reviews)
print("Done")
# %%
print("Writing restaurant data...")
save_preprocessed_table("restaurants", processed_restaurants)
print("Done")
# %%
print("Writing user data...")
save_preprocessed_table("users", processed_users)
print("Done")
