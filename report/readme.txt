To use the recommender system command-line interface, open a winodws terminal (for unicode character support) in /src/ and run:
python rs_cli.py
This will present the input/output interface which will call down to the hybrid recommender system.
When prompted for a user ID, any ID (first column value) from /data/preprocessed/users.csv can be used.

The sampled data is in /src/data/sampled and consists of:
- sampled_businesses.csv: Set of sampled businesses
- sampled_covid.csv: Covid-19 data for the sampled businesses
- sampled_reviews.csv: Set of reviews for the sampled businesses
- sampled_users.csv: Set of users for the sampled reviews

These were sampled using the code in rs_datasampling.py, in order to use this it requires the raw data to be extracted to /data/raw (retaining default filenames).
Note: for the Covid-19 data, businesses with non-false "Virtual Services Offered" have one record per virtual service (with a repeated business_id); as these are not used, only the first record is sampled.

The preprocessing occurs in rs_dataprep.py and is output to /src/data/preprocessed consisting of:
- restaurants.csv
- reviews.csv
- users.csv

Note that there is no file for Covid-19 data here as its feature data is merged into the restaurant data during preprocessing.

The hybrid recommender system is in rs_hybrid.py, collaborative filtering is in rs_collaborative.py and content-based filtering is in rs_content.py.
