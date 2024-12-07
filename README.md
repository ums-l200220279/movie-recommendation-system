# Movie Recommendation System using SVD (Singular Value Decomposition)

This repository contains the code for building a movie recommendation system using the **MovieLens 20M dataset**. The system uses **Singular Value Decomposition (SVD)** to recommend movies to users based on their ratings.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Recommendation System](#recommendation-system)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [License](#license)

## Project Overview
This project aims to create a movie recommendation system that predicts movie ratings based on user preferences. The system uses **SVD** from the **Surprise** library to build the model and provide movie recommendations to users.

## Dataset
The project uses the **MovieLens 20M** dataset, which contains movie ratings data, movie information, and user tags. The dataset consists of the following files:
- `movie.csv` - Contains movie information (ID, title, genre).
- `rating.csv` - Contains user ratings for movies.
- `tag.csv` - Contains user tags for movies.
- `genome_scores.csv` - Contains movie genome scores (describes movies with tags).
- `genome_tags.csv` - Contains tags used in genome scores.
- `link.csv` - Contains movie links to external sources.

Dataset can be found [here](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data).

## Installation

### Requirements
- Python 3.x
- Google Colab or Jupyter Notebook

### Install Required Libraries
1. Clone this repository or download the script.
2. Install necessary Python libraries:

```bash
pip install pandas matplotlib scikit-surprise

Data Collection
In this project, the dataset is loaded from Google Drive using the Google Colab environment. The paths to the dataset are specified in the script, and the CSV files are loaded into pandas DataFrames for further analysis.

python
Copy code
from google.colab import drive
drive.mount('/content/drive')

# Load dataset from Google Drive
dataset_path = '/content/drive/MyDrive/movielens_20m_data/'
movies_df = pd.read_csv(dataset_path + 'movie.csv')
ratings_df = pd.read_csv(dataset_path + 'rating.csv')
Data Preprocessing
The dataset is cleaned by performing the following tasks:

Checking and handling missing values (NaN).
Removing duplicate rows.
Ensuring the correct data types for columns.
python
Copy code
# Check for missing values and drop them
ratings_df = ratings_df.dropna()
movies_df = movies_df.dropna()

# Remove duplicates
ratings_df = ratings_df.drop_duplicates()
movies_df = movies_df.drop_duplicates()
Exploratory Data Analysis
We perform some initial analysis by visualizing:

The distribution of ratings.
The number of ratings per movie.
python
Copy code
import matplotlib.pyplot as plt

# Plot distribution of ratings
ratings_df['rating'].hist(bins=5)
Feature Engineering
The movie genres are one-hot encoded, which creates a separate column for each genre, allowing the model to use them as features.

python
Copy code
# One-hot encoding of genres
movies_df['genres'] = movies_df['genres'].str.split('|')
movies_df = pd.concat([movies_df, movies_df['genres'].apply(lambda x: pd.Series(1, index=x))], axis=1).fillna(0)
Modeling
We use the SVD (Singular Value Decomposition) algorithm from the Surprise library to train a recommendation model.

python
Copy code
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

# Prepare data for the model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train the model
svd = SVD()
svd.fit(trainset)
Recommendation System
The model generates movie recommendations for a given user based on their historical ratings.

python
Copy code
def get_movie_recommendations(user_id, n_recommendations=10):
    all_movie_ids = movies_df['movieId'].tolist()
    movie_predictions = [
        (movie_id, svd.predict(user_id, movie_id).est)
        for movie_id in all_movie_ids
    ]
    top_n = sorted(movie_predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
    recommended_movie_ids = [movie[0] for movie in top_n]
    recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
    return recommended_movies[['movieId', 'title']]

# Get recommendations for user with ID = 1
recommendations = get_movie_recommendations(user_id=1)
print("Top 10 Recommended Movies for User 1:\n", recommendations)
Evaluation
The model is evaluated using Root Mean Squared Error (RMSE) to check its performance.

python
Copy code
from surprise import accuracy

# Make predictions and calculate RMSE
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")
Visualization
The following plots are used to visualize the results:

RMSE plot to evaluate the performance of the model.
Distribution of ratings to understand user behavior.
python
Copy code
# Plot RMSE
plt.bar(['SVD'], [rmse])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE for SVD Model')
plt.show()

# Visualize rating distribution
plt.hist(ratings_df['rating'], bins=5, edgecolor='black')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Rating Distribution')
plt.show()
License
This project is licensed under the MIT License - see the LICENSE file for details.
