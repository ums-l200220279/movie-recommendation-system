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

Dataset can be found [here](https://grouplens.org/datasets/movielens/).

## Installation

### Requirements
- Python 3.x
- Google Colab or Jupyter Notebook

### Install Required Libraries
1. Clone this repository or download the script.
2. Install necessary Python libraries:

```bash
pip install pandas matplotlib scikit-surprise
