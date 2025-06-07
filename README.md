# Content-Based Movie Recommender System

## Project Overview

This repository contains a Python-based movie recommendation system that implements a content-based filtering approach. The system processes a dataset of movie attributes to compute content similarity and generate relevant recommendations. User input is handled robustly via fuzzy string matching to correct for potential typographical errors.

## Methodology

The recommendation engine follows a structured pipeline from data ingestion to recommendation generation. The core logic is based on natural language processing techniques to quantify the similarity between movie entities.

**1. Data Ingestion and Preprocessing**
*   The system loads two primary CSV files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) into Pandas DataFrames.
*   These DataFrames are merged on their common identifier (`id` / `movie_id`) to create a unified dataset.
*   A subset of relevant features is selected for analysis: `overview`, `genres`, `keywords`, `cast`, and `crew`.

**2. Feature Engineering**
*   Text-based, JSON-formatted columns (`genres`, `keywords`, `cast`, `crew`) are parsed to extract relevant information. Specifically, this includes movie genres, plot keywords, the top 3 actors, and the director's name.
*   To ensure data integrity and create unique identifiers, all spaces are removed from multi-word names and keywords (e.g., "James Cameron" is transformed into "JamesCameron"). This prevents the model from associating unrelated entities that share a common word.
*   All extracted features are concatenated into a single metadata string, or "corpus," for each movie.

**3. Vectorization**
*   The textual corpus is converted into a numerical format suitable for machine learning using Scikit-learn's `CountVectorizer`.
*   This process constructs a document-term matrix where each row corresponds to a movie and each column corresponds to a token (word) in the corpus vocabulary. The values represent token frequencies. Common English stop words are excluded.

**4. Similarity Computation**
*   The **cosine similarity** is computed for every pair of movie vectors in the document-term matrix.
*   This metric measures the cosine of the angle between two vectors, effectively quantifying their similarity irrespective of magnitude. The result is a square, symmetric similarity matrix where the element `(i, j)` contains the similarity score between movie `i` and movie `j`.

**5. Recommendation Generation**
*   When a user provides a movie title, `fuzzywuzzy` is employed to find the closest match within the dataset, accommodating for typos.
*   Upon successful identification, the system retrieves the corresponding row from the similarity matrix.
*   This row is then sorted to identify the indices of the most similar movies, and the top 5 results are presented to the user.

## Technology Stack

*   **Language:** Python 3
*   **Data Analysis and Manipulation:** Pandas
*   **Machine Learning & NLP:** Scikit-learn
*   **Fuzzy String Matching:** FuzzyWuzzy, python-levenshtein

## Installation and Execution

Follow these steps to set up and run the project on a local machine.

**1. Clone the Repository**
```bash
git clone https://github.com/3aboshe/Movie-Recommendor.git
cd Movie-Recommendor
```

**2. Install Dependencies**
Install the required Python libraries using pip:
```bash
pip install pandas scikit-learn fuzzywuzzy python-levenshtein
```

**3. Download the Dataset**
This project requires the **[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)** from Kaggle.
-   Navigate to the provided link to download the dataset. A Kaggle account may be required.
-   Unzip the `archive.zip` file.
-   Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project's root directory.

**4. Run the Application**
Execute the main script from the terminal:
```bash
python MovieRecommender.py
```
The application will process the data and prompt the user for input to begin generating recommendations.
