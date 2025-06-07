import pandas as pd
import ast # Used to safely evaluate string-formatted lists/dictionaries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process # For the fuzzy string matching


# --- Part 1: Loading and Preparing the Data (Corrected) ---

def load_and_prepare_data():
    """
    This function handles the dirty work: loading the files, merging them,
    and cleaning up the important columns to be usable.
    """
    print("Welcome! Loading the movie data...")

    # Load the datasets
    credits_df = pd.read_csv('tmdb_5000_credits.csv')
    movies_df = pd.read_csv('tmdb_5000_movies.csv')

    # The 'movies' file uses 'id', and the 'credits' file uses 'movie_id'. Let's rename for a clean merge.
    credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
    
    # Merge them on the 'id' column. We now have one big dataframe!
    df = movies_df.merge(credits_df, on='id')

    df.rename(columns={'title_x': 'title'}, inplace=True)

    df = df[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    # The 'genres', 'keywords', 'cast', and 'crew' columns are strings that look like lists of dictionaries.
    # We need to turn them into actual lists so we can work with them.
    # The 'ast.literal_eval' function is perfect for this.
    
    # This helper function will pull out the 'name' from each dictionary in the list.
    def extract_names(text):
        # Handle cases where the data might be missing (NaN)
        if isinstance(text, str):
            return [item['name'] for item in ast.literal_eval(text)]
        return []

    # This one is special for the cast, we only want the first 3 actors.
    def extract_top_cast(text):
        if isinstance(text, str):
            return [item['name'] for item in ast.literal_eval(text)][:3]
        return []

    # This one finds the director from the crew list. The most important person!
    def extract_director(text):
        if isinstance(text, str):
            for item in ast.literal_eval(text):
                if item['job'] == 'Director':
                    return [item['name']]
        return []

    print("Cleaning and processing features...")
    for feature in ['genres', 'keywords']:
        df[feature] = df[feature].apply(extract_names)
        
    df['cast'] = df[feature].apply(extract_top_cast)
    df['crew'] = df[feature].apply(extract_director)
    
    # We need to remove spaces between names ('Sam Worthington' becomes 'SamWorthington').
    # This prevents the model from getting confused between 'Sam Worthington' and 'Sam Mendes'.
    def remove_spaces(word_list):
        return [word.replace(" ", "") for word in word_list]

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        df[feature] = df[feature].apply(remove_spaces)

    # Now let's combine all our features into one big "tags" string for each movie.
    # The overview is already a string, so we'll just split it into words.
    df['overview'] = df['overview'].fillna('').apply(lambda x: x.split())
    
    df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
    
    # Finally, we'll convert the list of tags back into a single string.
    df['tags'] = df['tags'].apply(lambda x: " ".join(x))
    
    # Create our final, clean dataframe.
    final_df = df[['id', 'title', 'tags']]
    
    print("Data processing complete!\n")
    return final_df

   
# --- Part 2: Vectorization and Similarity Calculation ---

def calculate_similarity(df):
    """
    Takes the cleaned dataframe, converts the 'tags' into a matrix of word counts,
    and then calculates the cosine similarity between all movies.
    """
    print("Building the recommendation engine (this might take a moment)...")
    
    # Use CountVectorizer to convert our text into a matrix of token counts.
    # We'll remove common English "stop words" like 'the', 'a', 'in'.
    # max_features=5000 means we only care about the 5000 most frequent words.
    cv = CountVectorizer(max_features=5000, stop_words='english')
    
    # This creates a giant matrix where rows are movies and columns are words.
    vectors = cv.fit_transform(df['tags']).toarray()
    
    # Calculate the cosine similarity. This gives us a score of how similar
    # each movie is to every other movie.
    similarity_matrix = cosine_similarity(vectors)
    
    print("Engine is ready!\n")
    return similarity_matrix

# --- Part 3: The Recommender Function ---

def get_recommendations(movie_title, df, similarity_matrix):
    """
    The main recommendation logic. Finds the movie in our dataset
    and returns the top 5 most similar movies.
    """
    try:
        # Find the index of the movie the user chose.
        movie_index = df[df['title'] == movie_title].index[0]
        
        # Get the similarity scores for that movie against all other movies.
        distances = similarity_matrix[movie_index]
        
        # Sort the movies by similarity and get the top 5 (the first one is the movie itself, so we skip it).
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print(f"\nSince you liked '{movie_title}', you might also enjoy:")
        for i in movies_list:
            # i[0] is the index of the recommended movie in the dataframe.
            print(f"- {df.iloc[i[0]].title}")
            
    except IndexError:
        print("Oops! I couldn't find that movie in my database.")


# --- Part 4: The Interactive 'Human-Like' Main Loop ---

def main():
    # Step 1: Prepare all the data and models
    movies_data = load_and_prepare_data()
    similarity = calculate_similarity(movies_data)
    
    # Create a list of all movie titles for our fuzzy matching tool
    movie_titles = movies_data['title'].tolist()

    print("--- Your Personal Movie Recommender ---")
    print("You can ask for recommendations based on a movie you like.")
    print("For example, try 'The Dark Knight' or 'Avatar'.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter a movie title: ").strip()

        if user_input.lower() == 'quit':
            print("Thanks for using the recommender. Enjoy your movie night!")
            break
            
        if not user_input:
            continue
            
        # Step 2: Use the fuzzy matching tool to find the best match
        # process.extractOne returns the best match and its similarity score (0-100)
        best_match, score = process.extractOne(user_input, movie_titles)
        
        # We can set a threshold. If the score is too low, we're not confident it's the right movie.
        if score >= 80:
            # Confirm with the user
            confirmation = input(f"Did you mean '{best_match}'? (yes/no): ").strip().lower()
            if confirmation in ['yes', 'y']:
                # If they confirm, get the recommendations!
                get_recommendations(best_match, movies_data, similarity)
            else:
                print("My apologies. Please try typing the movie title again.")
        else:
            print("Sorry, I couldn't find a close match for that movie. Please try another one.")
            
        print("\n" + "-"*40 + "\n") # Separator for the next round

if __name__ == '__main__':
    main()