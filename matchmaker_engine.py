import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_recommender():
    # Load the structured dataset
    try:
        df = pd.read_csv('raw_skills.csv')
    except FileNotFoundError:
        print("Error: 'raw_skills.csv' not found. Make sure it is in the same folder.")
        return

    print("\n" + "="*50)
    print(" DIGITAL MATCHMAKER: TECH STACK RECOMMENDER ")
    print("="*50)
    
    # ---------------------------------------------------------
    # PIPELINE STEP 1: INGESTION
    # ---------------------------------------------------------
    print("\nEnter at least 3 skills to map your career path.")
    print("Example: Python Cloud Automation")
    
    user_input = input("\nYour Skills: ").strip()
    
    # Prevent User Cold Start problem by enforcing minimum data density
    if len(user_input.split()) < 3:
        print("\n[!] Error: Insufficient data. You must provide a minimum of three skills.")
        return

    # ---------------------------------------------------------
    # PIPELINE STEP 2: SCORING (TF-IDF & Cosine Similarity)
    # ---------------------------------------------------------
    # Combine item dataset and user profile into a shared vocabulary space
    all_text_data = df['Skills'].tolist()
    all_text_data.append(user_input) # Append user state to the end

    # Transform raw text into numerical TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text_data)

    # Isolate item vectors and the single user vector
    item_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    # Calculate angular alignment (Cosine Similarity)
    similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()
    df['Similarity_Score'] = similarity_scores

    # ---------------------------------------------------------
    # PIPELINE STEP 3: SORTING
    # ---------------------------------------------------------
    # Organize dataset in descending order based on score
    sorted_df = df.sort_values(by='Similarity_Score', ascending=False)

    # ---------------------------------------------------------
    # PIPELINE STEP 4: FILTERING
    # ---------------------------------------------------------
    # Truncate output to prevent choice overload (Top-N List)
    top_n = 3
    final_matches = sorted_df.head(top_n)

    # Output generation
    print("\n" + "-"*30)
    print(" TOP 3 RECOMMENDED PATHS ")
    print("-"*30)
    
    rank = 1
    for index, row in final_matches.iterrows():
        # Convert float score to readable percentage
        match_percentage = round(row['Similarity_Score'] * 100, 1)
        
        print(f"{rank}. {row['Role']}")
        print(f"   Alignment Score: {match_percentage}%")
        print(f"   Required Skills: {row['Skills']}\n")
        rank += 1

if __name__ == "__main__":
    run_recommender()