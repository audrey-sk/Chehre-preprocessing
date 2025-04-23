import json
import pandas as pd
import sys
import os 
from BERTcosinesim import word_cosine_sim
import matplotlib.pyplot as plt 

DEFAULT_JSON_FILE = 'emojiJSON.json'
JSON_FILE_PATH = DEFAULT_JSON_FILE

# Check if the default JSON file exists, otherwise ask the user
if not os.path.exists(JSON_FILE_PATH):
    print(f"Default file '{DEFAULT_JSON_FILE}' not found.")
    JSON_FILE_PATH = input("Please enter the path to your emoji JSON file: ")

# --- Functions ---
def extract_first_words(filepath):
    #Reads the JSON file and extracts the first word from each 'S' label.
    unique_words = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{filepath}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    if 'Data' not in data or not isinstance(data['Data'], list):
        print(f"Error: Expected 'Data' key with a list value in '{filepath}'")
        return None

    for emoji_entry in data['Data']:
        if 'SurveyLabels' in emoji_entry and isinstance(emoji_entry['SurveyLabels'], dict):
            for survey_id, survey_data in emoji_entry['SurveyLabels'].items():
                if isinstance(survey_data, dict) and 'S' in survey_data:
                    label_string = survey_data['S']
                    if isinstance(label_string, str) and label_string.strip():
                        # Split by whitespace and take the first word
                        words = label_string.split()
                        if words: # Check if the split resulted in any words
                            first_word = words[0]
                            if first_word: # Ensure first word isn't empty
                                unique_words.add(first_word)
    return unique_words

def calculate_similarity_matrix(words_list, sim_function):
    #Calculates the similarity matrix for a list of words  
    n = len(words_list)
    similarity_matrix = [[0.0] * n for _ in range(n)] # Initialize matrix with zeros


    for i in range(n):
        for j in range(n):
            # Optimization: Calculate only one half and mirror (since sim(a,b) == sim(b,a))
                if i <= j:
                    word1 = words_list[i]
                    word2 = words_list[j]
                    similarity = sim_function(word1, word2)
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity # Mirror the value

        # Create pandas DataFrame for better display
    df = pd.DataFrame(similarity_matrix, index=words_list, columns=words_list)
    return df
# Draw similarity matrix heatmap
def plot_similarity_matrix(df,
                           title: str = "Cosine-Similarity Matrix",
                           cmap: str = "coolwarm",
                           save_path: str | None = None):
    if df.empty:
        raise ValueError("The similarity DataFrame is empty.")

    #axes 
    n = len(df)
    fig, ax = plt.subplots(figsize=(0.45 * n + 4, 0.45 * n + 4))
    im = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=1)

    #  Axis labels 
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(df.index, fontsize=8)

    # Grid lines for readability
    ax.set_xticks([x - 0.5 for x in range(1, n)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)

    # Color bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine Similarity", rotation=-90, va="bottom")

    # Titles 
    ax.set_title(title, pad=15)
    ax.set_xlabel("Words")
    ax.set_ylabel("Word")
    plt.tight_layout()

    if save_path:                                 # save & close
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heat-map image saved to '{save_path}'.")
        plt.close(fig)
    else:                                         # or show it
        plt.show()

     

# Main 
if __name__ == "__main__":
    print(f"Reading data from: {JSON_FILE_PATH}")
    extracted_words = extract_first_words(JSON_FILE_PATH)

    if extracted_words is not None:
        if not extracted_words:
            print("No valid first words found in the specified 'S' labels.")
        else:
            # Sort words for consistent matrix order
            sorted_words = sorted(list(extracted_words))
            print(f"\nFound {len(sorted_words)} unique first words:")
            print(sorted_words)

            similarity_df = calculate_similarity_matrix(sorted_words, word_cosine_sim)

            if similarity_df is not None:
                print("\n--- Word Similarity Matrix ---")
                output_csv_file = 'word_similarity_matrix.csv'
                similarity_df.to_csv(output_csv_file, float_format='%.3f') # Save with 3 decimal places
                print(f"\n--- Word Similarity Matrix saved to '{output_csv_file}' ---")
                plot_similarity_matrix( similarity_df, save_path="word_similarity_heatmap.png" ) 
                
    else:
        print("Could not process emoji data due to previous errors.")