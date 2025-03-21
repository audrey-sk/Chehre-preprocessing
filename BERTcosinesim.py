from transformers import AutoTokenizer, AutoModel # pre-trained tokenizer for BERT and pre-trained BERT model 
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') #a version of BERT that preserves uppercase/lowercase 
model = AutoModel.from_pretrained('bert-base-cased', output_hidden_states=True).eval() #get all hidden layers

def word_cosine_sim(word1, word2):
    # Tokenize words separately
    tok1 = tokenizer(word1, return_tensors='pt', add_special_tokens=True)
    tok2 = tokenizer(word2, return_tensors='pt', add_special_tokens=True)

    # Gets word-to-token mappings (needed because BERT might split words into smaller subwords).
    # Example: "aardvark" → ['a', '##ard', '##var', '##k'] (all these tokens map to the same word index).
    # word_ids() helps find the correct embeddings later

    word_ids1 = tok1.word_ids()
    word_ids2 = tok2.word_ids()

    # Find the first actual token index 
    tok1_idx = next(i for i, w in enumerate(word_ids1) if w is not None)
    tok2_idx = next(i for i, w in enumerate(word_ids2) if w is not None)

    # Compute BERT embeddings
    with torch.no_grad():
        out1 = model(**tok1)
        out2 = model(**tok2)

    # Extract last hidden state
    states1 = out1.hidden_states[-1].squeeze(0)   
    states2 = out2.hidden_states[-1].squeeze(0)

    # Get embeddings of the main token
    emb1 = states1[tok1_idx]
    emb2 = states2[tok2_idx]

    # Compute cosine similarity
    similarity = torch.cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))

    return similarity.item()  # Return scalar similarity score

# Example usage
sim_score = word_cosine_sim("apple", "orange")
print("Cosine Similarity:", sim_score)
