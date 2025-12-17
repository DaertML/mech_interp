import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Set a seed for reproducibility
set_seed(42)

def global_induction_head_sweep():
    """
    Performs a sweep across all attention layers and heads of GPT-2 small
    to calculate the induction head score (attention weight to the token
    following the previous occurrence of the current token) and visualizes it.
    """
    MODEL_NAME = "gpt2"
    
    # Input sequence: 'A B C D E F A X'
    # The final 'X' (Query) should attend to the token following the first 'A'.
    # A (2) -> B (3)
    INPUT_TEXT = "A B C D E F A X"
    
    # Indices in the tokenized sequence (after tokenizing " A", " B", etc.)
    # We assume 'X' is the last token and 'B' is the target key token.
    # We must determine these indices accurately after tokenization.
    
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Using AutoModelForCausalLM to easily get attention outputs and model dimensions
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True)
    
    # --- 1. Input Preparation and Indexing ---
    
    # Tokenize the input sequence
    inputs = tokenizer(INPUT_TEXT, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode(t).strip() for t in input_ids]
    seq_len = len(tokens)
    
    # Dynamically find the indices of the critical tokens
    # Last token is always the query
    QUERY_TOKEN_IDX = seq_len - 1
    
    # Find the previous 'A' and the token immediately following it ('B')
    try:
        # Search for the *second to last* occurrence of 'A' to find the start of the pattern
        first_A_idx = tokens[:-2].index('A') 
        INDUCTION_TARGET_KEY_IDX = first_A_idx + 1 # Token 'B'
    except ValueError:
        print("Error: The pattern 'A X' was not found or is too short.")
        return
        
    QUERY_TOKEN = tokens[QUERY_TOKEN_IDX]
    TARGET_TOKEN = tokens[INDUCTION_TARGET_KEY_IDX]

    print(f"Input Tokens: {tokens}")
    print(f"Query Token ('{QUERY_TOKEN}') at index: {QUERY_TOKEN_IDX}")
    print(f"Induction Target ('{TARGET_TOKEN}') at index: {INDUCTION_TARGET_KEY_IDX}")

    # --- 2. Forward Pass ---
    print("\nRunning forward pass to extract attention weights...")
    with torch.no_grad():
        outputs = model(**inputs)

    attention_outputs = outputs.attentions
    
    # GPT-2 Small dimensions
    num_layers = model.config.num_hidden_layers # 12
    num_heads = model.config.num_attention_heads # 12
    
    # Initialize a matrix to store the attention scores: [Layer, Head]
    induction_scores = np.zeros((num_layers, num_heads))

    # --- 3. Score Extraction (The "Swipe") ---
    print("Performing sweep across all 144 heads...")
    
    for layer_idx in range(num_layers):
        # Attention tensor for this layer: (1, 12, seq_len, seq_len)
        layer_attn = attention_outputs[layer_idx][0]
        
        for head_idx in range(num_heads):
            # Extract the attention weight from QUERY_TOKEN_IDX (query) to INDUCTION_TARGET_KEY_IDX (key)
            # This is the single score that defines the induction behavior for this head
            score = layer_attn[head_idx, QUERY_TOKEN_IDX, INDUCTION_TARGET_KEY_IDX].item()
            induction_scores[layer_idx, head_idx] = score

    # --- 4. Visualization (Heatmap) ---
    print("\nGenerating heatmap...")

    plt.figure(figsize=(10, 10))
    # Use 'viridis' for high contrast
    im = plt.imshow(induction_scores, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add color bar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight (Induction Score)', rotation=270, labelpad=15)

    # Set axis ticks and labels
    plt.xticks(np.arange(num_heads), labels=[f"H{i}" for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), labels=[f"L{i}" for i in range(num_layers)])
    
    plt.xlabel("Attention Head Index")
    plt.ylabel("Layer Index")
    
    plt.title(f"Induction Score Heatmap for GPT-2 Small\nQuery='{QUERY_TOKEN}' to Target='{TARGET_TOKEN}'")
    
    # Add text annotations to the heatmap for specific values
    for i in range(num_layers):
        for j in range(num_heads):
            # Annotate with the score, but only if it's above a certain threshold for readability
            score = induction_scores[i, j]
            if score > 0.1:
                 # Adjust text color based on background (score intensity)
                 color = "white" if score > induction_scores.max() / 2 else "black"
                 plt.text(j, i, f"{score:.2f}", ha="center", va="center", color=color, fontsize=8)


    plt.tight_layout()
    plt.show()

    # --- 5. Report Findings ---
    print("\n--- Summary of Findings ---")
    
    # Find the top 3 strongest induction heads
    flat_scores = induction_scores.flatten()
    # Get the indices of the top 3 scores
    top_indices = np.argsort(flat_scores)[-3:][::-1]
    
    print(f"Query: '{QUERY_TOKEN}' (Index {QUERY_TOKEN_IDX}) -> Key: '{TARGET_TOKEN}' (Index {INDUCTION_TARGET_KEY_IDX})")
    print("Top 3 Potential Induction Heads:")
    
    for rank, flat_idx in enumerate(top_indices):
        layer_idx, head_idx = np.unravel_index(flat_idx, induction_scores.shape)
        score = induction_scores[layer_idx, head_idx]
        print(f"  {rank+1}. Layer {layer_idx}, Head {head_idx} (Score: {score:.4f})")
    
    if induction_scores.max() > 0.4:
        print("\nConclusion: Strong evidence of an induction head detected (Score > 0.4).")
    else:
        print("\nConclusion: Induction behavior is subtle or distributed across heads for this prompt.")

if __name__ == "__main__":
    global_induction_head_sweep()
