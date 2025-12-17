import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

print("LOADED")

# Set a seed for reproducibility
set_seed(42)

def analyze_induction_heads():
    """
    Loads GPT-2, runs a repeated sequence through it, and analyzes the attention
    patterns in the first few attention heads (Layer 0) for induction behavior.

    Induction Head Criterion: For a token T at position P, it attends primarily
    to the token immediately following the *previous* occurrence of T.
    e.g., in 'A B C D A X', the final 'A' (P=4) attends to 'B' (P=1).
    """
    MODEL_NAME = "gpt2"
    # A short, repeating pattern is best to clearly isolate the attention.
    # We are interested in the attention of the final 'X' (index 6)
    # targeting the token immediately following the first 'A' (index 2), which is 'B' (index 3).
    INPUT_TEXT = "A B C D E F A X"
    TARGET_TOKEN_INDEX = 7 # Index of 'X'
    ATTENDED_TOKEN_INDEX = 3 # Index of 'B'
    ANALYSIS_HEADS = [(5, 0), (5, 1), (5, 2), (1, 1)] # Layers 0 and 1, Heads 0 and 1

    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Using AutoModelForCausalLM to easily get attention outputs
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True)
    
    # --- 1. Input Preparation ---
    print(f"Input Sequence: '{INPUT_TEXT}'")
    # Tokenize the input and convert to PyTorch tensors
    inputs = tokenizer(INPUT_TEXT, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # Get token list for display
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    
    # --- 2. Forward Pass ---
    print("Running forward pass to extract attention weights...")
    with torch.no_grad():
        outputs = model(**inputs)

    # Attention weights are returned as a tuple of length 12 (number of layers)
    # Each element is a tensor of shape: (batch_size, num_heads, seq_len, seq_len)
    attention_outputs = outputs.attentions

    # --- 3. Analysis and Visualization ---
    
    # Prepare a 2x2 grid for visualization (or adapt based on ANALYSIS_HEADS size)
    num_plots = len(ANALYSIS_HEADS)
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 15), 
        squeeze=False # Ensure axes is always a 2D array
    )
    
    print("\n--- Attention Analysis ---\n")
    
    for i, (layer_idx, head_idx) in enumerate(ANALYSIS_HEADS):
        # Extract attention weights for the specific head
        # Shape: (1, seq_len, seq_len) -> attention_weights[layer_idx][0, head_idx, :, :]
        attn_matrix = attention_outputs[layer_idx][0, head_idx].cpu().numpy()
        
        # Get the row corresponding to the target token 'X'
        # This shows where 'X' (the current token) is looking.
        target_token_attention = attn_matrix[TARGET_TOKEN_INDEX, :]
        
        # Find the specific plot location
        row, col = divmod(i, 2)
        ax = axes[row, col]

        # Plotting the attention from the target token
        ax.bar(np.arange(len(tokens)), target_token_attention)
        
        # Highlight the expected induction target 'B' (index 3)
        ax.axvline(x=ATTENDED_TOKEN_INDEX, color='r', linestyle='--', linewidth=2, 
                   label=f"Induction Target ('{tokens[ATTENDED_TOKEN_INDEX].strip()}')")
        
        # Highlight the current token 'X' (index 7)
        ax.axvline(x=TARGET_TOKEN_INDEX, color='k', linestyle='-', linewidth=1, 
                   label=f"Query Token ('{tokens[TARGET_TOKEN_INDEX].strip()}')")

        # Customize plot
        ax.set_title(f"L{layer_idx} H{head_idx}: Attention from '{tokens[TARGET_TOKEN_INDEX].strip()}'")
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels([f"({j}){t.strip()}" for j, t in enumerate(tokens)], rotation=45, ha='right')
        ax.set_xlabel("Attended Token Index & Value")
        ax.set_ylabel("Attention Weight")
        ax.grid(axis='y', alpha=0.5)
        ax.legend()
        
        # Quantitative Check
        target_weight = target_token_attention[ATTENDED_TOKEN_INDEX]
        print(f"Layer {layer_idx}, Head {head_idx}: Attention weight to Induction Target ('{tokens[ATTENDED_TOKEN_INDEX].strip()}') is {target_weight:.4f}")

    fig.suptitle(f"Attention Weight Analysis for Input: '{INPUT_TEXT}'", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # --- 4. Conclusion and Interpretation ---
    print("\n--- Interpretation Guide ---")
    print(f"The induction head pattern is demonstrated if the attention weight to token '{tokens[ATTENDED_TOKEN_INDEX].strip()}' (index {ATTENDED_TOKEN_INDEX}, marked by the red dashed line) is significantly higher than other positions.")
    print("In GPT-2 small, true induction heads often appear later (e.g., L5H1 or L5H5). However, heads in lower layers (L0, L1) often show *previous-token* or *positional* attention patterns which are precursors to induction.")
    print("If one of the heads analyzed shows a high weight on the induction target, it is exhibiting induction head behavior.")

if __name__ == "__main__":
    analyze_induction_heads()
