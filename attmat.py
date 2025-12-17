import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os

# Set a seed for reproducibility
set_seed(42)

def visualize_all_attention_matrices():
    """
    Loads GPT-2, runs an input sequence, extracts all 144 attention matrices
    (12 layers * 12 heads), and visualizes them in a 12x12 grid, saving the plot to a file.
    """
    MODEL_NAME = "gpt2"
    # A short, simple sentence for clear visualization
    INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
    INPUT_TEXT = "A, B, C, D, E, F, A, B, C"
    OUTPUT_FILENAME = 'gpt2_all_attention_matrices.png'
    
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Ensure attention outputs are requested
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True)
    
    # --- 1. Input Preparation ---
    print(f"Input Sequence: '{INPUT_TEXT}'")
    
    inputs = tokenizer(INPUT_TEXT, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    
    # Clean up token labels for display
    tokens = [tokenizer.decode(t).strip().replace('\n', '\\n') for t in input_ids]
    seq_len = len(tokens)
    
    # --- 2. Forward Pass ---
    print(f"Sequence length: {seq_len}. Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)

    attention_outputs = outputs.attentions
    
    num_layers = model.config.num_hidden_layers # 12
    num_heads = model.config.num_attention_heads # 12
    
    # --- 3. Visualization Setup (12x12 Grid) ---
    
    # Create a large figure to ensure individual plots are legible
    fig, axes = plt.subplots(
        num_layers, num_heads, 
        figsize=(36, 36), 
        squeeze=False
    )
    fig.suptitle(f"GPT-2 Attention Matrices (12 Layers x 12 Heads)\nInput: '{INPUT_TEXT}'", fontsize=32, y=1.005)
    
    print("Generating 144 attention matrix plots...")

    for layer_idx in range(num_layers):
        # Attention tensor for this layer: (1, 12, seq_len, seq_len)
        layer_attn = attention_outputs[layer_idx][0].cpu().numpy()
        
        for head_idx in range(num_heads):
            # Extract the attention matrix for this specific head (SeqLen x SeqLen)
            attn_matrix = layer_attn[head_idx]
            
            ax = axes[layer_idx, head_idx]
            
            # Plot the heatmap
            im = ax.imshow(attn_matrix, cmap='cividis', aspect='auto', interpolation='nearest')

            # Set axis labels (only for the leftmost column and bottom row for clarity)
            if head_idx == 0:
                ax.set_yticks(np.arange(seq_len))
                ax.set_yticklabels(tokens, fontsize=10)
                ax.set_ylabel(f"Layer {layer_idx}\n(Query Tokens)", rotation=90, labelpad=15, fontsize=12)
            else:
                ax.set_yticks([])

            if layer_idx == num_layers - 1:
                ax.set_xticks(np.arange(seq_len))
                ax.set_xticklabels(tokens, rotation=90, fontsize=10)
                ax.set_xlabel(f"Head {head_idx}\n(Key Tokens)", fontsize=12)
            else:
                ax.set_xticks([])
            
            # Set the title for the head
            if layer_idx == 0:
                 ax.set_title(f"H{head_idx}", fontsize=14, y=1.05)


    # Add a single color bar for the entire figure (representing attention weight)
    # The cbar will be placed outside the main 12x12 grid
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax).set_label('Attention Weight', rotation=270, labelpad=20, fontsize=16)

    plt.subplots_adjust(
        left=0.04, right=0.90,  # Adjust edges to fit labels/colorbar
        top=0.95, bottom=0.05,
        hspace=0.2, wspace=0.1
    )
    
    # --- 4. Save the figure ---
    print(f"\nSaving figure to {OUTPUT_FILENAME}...")
    # Use high DPI for better quality on a large plot
    plt.savefig(OUTPUT_FILENAME, dpi=200) 
    print(f"[Visualization Saved] The 144 attention matrices have been saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    
    # This show call is unlikely to work in non-interactive environments, 
    # but is left here for users in interactive environments.
    plt.show()

if __name__ == "__main__":
    visualize_all_attention_matrices()
