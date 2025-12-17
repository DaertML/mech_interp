import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os

# Set a seed for reproducibility
set_seed(42)

def visualize_intermediate_embeddings():
    """
    Loads GPT-2, extracts the output embeddings (activations) after the initial 
    embedding layer, and after every Attention and MLP block for all layers.
    It then plots the raw vector content (pixels) as a large heatmap.
    """
    MODEL_NAME = "gpt2"
    # A short, simple sentence for clear visualization
    INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
    OUTPUT_FILENAME = 'gpt2_intermediate_embeddings_heatmap.png'
    
    print(f"Loading model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the base model without the LM head, as we only need the transformer blocks
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # --- 1. Input Preparation ---
    print(f"Input Sequence: '{INPUT_TEXT}'")
    inputs = tokenizer(INPUT_TEXT, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    
    # Clean up token labels for display
    tokens = [tokenizer.decode(t).strip().replace('\n', '\\n') for t in input_ids]
    seq_len = len(tokens)
    num_layers = model.config.num_hidden_layers # 12
    hidden_size = model.config.hidden_size # 768
    
    # --- 2. Activation Collection Hooks ---
    
    # Dictionary to store collected activations: {checkpoint_name: tensor}
    collected_activations = {} 
    
    # Remove all existing hooks to ensure a clean slate, though generally unnecessary
    # if the script runs clean, it's a good practice.
    
    def store_activation_hook(name):
        """Returns a hook function that stores the tensor output."""
        def hook(module, input, output):
            # output[0] is typically the tensor we want. Shape: (1, seq_len, hidden_size)
            collected_activations[name] = output[0].detach().cpu().squeeze(0) 
        return hook

    # --- Initial Embeddings ---
    model.transformer.wte.register_forward_hook(store_activation_hook("Embeddings"))
    
    # --- Transformer Block Outputs (Attention and MLP) ---
    for layer_idx in range(num_layers):
        block = model.transformer.h[layer_idx]
        
        # 1. Output after Attention Block (MHA + Residual/LayerNorm)
        # We hook the output of the whole GPT2Block's Attention sub-module
        block.attn.register_forward_hook(store_activation_hook(f"L{layer_idx}-Attention"))
        
        # 2. Output after MLP Block (MLP + Residual/LayerNorm)
        # We hook the output of the whole GPT2Block's MLP sub-module
        block.mlp.register_forward_hook(store_activation_hook(f"L{layer_idx}-MLP"))

    # --- 3. Forward Pass to Trigger Hooks ---
    num_checkpoints = 2 * num_layers + 1
    print(f"Running forward pass to collect {num_checkpoints} checkpoints...")
    with torch.no_grad():
        # Pass the input through the model
        model(**inputs)

    # --- 4. Processing Activations (Concatenate Raw Tensors) ---

    # Generate an ordered list of checkpoint names 
    checkpoint_names = ["Embeddings"]
    for i in range(num_layers):
        checkpoint_names.append(f"L{i}-Attention")
        checkpoint_names.append(f"L{i}-MLP")

    # List to store raw (1, 768) token vectors for ordered heatmap visualization
    plot_data_list = []
    y_tick_labels = []

    for name in checkpoint_names:
        if name in collected_activations:
            activation = collected_activations[name]  # (seq_len, hidden_size)
            
            for token_idx in range(seq_len):
                # Append the token vector (a single row in the final heatmap)
                plot_data_list.append(activation[token_idx:token_idx+1].numpy())
                y_tick_labels.append(f"{name} | '{tokens[token_idx]}'")
        else:
            print(f"Warning: Activation for '{name}' not found.")

    if not plot_data_list:
        print("Error: No activation data collected.")
        return

    # Concatenate all (1, 768) token vectors into one large matrix (Total_Rows, 768)
    full_activation_matrix = np.concatenate(plot_data_list, axis=0)
    total_rows = full_activation_matrix.shape[0]
    
    # --- 5. Visualization (Heatmap) ---
    print("\nGenerating heatmap visualization (Raw Embeddings as Pixels)...")

    # Use a very wide figure for the 768 dimensions and dynamic height for rows
    plt.figure(figsize=(30, total_rows * 0.15)) 
    
    # Use 'seismic' or 'bwr' for activations, where negative values are one color and positive the other
    # Clamping vmin/vmax provides better contrast for typically small activation values
    im = plt.imshow(full_activation_matrix, cmap='seismic', aspect='auto', interpolation='none', 
                    vmin=-5, vmax=5) 

    plt.title(
        f"Raw Token Vector Content Across GPT-2 Layers (Hidden Dim: {hidden_size})\nInput: '{INPUT_TEXT}'", 
        fontsize=20
    )
    plt.xlabel("Hidden Dimension Index (0 to 767)")
    plt.ylabel("Token/Component Checkpoint")
    
    # Set Y-ticks to show Token and Component
    plt.yticks(np.arange(total_rows), y_tick_labels, fontsize=6)
    
    # Add horizontal lines to clearly separate the tokens
    for i in range(1, total_rows):
        # A thin line to separate tokens within a checkpoint
        plt.axhline(y=i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    # Add thicker horizontal lines to separate the major Checkpoints (Layers)
    for i in range(1, len(checkpoint_names)):
        # Calculate the row index where the new checkpoint starts
        start_row_index = i * seq_len
        # Thick white line to separate major components/layers
        plt.axhline(y=start_row_index - 0.5, color='white', linewidth=2.0, alpha=1.0) 

    plt.colorbar(im, fraction=0.015, pad=0.02, label="Activation Value")
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_FILENAME, dpi=300) 
    print(f"\n[Visualization Saved] The embedding visualization has been saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_intermediate_embeddings()
