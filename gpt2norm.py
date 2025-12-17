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
    It then plots the L2-Norm (magnitude) of these vectors across model depth.
    """
    MODEL_NAME = "gpt2"
    # A short, simple sentence for clear visualization
    INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
    OUTPUT_FILENAME = 'gpt2_intermediate_embeddings.png'
    
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
    
    # --- 2. Activation Collection Hooks ---
    
    # Dictionary to store collected activations: {checkpoint_name: tensor}
    collected_activations = {} 
    
    def store_activation_hook(name):
        """Returns a hook function that stores the tensor output."""
        def hook(module, input, output):
            # output[0] is typically the tensor we want (or output if it's a simple tensor)
            # We store the output tensor for all tokens
            # output is (batch_size, seq_len, hidden_size)
            collected_activations[name] = output[0].detach().cpu().squeeze(0) 
        return hook

    # --- Initial Embeddings ---
    # Hook the output of the primary WTE (Word Token Embeddings) layer
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
    print(f"Running forward pass to collect {2 * num_layers + 1} checkpoints...")
    with torch.no_grad():
        # Pass the input through the model
        model(**inputs)

    # --- 4. Processing Activations (L2-Norm) ---

    # Generate an ordered list of checkpoint names (x-axis labels)
    checkpoint_names = ["Embeddings"]
    for i in range(num_layers):
        checkpoint_names.append(f"L{i}-Attention")
        checkpoint_names.append(f"L{i}-MLP")

    # Matrix to store the L2-Norms: [Token Index, Checkpoint Index]
    # Checkpoint Index = 1 (Embeddings) + 12*2 (Attn/MLP pairs) = 25
    norms_matrix = np.zeros((seq_len, len(checkpoint_names)))

    for j, name in enumerate(checkpoint_names):
        if name in collected_activations:
            # Activation shape is (seq_len, hidden_size)
            activation = collected_activations[name]
            # Calculate L2-Norm for each token (across the hidden_size dimension)
            # L2-Norm (Euclidean length) = sqrt(sum(x^2))
            token_norms = torch.linalg.norm(activation, dim=1).numpy()
            norms_matrix[:, j] = token_norms
        else:
            print(f"Warning: Activation for '{name}' not found.")

    # --- 5. Visualization ---
    print("\nGenerating line plot visualization...")

    plt.figure(figsize=(18, 10))
    
    # Plot a line for each token
    for i in range(seq_len):
        # Use a distinct marker and plot the line for the token
        plt.plot(
            np.arange(len(checkpoint_names)), 
            norms_matrix[i, :], 
            label=f"Token: '{tokens[i]}'", 
            marker='o',
            alpha=0.8,
            linewidth=2
        )
    
    plt.title(
        f"L2-Norm (Activity) of Token Embeddings Across GPT-2 Layers\nInput: '{INPUT_TEXT}'", 
        fontsize=16
    )
    plt.xlabel("Model Checkpoint (Block Output)")
    plt.ylabel("L2-Norm of Token Vector (Magnitude of Activity)")
    
    # Set X-ticks using the checkpoint names
    plt.xticks(
        np.arange(len(checkpoint_names)), 
        checkpoint_names, 
        rotation=90, 
        fontsize=10
    )
    
    # Add vertical lines to separate layers clearly
    for i in range(num_layers):
        # Line after the MLP output (before the next layer starts)
        plt.axvline(x=2 * i + 2.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_FILENAME, dpi=150) 
    print(f"\n[Visualization Saved] The embedding visualization has been saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_intermediate_embeddings()
