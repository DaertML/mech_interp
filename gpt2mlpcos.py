import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set a seed for reproducibility
set_seed(42)

def visualize_intermediate_embeddings():
    """
    Loads GPT-2 and analyzes the effect of the MLP layers by plotting the 
    cosine similarity between the vector entering the MLP and the vector 
    output by the MLP, for every token across all layers.
    
    Low similarity indicates a large angular change or "write" operation by the MLP.
    """
    MODEL_NAME = "gpt2"
    # A short, simple sentence for clear visualization
    INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
    OUTPUT_FILENAME = 'gpt2_mlp_cosine_similarity.png'
    
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
    
    # Dictionary to store collected activations: 
    # {layer_idx: {'input': tensor_MLP_input, 'output': tensor_MLP_output}}
    collected_activations = {i: {} for i in range(num_layers)} 
    
    def store_mlp_input_hook(layer_idx):
        """Returns a hook that stores the MLP input (output of ln_2)."""
        def hook(module, input, output):
            # output[0] is the tensor (1, seq_len, hidden_size)
            collected_activations[layer_idx]['input'] = output.detach().cpu().squeeze(0)
        return hook

    def store_mlp_output_hook(layer_idx):
        """Returns a hook that stores the MLP output (before residual addition)."""
        def hook(module, input, output):
            # output is the tensor (1, seq_len, hidden_size)
            collected_activations[layer_idx]['output'] = output.detach().cpu().squeeze(0)
        return hook

    # --- Transformer Block Hooks (MLP In/Out) ---
    for layer_idx in range(num_layers):
        block = model.transformer.h[layer_idx]
        
        # 1. Input to MLP (Output of LayerNorm 2)
        # We hook the output of the layer norm that feeds the MLP
        block.ln_2.register_forward_hook(store_mlp_input_hook(layer_idx))
        
        # 2. Output of MLP (Output of the MLP sub-module)
        # This vector is what is added back to the residual stream
        block.mlp.register_forward_hook(store_mlp_output_hook(layer_idx))

    # --- 3. Forward Pass to Trigger Hooks ---
    print(f"Running forward pass to collect activations...")
    with torch.no_grad():
        # Pass the input through the model
        model(**inputs)

    # --- 4. Processing Activations (Cosine Similarity) ---

    # Matrix to store similarity: [Token Index, Layer Index]
    similarity_matrix = np.zeros((seq_len, num_layers))

    for layer_idx in range(num_layers):
        layer_data = collected_activations[layer_idx]
        
        if 'input' in layer_data and 'output' in layer_data:
            mlp_input = layer_data['input'].numpy() # (seq_len, hidden_size)
            mlp_output = layer_data['output'].numpy() # (seq_len, hidden_size)
            
            # Calculate cosine similarity for each token
            for token_idx in range(seq_len):
                # Cosine similarity requires reshaping to (1, hidden_size) for sklearn
                input_vec = mlp_input[token_idx:token_idx+1, :]
                output_vec = mlp_output[token_idx:token_idx+1, :]
                
                # Cosine similarity of input vector with the output vector
                # The shape will be (1, 1), so we extract the single score
                similarity = cosine_similarity(input_vec, output_vec)[0, 0]
                similarity_matrix[token_idx, layer_idx] = similarity
        else:
            print(f"Warning: Missing input/output data for Layer {layer_idx}")

    # --- 5. Visualization (Line Plot) ---
    print("\nGenerating Cosine Similarity line plot visualization...")

    plt.figure(figsize=(16, 8))
    
    # Plot a line for each token
    for i in range(seq_len):
        plt.plot(
            np.arange(num_layers), 
            similarity_matrix[i, :], 
            label=f"Token: '{tokens[i]}'", 
            marker='o',
            alpha=0.8,
            linewidth=2
        )
    
    plt.title(
        f"Cosine Similarity (MLP Input vs. MLP Output) Across GPT-2 Layers\nInput: '{INPUT_TEXT}'", 
        fontsize=16
    )
    plt.xlabel("Transformer Layer Index")
    plt.ylabel("Cosine Similarity Score (MLP Input vs. MLP Output)")
    
    # Set X-ticks to Layer Index
    plt.xticks(
        np.arange(num_layers), 
        [f"L{i}" for i in range(num_layers)], 
        rotation=0, 
        fontsize=10
    )
    
    # Add a horizontal line at 0 (orthogonal change) and 1 (no change)
    plt.axhline(y=1.0, color='grey', linestyle=':', linewidth=1.0, alpha=0.5)
    plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', fontsize=10, ncol=2, title="Tokens")
    plt.ylim(-0.1, 1.05) # Ensure the range is visible
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_FILENAME, dpi=150) 
    print(f"\n[Visualization Saved] The cosine similarity plot has been saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_intermediate_embeddings()
