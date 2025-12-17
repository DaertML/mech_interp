import torch as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer, utils

# --- Configuration ---
MODEL_NAME = "gpt2"
INPUT_TEXT = "The quick brown fox jumps over the lazy dog."
LAYERS_TO_VISUALIZE = 6 # Use a subset for clearer visualization
t.set_grad_enabled(False)

# Helper function for calculating cosine similarity (alignment)
def calculate_alignment(weights, subspace_basis):
    """
    Calculates the average cosine similarity between the weight vectors
    (which define the read/write directions for a head) and the subspace basis vectors.
    A higher value means the weight projects more strongly onto the subspace.

    weights: Tensor of shape [n_heads, d_model, d_head] or [n_heads, d_head, d_model]
    subspace_basis: Tensor of shape [d_model, d_basis] (e.g., W_E or W_POS)
    
    Returns: Tensor of shape [n_layers, n_heads]
    """
    
    # 1. Normalize the basis vectors
    subspace_norm = t.linalg.norm(subspace_basis, dim=0, keepdim=True)
    subspace_unit = subspace_basis / (subspace_norm + 1e-6)

    # Reshape weights if it's W_O (d_head, d_model) to (d_model, d_head) for consistency
    if weights.shape[-2] != weights.shape[-1] and weights.shape[-1] < weights.shape[-2]:
        # Assume W_O shape [n_heads, d_head, d_model], transpose d_head and d_model axes
        weights = weights.transpose(-2, -1)
    
    # Flatten the weights along the d_model axis for projection calculation
    # Weights shape is typically [n_heads, d_model, d_head]
    
    # 2. Calculate the projection magnitude onto the subspace for each head/dimension pair
    # Projection = (W @ subspace_unit)
    # The calculation is complex because we want the norm of the projection onto the *subspace*,
    # not just single dimensions. We simplify by calculating the average cosine similarity
    # between each row of W_Q/W_K/W_V/W_O and the columns of the subspace basis.
    
    # Reshape to [n_heads, d_model, d_head] -> [n_heads * d_head, d_model]
    n_heads = weights.shape[0]
    d_model = weights.shape[1]
    d_head = weights.shape[2]
    
    weights_flat = weights.reshape(n_heads * d_head, d_model) # [n_head * d_head, d_model]

    # Calculate average cosine similarity: (A dot B) / (|A| * |B|)
    # Since subspace_unit is normalized, we only need to normalize weights_flat
    weights_norm = t.linalg.norm(weights_flat, dim=1, keepdim=True)
    weights_unit = weights_flat / (weights_norm + 1e-6)
    
    # Cosine similarity matrix: [n_heads*d_head, d_basis]
    cosine_sim = weights_unit @ subspace_unit
    
    # Average the absolute similarity across all basis vectors for each head
    avg_alignment = t.mean(t.abs(cosine_sim), dim=1) # [n_heads * d_head]

    # Reshape back to [n_heads, d_head] and take the mean across d_head
    final_alignment = avg_alignment.reshape(n_heads, d_head).mean(dim=1) # [n_heads]
    
    return final_alignment.cpu().numpy()

def visualize_subspace_alignment(model: HookedTransformer):
    """
    Analyzes the alignment of W_Q, W_K, and W_O with W_E, W_POS, and W_U.
    """
    # 1. Extract Subspace Bases from model weights
    # W_E defines the token content features coming INTO the residual stream
    W_E = model.W_E.clone().float() # [d_vocab, d_model] -> use W_E.T as basis [d_model, d_vocab]
    
    # W_POS defines the positional features coming INTO the residual stream
    W_POS = model.W_pos.clone().float() # [max_ctx, d_model] -> use W_POS.T as basis [d_model, max_ctx]
    
    # W_U defines the token features projected OUT of the residual stream for decoding
    # Note: W_U is model.W_U [d_model, d_vocab]
    W_U = model.W_U.clone().float()

    # We will use the transpose of W_E and the transpose of W_POS as our subspace bases
    # Basis for Token Input/Content: W_E.T -> [d_model, d_vocab]
    token_input_basis = W_E.T 
    # Basis for Positional Input: W_POS.T -> [d_model, max_ctx]
    positional_input_basis = W_POS.T 
    # Basis for Token Output/Decoding: W_U -> [d_model, d_vocab]
    token_output_basis = W_U

    # List of matrices to analyze and their corresponding titles
    matrices_to_analyze = [
        ('W_Q', "Query (Q) Projection"),
        ('W_K', "Key (K) Projection"),
        ('W_V', "Value (V) Projection"),
        ('W_O', "Output (O) Projection")
    ]
    
    # Subspaces to project onto and their titles
    subspaces = [
        (token_input_basis, "Token Content (W_E)", "Input/Key-Side Subspace"),
        (positional_input_basis, "Positional (W_POS)", "Input/Key-Side Subspace"),
        (token_output_basis, "Token Decoding (W_U)", "Output/Logit-Side Subspace")
    ]

    # 2. Calculate Alignments
    results = {}
    n_layers = min(model.cfg.n_layers, LAYERS_TO_VISUALIZE)
    n_heads = model.cfg.n_heads

    for i in range(n_layers):
        for matrix_name, _ in matrices_to_analyze:
            # Get the weight tensor for the current layer
            weights_tensor = getattr(model.blocks[i].attn, matrix_name).float().squeeze() # [n_heads, d_model, d_head] or [n_heads, d_head, d_model]
            
            for subspace_basis, subspace_name, _ in subspaces:
                key = (matrix_name, subspace_name)
                
                # Reshape for a single layer's weights
                # weights_tensor has shape [n_heads, d_model, d_head] or [n_heads, d_head, d_model]
                
                # Check if W_O and transpose it to match [n_heads, d_model, d_head] pattern
                if matrix_name == 'W_O':
                    # W_O is [n_heads, d_head, d_model], need [n_heads, d_model, d_head]
                    weights_for_calc = weights_tensor.transpose(1, 2)
                else:
                    weights_for_calc = weights_tensor
                
                alignment = calculate_alignment(weights_for_calc, subspace_basis)
                
                if key not in results:
                    results[key] = np.zeros((n_layers, n_heads))
                
                results[key][i, :] = alignment

    # 3. Visualization
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
    plt.suptitle(
        f"Subspace Alignment of Attention Head Projections (Model: {MODEL_NAME}, Layers 0-{n_layers-1})",
        fontsize=16,
        y=1.02
    )

    # Define a common color map for comparison
    cmap = sns.cm.rocket_r 
    vmax = 0.05 # Empirically set max for gpt2-small alignment visualization

    # QK-Circuit Analysis (Inputs: W_E, W_POS)
    # ----------------------------------------
    
    # 1. W_Q Alignment with Token Content (W_E)
    ax = axes[0, 0]
    data = results[('W_Q', 'Token Content (W_E)')]
    sns.heatmap(data, cmap=cmap, annot=False, fmt=".3f", ax=ax, vmax=vmax,
                cbar_kws={'label': 'Avg Cosine Similarity'})
    ax.set_title("1. Query (W_Q) Alignment with Token Content (W_E)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Head")

    # 2. W_K Alignment with Positional Encoding (W_POS)
    ax = axes[0, 1]
    data = results[('W_K', 'Positional (W_POS)')]
    sns.heatmap(data, cmap=cmap, annot=False, fmt=".3f", ax=ax, vmax=vmax,
                cbar_kws={'label': 'Avg Cosine Similarity'})
    ax.set_title("2. Key (W_K) Alignment with Positional (W_POS)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Head")
    
    # OV-Circuit Analysis (Output: W_U)
    # ---------------------------------
    
    # 3. W_O Alignment with Token Decoding (W_U)
    ax = axes[1, 0]
    data = results[('W_O', 'Token Decoding (W_U)')]
    sns.heatmap(data, cmap=cmap, annot=False, fmt=".3f", ax=ax, vmax=vmax,
                cbar_kws={'label': 'Avg Cosine Similarity'})
    ax.set_title("3. Output (W_O) Alignment with Token Decoding (W_U)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Head")

    # 4. W_V Alignment with Token Content (W_E) - What information is being read?
    ax = axes[1, 1]
    data = results[('W_V', 'Token Content (W_E)')]
    sns.heatmap(data, cmap=cmap, annot=False, fmt=".3f", ax=ax, vmax=vmax,
                cbar_kws={'label': 'Avg Cosine Similarity'})
    ax.set_title("4. Value (W_V) Alignment with Token Content (W_E)")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Head")
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    print("Visualization complete. Please see the generated plot.")
    print("Interpretation Notes:")
    print(" - High alignment (darker red/yellow) means the head's weight is strongly in the direction of that subspace.")
    print(" - QK alignment (1 and 2) tells you what information (token/position) the attention mechanism is using to select source tokens.")
    print(" - OV alignment (3 and 4) tells you what kind of information (token/decoding features) the head is writing back to the residual stream.")


def main():
    print(f"Loading model: {MODEL_NAME}...")
    # Load a pre-trained model
    try:
        model = HookedTransformer.from_pretrained(
            MODEL_NAME, 
            device='cpu',
            # Setting init_weights to False speeds up the load time significantly
            # We only need the weights, not the full computation/cache for this analysis
            init_weights=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure transformer_lens is installed (e.g., pip install transformer-lens).")
        return

    print(f"Model loaded with {model.cfg.n_layers} layers and {model.cfg.n_heads} heads.")
    print(f"Analyzing alignment for the first {LAYERS_TO_VISUALIZE} layers...")
    
    visualize_subspace_alignment(model)


if __name__ == "__main__":
    # Ensure necessary libraries are available
    try:
        import torch
        import numpy
        import matplotlib.pyplot
        import seaborn
        from transformer_lens import HookedTransformer
    except ImportError:
        print("Error: Required libraries (torch, numpy, matplotlib, seaborn, transformer_lens) are not installed.")
        print("Please run: pip install transformer-lens matplotlib seaborn")
    else:
        main()

