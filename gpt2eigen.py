import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os

# Set a seed for reproducibility
set_seed(42)

def analyze_wo_wv_eigenvalues():
    """
    Loads GPT-2 weights and plots the minimum eigenvalue of the W_O * W_V 
    matrix for every attention head across all layers.

    Negative eigenvalues suggest potential information suppression or deletion
    capacity in the head's mechanism.
    """
    MODEL_NAME = "gpt2"
    OUTPUT_FILENAME = 'gpt2_attention_wo_wv_eigenvalues.png'
    
    print(f"Loading model weights: {MODEL_NAME}...")
    # Load the base model to access the transformer weights
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    num_layers = model.config.num_hidden_layers # 12
    num_heads = model.config.num_attention_heads # 12
    d_model = model.config.hidden_size # 768
    d_head = d_model // num_heads # 64
    
    # Matrix to store the minimum eigenvalue: [Layer Index, Head Index]
    eigenvalue_matrix = np.zeros((num_layers, num_heads))
    
    print(f"Analyzing {num_layers} layers and {num_heads} heads ({num_layers * num_heads} total)...")

    for layer_idx in range(num_layers):
        block = model.transformer.h[layer_idx]
        
        # --- 1. Extract W^V (Value) and W^O (Output) Weights ---
        
        # W_QKC: (d_model, 3 * d_model). PyTorch stores weights as (out_features, in_features) for linear layers, 
        # but Conv1D/CausalLM implementations transpose this to (in_features, out_features).
        # We must treat the weight tensors as (in_features, out_features).
        W_QKC = block.attn.c_attn.weight.data # (d_model, 3 * d_model) -> (768, 2304)
        W_O_full = block.attn.c_proj.weight.data # (d_model, d_model) -> (768, 768)
        
        # W^V_full is the last third of W_QKC
        W_V_full = W_QKC[:, 2 * d_model : 3 * d_model] # (d_model, d_model) -> (768, 768)
        
        for head_idx in range(num_heads):
            # --- 2. Split into Head-specific Matrices ---
            
            # W^V_h: Maps residual stream (d_model=768) to head value space (d_head=64)
            # W^V_h is a block of W_V_full: (d_model, d_head) -> (768, 64)
            start_v = head_idx * d_head
            end_v = (head_idx + 1) * d_head
            W_V_h = W_V_full[:, start_v:end_v].cpu().numpy()
            
            # W^O_h: Maps head output (d_head=64) back to residual stream (d_model=768)
            # W^O_h is the slice of W_O_full (c_proj weight) corresponding to the head's output.
            # W_O_full is (d_model, d_model). We take the columns (dimension 1) of the matrix.
            start_o = head_idx * d_head
            end_o = (head_idx + 1) * d_head
            W_O_h = W_O_full[:, start_o:end_o].cpu().numpy() # Shape (768, 64)
            
            # --- 3. Calculate W^O W^V (M_h) ---
            
            # The matrix for eigenvalue analysis is M_h = (W^O_h)^T @ W^V_h 
            # where W^V_h is (768, 64) and W^O_h is (768, 64).
            # M_h shape: (64, 768) @ (768, 64) -> (64, 64). This resolves the dimension mismatch.
            M_h = W_O_h.T @ W_V_h
            
            # Calculate eigenvalues. We only care about the real part.
            eigenvalues = np.linalg.eigvals(M_h)
            real_eigenvalues = np.real(eigenvalues)
            
            # Store the minimum eigenvalue (most negative is the best indicator of deletion/suppression)
            min_eigenvalue = np.min(real_eigenvalues)
            eigenvalue_matrix[layer_idx, head_idx] = min_eigenvalue

    # --- 4. Visualization (Heatmap) ---
    print("\nGenerating W^O W^V Minimum Eigenvalue Heatmap...")

    plt.figure(figsize=(12, 12))
    
    # Use 'RdBu_r' to clearly show negative values (red) and positive values (blue)
    # Centering the colormap around zero is crucial for this analysis
    max_abs_val = np.max(np.abs(eigenvalue_matrix))
    
    im = plt.imshow(eigenvalue_matrix, cmap='RdBu_r', aspect='equal', 
                    vmin=-max_abs_val, vmax=max_abs_val)

    plt.title(
        f"GPT-2 Attention Head Minimum $W^O W^V$ Eigenvalues\n(Indicator of Suppressive Capacity)", 
        fontsize=16
    )
    plt.xlabel("Attention Head Index")
    plt.ylabel("Transformer Layer Index")
    
    # Set X and Y ticks
    plt.xticks(np.arange(num_heads), [f"H{i}" for i in range(num_heads)])
    plt.yticks(np.arange(num_layers), [f"L{i}" for i in range(num_layers)])
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Minimum Real Eigenvalue of $W^O W^V$")
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(OUTPUT_FILENAME, dpi=150) 
    print(f"\n[Visualization Saved] The $W^O W^V$ eigenvalue heatmap has been saved to: {os.path.abspath(OUTPUT_FILENAME)}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    analyze_wo_wv_eigenvalues()

