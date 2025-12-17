import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow

def create_attention_mechanism_plot():
    """
    Creates a visualization of the attention mechanism based on the provided diagram
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Set up the main plot area
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
    
    # Title
    fig.suptitle('Layer 0 Attention Head Mechanism\n"Each token looks one position backwards"', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # === LEFT PLOT: QK Circuit ===
    ax1.set_title('QK Circuit: Attention Affinity', fontsize=14, fontweight='bold', pad=20)
    
    # Token positions
    positions = [(2, 6), (4, 6), (6, 6), (8, 6)]  # Previous tokens
    current_pos = (5, 2)  # Current token "urs" at position n+1
    
    # Draw previous tokens (keys)
    for i, (x, y) in enumerate(positions):
        label = "D" if i % 2 == 0 else "urs"
        pos_label = f"pos={i}"
        
        # Key box
        key_box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, boxstyle="round,pad=0.1",
                               facecolor='lightcoral', edgecolor='darkred', linewidth=2)
        ax1.add_patch(key_box)
        ax1.text(x, y, f'Key: "{label}"\n{pos_label}', ha='center', va='center', 
                fontweight='bold', fontsize=10)
        
        # Key vector description
        ax1.text(x, y-1, 'W_K: "I am at pos=0"', ha='center', va='center', 
                fontsize=9, style='italic', color='darkred')
    
    # Draw current token (query)
    query_box = FancyBboxPatch((current_pos[0]-0.8, current_pos[1]-0.3), 1.6, 0.6, 
                             boxstyle="round,pad=0.1", facecolor='lightblue', 
                             edgecolor='darkblue', linewidth=2)
    ax1.add_patch(query_box)
    ax1.text(current_pos[0], current_pos[1], 'Query: "urs"\npos=n+1', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Query vector description
    ax1.text(current_pos[0], current_pos[1]-1, 'W_Q: "I\'m looking for pos=n"', 
            ha='center', va='center', fontsize=9, style='italic', color='darkblue')
    
    # Draw attention arrows from query to keys
    for i, (x, y) in enumerate(positions):
        # Only highlight the correct match (position n)
        if i == 1:  # This would be position n in a real sequence
            arrow = Arrow(current_pos[0], current_pos[1]+0.3, x-current_pos[0], y-2.7,
                         width=0.1, color='green', alpha=0.7)
            ax1.add_patch(arrow)
            
            # Attention score text
            ax1.text((current_pos[0]+x)/2, (current_pos[1]+y)/2, 'High Attention\nScore', 
                    ha='center', va='center', fontsize=9, color='green', 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7))
        else:
            arrow = Arrow(current_pos[0], current_pos[1]+0.3, x-current_pos[0], y-2.7,
                         width=0.1, color='gray', alpha=0.3)
            ax1.add_patch(arrow)
    
    # Fixed mathematical formulation - removed \text commands
    math_text = r'$Attention\ Score = Q_{n+1} \cdot K_n = "urs" \cdot "D"$'
    ax1.text(5, 0.5, math_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    
    # === RIGHT PLOT: OV Circuit ===
    ax2.set_title('OV Circuit: Information Movement', fontsize=14, fontweight='bold', pad=20)
    
    # Source token (D at position n)
    source_pos = (2, 6)
    source_box = FancyBboxPatch((source_pos[0]-0.8, source_pos[1]-0.3), 1.6, 0.6,
                              boxstyle="round,pad=0.1", facecolor='lightcoral',
                              edgecolor='darkred', linewidth=2)
    ax2.add_patch(source_box)
    ax2.text(source_pos[0], source_pos[1], 'Value: "D"\npos=n', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Destination token (urs at position n+1)
    dest_pos = (8, 2)
    dest_box = FancyBboxPatch((dest_pos[0]-0.8, dest_pos[1]-0.3), 1.6, 0.6,
                            boxstyle="round,pad=0.1", facecolor='lightblue',
                            edgecolor='darkblue', linewidth=2)
    ax2.add_patch(dest_box)
    ax2.text(dest_pos[0], dest_pos[1], 'Output: "urs"\npos=n+1', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # OV transformation process
    # Value transformation
    ax2.text(source_pos[0], source_pos[1]-1, 'W_V: "I am D"', 
            ha='center', va='center', fontsize=9, style='italic', color='darkred')
    
    # Output transformation
    ax2.text(dest_pos[0], dest_pos[1]-1, 'W_O: "I follow D"', 
            ha='center', va='center', fontsize=9, style='italic', color='darkblue')
    
    # Information flow arrow
    arrow = Arrow(source_pos[0]+0.5, source_pos[1]-0.3, 5, -3.4,
                 width=0.1, color='purple', alpha=0.8)
    ax2.add_patch(arrow)
    
    # Transformation label
    ax2.text(5, 4.5, 'OV Circuit\nW_V · W_O', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lavender'))
    
    # Information content
    info_text = 'Information: "I follow D"\ngets added to residual stream'
    ax2.text(5, 3, info_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan'))
    
    # Fixed mathematical formulation - removed \text commands
    math_text = r'$Output = Attention \times (V \cdot W_O) = "I follow D"$'
    ax2.text(5, 0.5, math_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))
    
    # Summary text
    summary_text = (
        "Summary:\n"
        "• Each token looks one position backwards\n"
        "• QK circuit finds which previous token to attend to\n"
        "• OV circuit moves information to current token\n"
        "• Information format: 'I follow X'"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='whitesmoke'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('attention_mechanism_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_analyze_custom_model():
    """
    Loads a custom attention-only model and performs Q/K/V circuit composition analysis,
    focusing on the progressive flow of information through the head and token-level results.
    """
    try:
        # --- 1. Define Model and Environment ---
        REPO_ID = "callummcdougall/attn_only_2L_half"
        FILENAME = "attn_only_2L_half.pth"
        
        # Set device dynamically
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 2. Download weights (requires 'huggingface_hub')
        print(f"Downloading weights from {REPO_ID}/{FILENAME}...")
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        print(f"Weights downloaded to: {weights_path}")


        # --- 3. Define HookedTransformer Configuration ---
        cfg = HookedTransformerConfig(
            d_model=768,
            d_head=64,
            n_heads=12,
            n_layers=2,
            n_ctx=2048,
            d_vocab=50278,
            attention_dir="causal",
            attn_only=True,
            tokenizer_name="EleutherAI/gpt-neox-20b",
            seed=398,
            use_attn_result=True,
            normalization_type=None,
            positional_embedding_type="shortformer"
        )
        print("\nConfiguration defined successfully.")


        # --- 4. Initialize model and load weights ---
        model = HookedTransformer(cfg)
        pretrained_weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(pretrained_weights)
        print("Model initialized and weights loaded successfully.")

        
        # --- 5. Full Circuit Composition Analysis ---
        layer_idx = 0
        head_idx = 0
        print(f"\n--- Analyzing Weights (Layer {layer_idx}, Head {head_idx}) ---")

        # Extract weights
        W_E = model.W_E
        W_U = model.W_U
        W_Q = model.W_Q[layer_idx, head_idx]
        W_K = model.W_K[layer_idx, head_idx]
        W_V = model.W_V[layer_idx, head_idx]
        W_O = model.W_O[layer_idx, head_idx]
        
        # Calculate composition matrices
        M_Q = W_E @ W_Q
        M_K = W_E @ W_K
        M_QK_Residual = W_Q @ W_K.T
        M_OV = W_V @ W_O
        M_Logit_Contrib = M_OV @ W_U
        M_QK_Vocab = M_Q @ M_K.T
        
        print(f"\n--- Composition Matrix Shapes (Layer {layer_idx}, Head {head_idx}) ---")
        print(f"1. M_Q (Vocab -> Q-Space) shape: {M_Q.shape}")
        print(f"2. M_K (Vocab -> K-Space) shape: {M_K.shape}")
        print(f"3. M_QK_Residual (Residual -> Residual Affinity) shape: {M_QK_Residual.shape}")
        print(f"4. M_OV (Residual -> Residual Output) shape: {M_OV.shape}")
        print(f"5. M_Logit_Contrib (Residual -> Logit) shape: {M_Logit_Contrib.shape}")
        print(f"6. M_QK_Vocab (Vocab -> Vocab Affinity) shape: {M_QK_Vocab.shape}")
        
        
        # --- 6. Create Visualization ---
        print("\n--- Creating Attention Mechanism Visualization ---")
        create_attention_mechanism_plot()
        print("Visualization saved as 'attention_mechanism_diagram.png'")

    except ImportError as e:
        print(f"\n--- CRITICAL DEPENDENCY ERROR ---")
        print(f"Failed to import a necessary library: {e}")
        print("Please ensure you have 'transformer-lens', 'torch', and 'huggingface_hub' installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during model loading or running: {e.__class__.__name__}: {e}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    load_and_analyze_custom_model()
