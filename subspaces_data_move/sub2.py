import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download 
from transformers import AutoTokenizer
def load_and_analyze_custom_model():
    """
    Loads a custom attention-only model and performs Q/K/V circuit composition analysis.
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
            attn_only=True, # defaults to False
            tokenizer_name="gpt2",
            #tokenizer_name=AutoTokenizer.from_pretrained('./gptneo', use_fast=False),
            tokenizer_prepends_bos=True,
            seed=398,
            use_attn_result=True,
            normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
            positional_embedding_type="shortformer"
        )
        print("\nConfiguration defined successfully.")


        # --- 4. Initialize model and load weights ---
        # Initialize the model with the config
        model = HookedTransformer(cfg)
        
        # Load the pretrained weights
        pretrained_weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(pretrained_weights)
        print("Model initialized and weights loaded successfully.")

        
        # --- 5. Circuit Composition Analysis ---
        layer_idx = 0
        head_idx = 0
        print(f"\n--- Analyzing Weights (Layer {layer_idx}, Head {head_idx}) ---")

        # a. Extract individual weights for the specific head
        W_E = model.W_E # (d_vocab, d_model)
        W_Q = model.W_Q[layer_idx, head_idx] # (d_model, d_head)
        W_K = model.W_K[layer_idx, head_idx] # (d_model, d_head)
        W_V = model.W_V[layer_idx, head_idx] # (d_model, d_head)
        W_O = model.W_O[layer_idx, head_idx] # (d_head, d_model)
        
        print(f"W_E shape: {W_E.shape}")
        print(f"W_Q shape: {W_Q.shape}")
        
        # b. Calculate Composition Matrices
        
        # Q-Composition (Maps Vocab to Query Space): W_E @ W_Q
        # (d_vocab, d_model) @ (d_model, d_head) -> (d_vocab, d_head)
        M_Q = W_E @ W_Q
        
        # K-Composition (Maps Vocab to Key Space): W_E @ W_K
        # (d_vocab, d_model) @ (d_model, d_head) -> (d_vocab, d_head)
        M_K = W_E @ W_K
        
        # OV-Circuit (Maps Residual Stream to Residual Stream via Head): W_V @ W_O
        # (d_model, d_head) @ (d_head, d_model) -> (d_model, d_model)
        M_OV = W_V @ W_O 

        # c. Print Composition Shapes
        print(f"\n--- Composition Matrix Shapes (Layer {layer_idx}, Head {head_idx}) ---")
        print(f"M_Q (W_E @ W_Q) shape: {M_Q.shape}")
        print(f"M_K (W_E @ W_K) shape: {M_K.shape}")
        print(f"M_OV (W_V @ W_O) shape: {M_OV.shape}")
        print("Circuit composition matrices calculated successfully.")
        
        
        # --- 6. Run forward pass (optional but useful for testing) ---
        text = "The composition matrices show us how the input embedding space is transformed into the query and key vectors."
        
        # Run with cache to get attention patterns
        logits, cache = model.run_with_cache(text, remove_batch_dim=True)
        
        str_tokens = model.to_str_tokens(text)

        print("\n--- Model Run Verification ---")
        print("Test text run successfully.")
        print(f"Output logits shape: {logits.shape}")
        print(f"Token count: {len(str_tokens)}")


    except ImportError as e:
        print(f"\n--- CRITICAL DEPENDENCY ERROR ---")
        print(f"Failed to import a necessary library: {e}")
        print("Please ensure you have 'transformer-lens', 'torch', and 'huggingface_hub' installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during model loading or running: {e}")
        print("Check if the model files are accessible and if your network connection is stable.")

if __name__ == "__main__":
    load_and_analyze_custom_model()

