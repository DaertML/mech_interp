import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download 

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
            attn_only=True, # defaults to False
            tokenizer_name="EleutherAI/gpt-neox-20b",
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

        
        # --- 5. Full Circuit Composition Analysis ---
        layer_idx = 0
        head_idx = 0
        print(f"\n--- Analyzing Weights (Layer {layer_idx}, Head {head_idx}) ---")

        # a. Extract individual weights
        W_E = model.W_E # (d_vocab, d_model) - Input Embedding
        W_U = model.W_U # (d_model, d_vocab) - Unembedding
        W_Q = model.W_Q[layer_idx, head_idx] # (d_model, d_head)
        W_K = model.W_K[layer_idx, head_idx] # (d_model, d_head)
        W_V = model.W_V[layer_idx, head_idx] # (d_model, d_head)
        W_O = model.W_O[layer_idx, head_idx] # (d_head, d_model)
        
        # b. Calculate Composition Matrices (Subspace Movement)
        
        # M_Q: Vocab to Q-Space (d_vocab, d_head) - How tokens map to Q vectors
        M_Q = W_E @ W_Q
        
        # M_K: Vocab to K-Space (d_vocab, d_head) - How tokens map to K vectors
        M_K = W_E @ W_K
        
        # M_QK_Residual: QK Interaction Matrix (d_model, d_model)
        M_QK_Residual = W_Q @ W_K.T # (d_model, d_head) @ (d_head, d_model)
        
        # M_OV: OV-Circuit (d_model, d_model)
        M_OV = W_V @ W_O 
        
        # M_Logit_Contrib: Full Attention-Only Logit Contribution (d_model, d_vocab)
        M_Logit_Contrib = M_OV @ W_U

        # NEW: M_QK_Vocab: Full QK Affinity (Vocab x Vocab) - The flow of attention affinity from
        # a source token's embedding (row index) to a destination token's query (column index).
        # Shape: (d_vocab, d_vocab)
        # Formula: M_Q @ M_K.T, or (W_E @ W_Q) @ (W_E @ W_K).T
        M_QK_Vocab = M_Q @ M_K.T
        
        # c. Print Composition Shapes
        print(f"\n--- Composition Matrix Shapes (Layer {layer_idx}, Head {head_idx}) ---")
        
        print(f"1. M_Q (Vocab -> Q-Space) shape: {M_Q.shape}")
        print(f"2. M_K (Vocab -> K-Space) shape: {M_K.shape}")
        print(f"3. M_QK_Residual (Residual -> Residual Affinity) shape: {M_QK_Residual.shape}")
        print(f"4. M_OV (Residual -> Residual Output) shape: {M_OV.shape}")
        print(f"5. M_Logit_Contrib (Residual -> Logit) shape: {M_Logit_Contrib.shape}")
        print(f"6. M_QK_Vocab (Vocab -> Vocab Affinity) shape: {M_QK_Vocab.shape}")
        print("Circuit composition matrices calculated successfully.")
        
        
        # --- 6. Run forward pass and Token Flow Analysis ---
        text = "The dog sat on the mat and wagged its"
        
        # Run with cache to get attention patterns and logit scores
        logits, cache = model.run_with_cache(text, remove_batch_dim=True)
        
        str_tokens = model.to_str_tokens(text)
        
        # Analysis focuses on the prediction for the next token (index -1)
        dest_pos = len(str_tokens) - 1
        
        # Extract attention pattern for the final token (dest_pos)
        # Shape: (dest_pos + 1, dest_pos + 1). We want the row corresponding to dest_pos
        attention_pattern = cache["pattern", layer_idx, "attn"].squeeze(0)[dest_pos, :] 
        
        # Get the logits for the predicted next token
        next_token_logits = logits[-1, :]
        
        # Find the top 5 predicted tokens
        top_k_values, top_k_indices = torch.topk(next_token_logits, k=5)
        top_k_tokens = [model.tokenizer.decode(idx.item()) for idx in top_k_indices]

        # Find the source tokens that the final token (its) attended to most strongly
        attended_weights, attended_tokens_indices = torch.topk(attention_pattern, k=5)
        attended_tokens = [str_tokens[i] for i in attended_tokens_indices]

        print("\n--- Token Flow Illustration ---")
        print(f"Input Sentence: '{text}'")
        print(f"Tokens: {str_tokens}")
        print(f"Analyzing Destination Token: '{str_tokens[dest_pos]}'")

        print("\nFlow 1: Attention Affinity (QK-Circuit in action)")
        print(f"Which source tokens (Source Token $\mathbf{E} \to W_K$) does the destination token's Query ($\mathbf{Q}$) attend to most strongly?")
        for token, weight in zip(attended_tokens, attended_weights):
            print(f"- Source Token: '{token}' (Attention Weight: {weight.item():.4f})")

        print("\nFlow 2: Value $\to$ Logit (OV-Circuit $\to W_U$ in action)")
        print(f"What tokens does the attended information (Value vectors) predict after passing through $W_V \cdot W_O \cdot W_U$?")
        for token, logit_score in zip(top_k_tokens, top_k_values):
            print(f"- Predicted Token: '{token.strip()}' (Logit Score: {logit_score.item():.4f})")


    except ImportError as e:
        print(f"\n--- CRITICAL DEPENDENCY ERROR ---")
        print(f"Failed to import a necessary library: {e}")
        print("Please ensure you have 'transformer-lens', 'torch', and 'huggingface_hub' installed.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during model loading or running: {e}")
        print("Check if the model files are accessible and if your network connection is stable.")

if __name__ == "__main__":
    load_and_analyze_custom_model()
