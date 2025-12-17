import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm import tqdm
import os

# Set a seed for reproducibility
set_seed(42)

def format_token(token_id, tokenizer):
    """Decodes a token ID and formats it, escaping newlines for clear printing."""
    token_str = tokenizer.decode([token_id])
    # Replace the actual newline character (\n) with the literal escape sequence (\\n)
    return token_str.replace('\n', '\\n')

def analyze_attention_features():
    """
    Computes and prints the top logit-boosting bigrams and induction candidates 
    for every attention head using the W^V * W^O * W_U circuit analysis.
    """
    MODEL_NAME = "gpt2"
    
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Model Configuration ---
    num_layers = model.config.num_hidden_layers # 12
    num_heads = model.config.num_attention_heads # 12
    d_model = model.config.hidden_size # 768
    d_head = d_model // num_heads # 64
    d_vocab = model.config.vocab_size # 50257
    
    # --- 1. Extract Unembedding (W_U) and Embedding (W_E) Matrices ---
    # In GPT-2, W_E and W_U are tied, derived from the language model head.
    W_U = model.lm_head.weight.data # Shape: (d_vocab, d_model) -> (50257, 768)
    W_E = model.transformer.wte.weight.data # Shape: (d_vocab, d_model) -> (50257, 768)
    
    # --- 2. Select Input Tokens for Analysis ---
    # We analyze the first 200 tokens (usually common words/punctuation)
    # The OV circuit operates on embeddings, so we select embeddings corresponding to these tokens.
    TOKENS_TO_ANALYZE = 200
    
    # Slice the token embeddings corresponding to the first N tokens
    E_in = W_E[:TOKENS_TO_ANALYZE, :].cpu().numpy() # Shape: (N, d_model) -> (200, 768)
    
    print(f"Analyzing top {TOKENS_TO_ANALYZE} vocabulary tokens...")
    print("-" * 50)
    
    # --- 3. Iterate and Analyze Heads ---
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        block = model.transformer.h[layer_idx]
        
        # --- Extract W^V and W^O Full Matrices ---
        W_QKC = block.attn.c_attn.weight.data # (d_model, 3 * d_model) -> (768, 2304)
        W_O_full = block.attn.c_proj.weight.data # (d_model, d_model) -> (768, 768)
        
        # True W_V_full (d_model, d_model)
        W_V_full = W_QKC[:, 2 * d_model : 3 * d_model] 
        
        # True W_O matrix (d_model, d_model) is the transpose of c_proj.weight
        W_O = W_O_full.T # (768, 768)
        
        for head_idx in range(num_heads):
            
            # --- Head-specific Slicing (W^V_h and W^O_h) ---
            start = head_idx * d_head
            end = (head_idx + 1) * d_head
            
            # W^V_h: Residual (768) -> Value Head (64). Shape (768, 64)
            W_V_h = W_V_full[:, start:end].cpu().numpy()
            
            # W^O_h: Value Head (64) -> Residual (768). Shape (64, 768)
            # This is the row slice of the True W_O matrix
            W_O_h = W_O[start:end, :].cpu().numpy()
            
            # --- 4. Calculate OVW_U Matrix ---
            
            # M_h = W^V_h @ W^O_h  -> (768, 768). The full OV map in residual space.
            M_h = W_V_h @ W_O_h
            
            # A_h = M_h @ W_U^T -> (768, 50257). The full OVW_U map.
            # W_U is (50257, 768), so we need W_U.T: (768, 50257)
            W_U_T = W_U.T.cpu().numpy() # (768, 50257)
            A_h = M_h @ W_U_T
            
            # --- 5. Compute Logit Boost and Extract Top Outputs ---
            
            # Logit_Boost_Matrix = E_in @ A_h 
            # (200, 768) @ (768, 50257) -> (200, 50257)
            logit_boost_matrix = E_in @ A_h
            
            # Find the indices of the top 5 output tokens for each input token
            TOP_K = 5
            top_indices = np.argsort(logit_boost_matrix, axis=1)[:, ::-1][:, :TOP_K]
            
            # --- 6. Print Results ---
            
            print(f"\n--- Layer {layer_idx:02d}, Head {head_idx:02d} ---")
            
            head_output_strings = []
            
            for i in range(TOKENS_TO_ANALYZE):
                input_token_id = i
                # Use the helper function to decode and escape the input token
                input_token_str = format_token(input_token_id, tokenizer)
                
                # Extract the top k output token IDs
                output_token_ids = top_indices[i]
                
                # Decode the output tokens and format them using the helper function
                output_tokens_str = ", ".join([
                    f"'{format_token(token_id, tokenizer)}'"
                    for token_id in output_token_ids
                ])

                # Check for induction candidates (input token strongly predicts itself)
                is_induction_candidate = input_token_id in output_token_ids
                
                output_line = f"  Input: '{input_token_str}': "
                output_line += f"  Top Outputs: [{output_tokens_str}]"
                if is_induction_candidate:
                    output_line += " (Self-Predicts: Potential Induction/Copying)"
                
                head_output_strings.append(output_line)
            
            # Only print patterns for heads that have strong, interesting signals
            # We can define "interesting" as having at least 5 tokens self-predicting.
            induction_count = sum(1 for i in range(TOKENS_TO_ANALYZE) if i in top_indices[i])
            if induction_count > 5:
                 print(f"  [SUMMARY]: Strong Induction/Copying Feature Found ({induction_count} tokens self-predicting).")
            
            # Print only a subset of the results for common tokens for brevity
            for i in [0, 1, 10, 50, 100, 150, 199]:
                 print(head_output_strings[i])


if __name__ == "__main__":
    analyze_attention_features()

