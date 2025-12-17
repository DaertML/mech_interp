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
    Analyzes the W^V * W^O * W_U circuit for targeted skip-trigram patterns.
    It takes the last token of a prompt (T_in) and reports the logit boost 
    it provides to the TOP_K most likely completion tokens (T_out).
    """
    MODEL_NAME = "gpt2"
    
    # --- USER INPUT SECTION ---
    
    # 1. Define the natural language prompts. The *last token* of each prompt 
    # will be used as the T_in (Value Token) for the OVW_U analysis.
    TARGET_PROMPTS = [
        'Ralph ... R',
        #'def ... ('
        #'open ... ","', # Analyzing the token '"' for file modes
        #'The quick brown fox jumped over the', # Analyzing ' the' for nouns/verbs
        #'Q: John said that Mary', # Analyzing ' Mary' for pronouns/verbs
    ] 
    
    # 2. Define the number of top predicted tokens to display.
    TOP_K = 5
    
    # --- END USER INPUT SECTION ---
    
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    # Suppress the warning about weight initialisation when loading
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    analysis_data = []
    
    # 1. Pre-process prompts
    for prompt in TARGET_PROMPTS:
        token_ids = tokenizer.encode(prompt)
        
        if not token_ids:
            print(f"Warning: Prompt '{prompt}' resulted in no tokens. Skipping.")
            continue
            
        target_token_id = token_ids[-1]
        target_token_str = format_token(target_token_id, tokenizer)
        
        analysis_data.append({
            'prompt': prompt,
            'input_id': target_token_id,
            'input_str': target_token_str,
        })
    
    if not analysis_data:
        print("Error: No valid analysis data prepared.")
        return

    # --- Model Configuration ---
    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // num_heads
    d_vocab = model.config.vocab_size
    
    # --- 2. Extract W_U and W_E Matrices ---
    # NOTE: No inference is run; we only use the static weights.
    W_U = model.lm_head.weight.data # (d_vocab, d_model)
    W_E = model.transformer.wte.weight.data # (d_vocab, d_model)
    
    # Extract the embeddings for ALL unique input tokens in one go
    unique_input_ids = list(set(data['input_id'] for data in analysis_data))
    E_in_map = {
        idx: W_E[idx, :].cpu().numpy()
        for idx in unique_input_ids
    }
    
    print(f"Analysis focusing on the OVW_U contribution of the last token of each prompt.")
    print(f"Reporting the top {TOP_K} boosted tokens for each head.")
    print("-" * 50)
    
    # --- 3. Iterate and Analyze Heads ---
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        block = model.transformer.h[layer_idx]
        
        # --- Extract W^V and W^O Full Matrices ---
        W_QKC = block.attn.c_attn.weight.data 
        W_O_full = block.attn.c_proj.weight.data 
        W_V_full = W_QKC[:, 2 * d_model : 3 * d_model] 
        W_O = W_O_full.T 
        
        for head_idx in range(num_heads):
            
            # --- Head-specific Slicing (W^V_h and W^O_h) ---
            start = head_idx * d_head
            end = (head_idx + 1) * d_head
            
            W_V_h = W_V_full[:, start:end].cpu().numpy() # (768, 64)
            W_O_h = W_O[start:end, :].cpu().numpy() # (64, 768)
            
            # --- 4. Calculate OVW_U Matrix ---
            M_h = W_V_h @ W_O_h # (768, 768)
            W_U_T = W_U.T.cpu().numpy() # (768, 50257)
            A_h = M_h @ W_U_T # (768, 50257)
            
            # --- 5. Compute and Report Top K Logit Boosts ---
            
            print(f"\n--- Layer {layer_idx:02d}, Head {head_idx:02d} ---")
            
            for data in analysis_data:
                E_in = E_in_map[data['input_id']] # (1, 768)
                
                # Logit_Boost_Vector = E_in @ A_h 
                # (1, 768) @ (768, 50257) -> (1, 50257)
                logit_boost_vector = E_in @ A_h
                
                # Find the indices of the top K largest boosts
                # argsort returns indices that would sort the array.
                # [::-1] reverses it to get descending order.
                # [:TOP_K] selects the top K indices.
                top_k_indices = np.argsort(logit_boost_vector)[::-1][:TOP_K]
                
                # Extract the corresponding boost scores
                top_k_boosts = logit_boost_vector[top_k_indices]
                
                # Decode the token IDs
                top_k_strs = [format_token(i, tokenizer) for i in top_k_indices]
                
                # Pair the output token string with its boost score
                results = []
                for j in range(TOP_K):
                    results.append(f"'{top_k_strs[j]}' ({top_k_boosts[j]:.3f})")

                print(f"  Prompt: '{data['prompt']}'")
                print(f"  Value Token (T_in): '{data['input_str']}'")
                print(f"    -> Top {TOP_K} Boosts: [{', '.join(results)}]")

if __name__ == "__main__":
    analyze_attention_features()
