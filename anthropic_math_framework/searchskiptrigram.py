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
    Analyzes the W^V * W^O * W_U circuit by treating the last token of a user-defined
    prompt as the Value Token (T_in) to determine its logit-boosting influence.
    """
    MODEL_NAME = "gpt2"
    
    # --- USER INPUT SECTION ---
    # Define the natural language prompts. The *last token* of each prompt 
    # will be used as the T_in (Value Token) for the OVW_U analysis.
    # Examples:
    # 1. 'open ... ","' - The Value Token is ' "'. We are looking for the token
    #    that should follow the quote, e.g., ' r', ' w', or a letter.
    # 2. 'The capital of France is ' - The Value Token is ' is'.
    # 3. 'a new file named ' - The Value Token is ' '. We expect a quote or a word.
    TARGET_PROMPTS = [
        'open(', # Analyzing the token '"'
        'The capital of France is', # Analyzing the token ' is'
        'The quick brown fox', # Analyzing the token ' fox'
    ] 
    
    # --- END USER INPUT SECTION ---
    
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    analysis_inputs = []
    
    # Pre-process prompts to get the target token ID and string
    for prompt in TARGET_PROMPTS:
        token_ids = tokenizer.encode(prompt)
        
        if not token_ids:
            print(f"Warning: Prompt '{prompt}' resulted in no tokens. Skipping.")
            continue
            
        target_token_id = token_ids[-1]
        target_token_str = format_token(target_token_id, tokenizer)
        
        analysis_inputs.append({
            'prompt': prompt,
            'id': target_token_id,
            'str': target_token_str
        })
    
    if not analysis_inputs:
        print("Error: No valid analysis inputs found.")
        return

    # --- Model Configuration ---
    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // num_heads
    d_vocab = model.config.vocab_size
    
    # --- 1. Extract W_U and W_E Matrices ---
    W_U = model.lm_head.weight.data # (d_vocab, d_model)
    W_E = model.transformer.wte.weight.data # (d_vocab, d_model)
    
    # Extract the embeddings for ALL target tokens in one go
    target_ids = [item['id'] for item in analysis_inputs]
    E_in = W_E[target_ids, :].cpu().numpy() # Shape: (Num_Targets, d_model)
    NUM_TARGETS = len(analysis_inputs)
    
    print(f"Analyzing {NUM_TARGETS} Value Tokens extracted from prompts.")
    print("-" * 50)
    
    # --- 2. Iterate and Analyze Heads ---
    
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
            
            # --- 3. Calculate OVW_U Matrix ---
            M_h = W_V_h @ W_O_h # (768, 768)
            W_U_T = W_U.T.cpu().numpy() # (768, 50257)
            A_h = M_h @ W_U_T # (768, 50257)
            
            # --- 4. Compute Logit Boost ---
            
            # Logit_Boost_Matrix = E_in @ A_h 
            # (Num_Targets, 768) @ (768, 50257) -> (Num_Targets, 50257)
            logit_boost_matrix = E_in @ A_h
            
            # Find the indices of the top 5 output tokens for each input token
            TOP_K = 5
            top_indices = np.argsort(logit_boost_matrix, axis=1)[:, ::-1][:, :TOP_K]
            
            # --- 5. Print Results ---
            
            print(f"\n--- Layer {layer_idx:02d}, Head {head_idx:02d} ---")
            
            for i in range(NUM_TARGETS):
                input_data = analysis_inputs[i]
                
                # Extract the top k output token IDs
                output_token_ids = top_indices[i]
                
                # Format output tokens with their logit boost score
                output_tokens_str = ", ".join([
                    f"'{format_token(token_id, tokenizer)}' ({logit_boost_matrix[i, token_id]:.2f})"
                    for token_id in output_token_ids
                ])
                
                print(f"  Prompt: '{input_data['prompt']}'")
                print(f"  Value Token: '{input_data['str']}'")
                print(f"    -> Top 5 Logit Boosts: [{output_tokens_str}]")


if __name__ == "__main__":
    analyze_attention_features()

