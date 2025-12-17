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
    it provides to a set of expected completion tokens (T_out).
    """
    MODEL_NAME = "gpt2"
    
    # --- USER INPUT SECTION ---
    
    # 1. Define the natural language prompts. The *last token* of each prompt 
    # will be used as the T_in (Value Token) for the OVW_U analysis.
    TARGET_PROMPTS = [
        'def ... "("'
        #'open ... ","', # Analyzing the token '"' for file modes

        #'The quick brown fox jumped over the', # Analyzing ' the' for nouns/verbs
        #'Q: John said that Mary', # Analyzing ' Mary' for pronouns/verbs
    ] 
    
    # 2. Define the expected continuation tokens for each prompt.
    # We will specifically calculate the logit boost for these tokens.
    EXPECTED_CONTINUATIONS = {
        'def ... "("': ["self","not","hello"]
        #'open ... ","': ['r', 'w', 'a', 'b', ' rb', ' wb', ' r', ' w'],
        #'The quick brown fox jumped over the': [' lazy', ' moon', ' fence', ' river'],
        #'Q: John said that Mary': [' loves', ' said', ' is', ' did', ' herself'],
    }
    
    # --- END USER INPUT SECTION ---
    
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    analysis_data = []
    
    # 1. Pre-process prompts and target outputs
    for prompt in TARGET_PROMPTS:
        token_ids = tokenizer.encode(prompt)
        
        if not token_ids:
            print(f"Warning: Prompt '{prompt}' resulted in no tokens. Skipping.")
            continue
            
        target_token_id = token_ids[-1]
        target_token_str = format_token(target_token_id, tokenizer)
        
        # Get and tokenize the expected output tokens
        expected_tokens = EXPECTED_CONTINUATIONS.get(prompt, [])
        expected_ids = tokenizer.convert_tokens_to_ids(expected_tokens)
        
        # Filter out unknown tokens (might happen with single letters like 'r', 'w')
        valid_expected_ids = [idx for idx in expected_ids if idx != tokenizer.unk_token_id]
        
        if not valid_expected_ids:
            print(f"Warning: No valid expected continuation tokens found for '{prompt}'. Skipping.")
            continue
            
        analysis_data.append({
            'prompt': prompt,
            'input_id': target_token_id,
            'input_str': target_token_str,
            'output_ids': valid_expected_ids,
            'output_strs': [format_token(i, tokenizer) for i in valid_expected_ids]
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
            
            # --- 5. Compute and Report Logit Boost for Targeted Tokens ---
            
            print(f"\n--- Layer {layer_idx:02d}, Head {head_idx:02d} ---")
            
            for data in analysis_data:
                E_in = E_in_map[data['input_id']] # (1, 768)
                
                # Logit_Boost_Vector = E_in @ A_h 
                # (1, 768) @ (768, 50257) -> (1, 50257)
                logit_boost_vector = E_in @ A_h
                
                # Extract specific logit boosts for the expected output tokens
                output_boosts = logit_boost_vector[data['output_ids']]
                
                # Pair the output token string with its boost score
                results = []
                for j in range(len(data['output_strs'])):
                    results.append(f"'{data['output_strs'][j]}' ({output_boosts[j]:.3f})")

                print(f"  Prompt: '{data['prompt']}'")
                print(f"  Value Token (T_in): '{data['input_str']}'")
                print(f"    -> Targeted Boosts: [{', '.join(results)}]")

if __name__ == "__main__":
    analyze_attention_features()

