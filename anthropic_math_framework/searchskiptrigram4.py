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

def analyze_trigram_head_contribution():
    """
    Analyzes which specific attention head (L, H) provides the largest 
    W^V * W^O * W_U logit boost for a set of expected completion tokens, 
    based on the input token T_in (the last token of the prompt).
    """
    MODEL_NAME = "gpt2"
    
    # --- USER INPUT SECTION ---
    
    # 1. Define the target prompt. The *last token* will be T_in.
    TARGET_PROMPT = 'def __init__ ( self'
    
    # 2. Define the expected continuation tokens (T_out).
    EXPECTED_CONTINUATIONS = ['):', '):\\n', '):\\n\\n', '):']
    
    # --- END USER INPUT SECTION ---
    
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Data Pre-processing ---
    token_ids = tokenizer.encode(TARGET_PROMPT)
    if not token_ids:
        print("Error: Prompt resulted in no tokens.")
        return
        
    prompt_tensor = torch.tensor([token_ids])
    T_in_id = token_ids[-1]
    T_in_str = format_token(T_in_id, tokenizer)
    
    # Get and tokenize the expected output tokens
    expected_ids = tokenizer.convert_tokens_to_ids(EXPECTED_CONTINUATIONS)
    # Filter out unknown tokens
    EXPECTED_IDS = [idx for idx in expected_ids if idx != tokenizer.unk_token_id]
    EXPECTED_STRS = [format_token(i, tokenizer) for i in EXPECTED_IDS]
    
    if not EXPECTED_IDS:
        print("Error: No valid expected continuation tokens found.")
        return
        
    # --- Model Configuration ---
    num_layers = model.config.num_hidden_layers 
    num_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // num_heads
    d_vocab = model.config.vocab_size
    
    # --- 1. Run Full Inference (Optional, for context) ---
    with torch.no_grad():
        outputs = model(prompt_tensor)
    logits = outputs.logits[0, -1, :] # Logits for the next token
    
    # Get top predicted token from full model output
    top_pred_id = torch.argmax(logits).item()
    top_pred_str = format_token(top_pred_id, tokenizer)
    top_pred_logit = logits[top_pred_id].item()
    
    print("\n" + "=" * 60)
    print(f"Target Prompt: '{TARGET_PROMPT}'")
    print(f"Value Token (T_in): '{T_in_str}'")
    print(f"Expected Output Tokens: {EXPECTED_STRS}")
    print("-" * 60)
    print(f"Full Model Prediction (T_out): '{top_pred_str}' (Logit: {top_pred_logit:.3f})")
    print("=" * 60 + "\n")
    
    # --- 2. Extract W_U and W_E Matrices ---
    W_U = model.lm_head.weight.data.cpu().numpy() # (d_vocab, d_model)
    W_E = model.transformer.wte.weight.data # (d_vocab, d_model)
    E_in = W_E[T_in_id, :].cpu().numpy() # (d_model,)
    W_U_T = W_U.T # (d_model, d_vocab)
    
    # Store results for every head
    head_boost_results = {}
    
    # --- 3. Iterate and Analyze Heads ---
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing Heads"):
        block = model.transformer.h[layer_idx]
        
        # Extract W^V and W^O Full Matrices
        W_QKC = block.attn.c_attn.weight.data 
        W_O_full = block.attn.c_proj.weight.data 
        W_V_full = W_QKC[:, 2 * d_model : 3 * d_model] 
        W_O = W_O_full.T.cpu().numpy() 
        W_V_full = W_V_full.cpu().numpy()
        
        for head_idx in range(num_heads):
            
            # Head-specific Slicing (W^V_h and W^O_h)
            start = head_idx * d_head
            end = (head_idx + 1) * d_head
            
            W_V_h = W_V_full[:, start:end] # (d_model, d_head)
            W_O_h = W_O[start:end, :]      # (d_head, d_model)
            
            # 4. Calculate OVW_U Logit Boost Vector
            # M_h = W_V_h @ W_O_h                   # (d_model, d_model)
            # A_h = M_h @ W_U_T                     # (d_model, d_vocab)
            # logit_boost_vector = E_in @ A_h       # (d_vocab,)
            
            # Efficient combined calculation
            logit_boost_vector = E_in @ W_V_h @ W_O_h @ W_U_T
            
            # 5. Calculate Total Boost for Expected Tokens
            # Sum the boosts for all target tokens
            total_expected_boost = np.sum(logit_boost_vector[EXPECTED_IDS])
            
            # Store results
            head_boost_results[(layer_idx, head_idx)] = total_expected_boost

    # --- 6. Report Results ---
    
    # Find the head with the maximum boost
    best_head, max_boost = max(
        head_boost_results.items(), 
        key=lambda item: item[1]
    )
    
    L_best, H_best = best_head
    
    # Get the boost for the individual expected tokens from the best head
    L_best_block = model.transformer.h[L_best]
    W_QKC = L_best_block.attn.c_attn.weight.data 
    W_O_full = L_best_block.attn.c_proj.weight.data 
    W_V_full = W_QKC[:, 2 * d_model : 3 * d_model] 
    W_O = W_O_full.T.cpu().numpy() 
    W_V_full = W_V_full.cpu().numpy()

    start = H_best * d_head
    end = (H_best + 1) * d_head
    
    W_V_h = W_V_full[:, start:end]
    W_O_h = W_O[start:end, :]
    logit_boost_vector_best = E_in @ W_V_h @ W_O_h @ W_U_T
    
    individual_boosts = logit_boost_vector_best[EXPECTED_IDS]
    
    details = []
    for i in range(len(EXPECTED_STRS)):
        details.append(f"'{EXPECTED_STRS[i]}' ({individual_boosts[i]:.3f})")
    
    print("\n" + "=" * 60)
    print("🎯 **Targeted Skip-Trigram Head Identification** 🎯")
    print("-" * 60)
    print(f"🥇 **Best Head (L, H): Layer {L_best:02d}, Head {H_best:02d}**")
    print(f"  -> Total Logit Boost for Expected Tokens: **{max_boost:.3f}**")
    print("-" * 60)
    print("Individual Boosts from this Head:")
    print(f"  [{', '.join(details)}]")
    print("=" * 60)

if __name__ == "__main__":
    analyze_trigram_head_contribution()
