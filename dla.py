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

def find_target_tokens_and_mask(prompt: str, tokenizer):
    """
    Splits the prompt by '...' to find the Query/Key tokens (context) and the 
    Value token (T_in). The T_in is always the last token of the prompt.
    
    Returns:
        prompt_tensor: Tensor of all tokens.
        T_in_position: Index of the Value Token (T_in) in the tensor.
        T_in_id: ID of the Value Token.
    """
    parts = prompt.split('...')
    
    # 1. Tokenize all parts
    token_id_lists = [tokenizer.encode(part) for part in parts]
    
    # 2. Recombine the token lists
    all_token_ids = []
    for i, part_tokens in enumerate(token_id_lists):
        # Only add a single token in between to represent the '...' gap.
        # This token won't be used, but maintains the sequence length.
        # For simplicity, we just concatenate, and the attention mask will handle the 'skip'.
        all_token_ids.extend(part_tokens)

    if not all_token_ids:
        raise ValueError("Prompt resulted in no tokens.")
        
    # The last token is always the T_in (Value Token)
    T_in_id = all_token_ids[-1]
    T_in_position = len(all_token_ids) - 1
    
    return torch.tensor([all_token_ids]), T_in_position, T_in_id

def analyze_head_path_to_output(model, tokenizer, L_best: int, H_best: int, T_in_id: int, 
                                EXPECTED_IDS: list, E_in: np.ndarray):
    """
    Calculates the path logit contribution from the best head (L_best, H_best) 
    through all subsequent MLP and Attention W^O matrices to the final W_U.
    """
    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    d_head = d_model // model.config.num_attention_heads
    
    # --- Weight Extraction ---
    W_U = model.lm_head.weight.data.cpu().numpy()
    W_U_T = W_U.T # (d_model, d_vocab)
    
    # 1. Calculate the Best Head's full OV Matrix (d_model, d_model) as the STARTING POINT
    block_best = model.transformer.h[L_best]
    W_QKC = block_best.attn.c_attn.weight.data 
    W_O_full = block_best.attn.c_proj.weight.data 
    W_V_full = W_QKC[:, 2 * d_model : 3 * d_model].cpu().numpy() 
    
    # W_O for full Attention block: Stored (768, 768) -> Transposed (768, 768) for dot
    W_O = W_O_full.transpose(0, 1).cpu().numpy() 
    
    start = H_best * d_head
    end = (H_best + 1) * d_head
    W_V_h = W_V_full[:, start:end] # (d_model, d_head)
    W_O_h = W_O[start:end, :]      # (d_head, d_model)
    
    # Starting path matrix: V_h * O_h (d_model, d_model)
    path_matrix_product = W_V_h.dot(W_O_h) 
    
    print(f"\nTracing path from Layer {L_best} to Output (W_U)...")
    
    # We iterate over layers L_best, L_best+1, ..., L_N-1
    for layer_idx in tqdm(range(L_best, num_layers), desc="Building Path Matrix Product"):
        block = model.transformer.h[layer_idx]
        
        # --- 2a. MLP Path (W_MLP) ---
        
        # W_in_mlp_stored: (3072, 768). We need W_in (768, 3072)
        W_in_mlp = block.mlp.c_fc.weight.data.cpu().numpy().T
        
        # W_out_mlp_stored: (768, 3072). We need W_out (3072, 768)
        W_out_mlp = block.mlp.c_proj.weight.data.cpu().numpy().T
        
        # Calculate the full linear MLP contribution matrix: W_in @ W_out = (768, 768)
        # This multiplication (W_in (768, 3072) @ W_out (3072, 768)) is GUARANTEED to be (768, 768).
        W_MLP_linear = W_in_mlp.dot(W_out_mlp) 
        
        # Apply the MLP matrix (768, 768)
        path_matrix_product = path_matrix_product.dot(W_MLP_linear)
        
        # --- 2b. Attention Path (W_O) of the *NEXT* layer ---
        if layer_idx < num_layers - 1:
            next_block = model.transformer.h[layer_idx + 1]
            
            # W_O_attn is the full Attention Output matrix (d_model, d_model)
            # Stored as (768, 768) -> Transposed (768, 768) for dot
            W_O_attn = next_block.attn.c_proj.weight.data.cpu().numpy().T
            
            # Apply the Attention W_O matrix (768, 768)
            path_matrix_product = path_matrix_product.dot(W_O_attn)

    # 3. Apply the final W_U
    full_circuit_matrix = path_matrix_product.dot(W_U_T) # (d_model, d_vocab)
    
    # 4. Calculate the Logit Boost (E_in @ Full_Circuit_Matrix)
    logit_boost_vector = E_in.dot(full_circuit_matrix) # (d_vocab,)
    
    # 5. Calculate Total Boost for Expected Tokens
    total_path_boost = np.sum(logit_boost_vector[EXPECTED_IDS])
    
    return total_path_boost, logit_boost_vector[EXPECTED_IDS]

def analyze_trigram_head_contribution():
    # ... (rest of the original function remains the same)
    # The rest of the original function is assumed to be correct and is not repeated here.
    # ...
    MODEL_NAME = "gpt2"
    TARGET_PROMPT = 'open ... ,'
    EXPECTED_CONTINUATIONS = ['rb','wb','r','w']
    
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # --- Data Pre-processing ---
    try:
        prompt_tensor, T_in_position, T_in_id = find_target_tokens_and_mask(
            TARGET_PROMPT, tokenizer
        )
    except ValueError as e:
        print(f"Error processing prompt: {e}")
        return

    T_in_str = format_token(T_in_id, tokenizer)
    expected_ids = tokenizer.convert_tokens_to_ids(EXPECTED_CONTINUATIONS)
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
    
    # --- 1. Run Full Inference (For context) ---
    with torch.no_grad():
        outputs = model(prompt_tensor)
    logits = outputs.logits[0, -1, :] 
    
    top_pred_id = torch.argmax(logits).item()
    top_pred_str = format_token(top_pred_id, tokenizer)
    top_pred_logit = logits[top_pred_id].item()
    
    print("\n" + "=" * 60)
    print(f"Target Prompt: '{TARGET_PROMPT}'")
    print(f"Value Token (T_in): '{T_in_str}' (ID: {T_in_id}, Position: {T_in_position})")
    print(f"Expected Output Tokens: {EXPECTED_STRS}")
    print("-" * 60)
    print(f"Full Model Prediction (T_out): '{top_pred_str}' (Logit: {top_pred_logit:.3f})")
    print("=" * 60 + "\n")
    
    # --- 2. Extract W_U and W_E Matrices ---
    W_U = model.lm_head.weight.data.cpu().numpy() 
    W_E = model.transformer.wte.weight.data 
    E_in = W_E[T_in_id, :].cpu().numpy() 
    W_U_T = W_U.T 
    
    # Store results for every head
    head_boost_results = {}
    
    # --- 3. Iterate and Analyze Heads (Find the Best) ---
    
    for layer_idx in tqdm(range(num_layers), desc="Analyzing Heads (W^V W^O W_U)"):
        block = model.transformer.h[layer_idx]
        
        W_QKC = block.attn.c_attn.weight.data 
        W_O_full = block.attn.c_proj.weight.data 
        W_V_full = W_QKC[:, 2 * d_model : 3 * d_model] 
        W_O = W_O_full.T.cpu().numpy() 
        W_V_full = W_V_full.cpu().numpy() 
        
        for head_idx in range(num_heads):
            
            start = head_idx * d_head
            end = (head_idx + 1) * d_head
            
            W_V_h = W_V_full[:, start:end] 
            W_O_h = W_O[start:end, :]      
            
            # 4. Calculate OVW_U Logit Boost Vector (E_in @ W_V_h @ W_O_h @ W_U_T)
            logit_boost_vector = E_in.dot(W_V_h).dot(W_O_h).dot(W_U_T)
            
            # 5. Calculate Total Boost for Expected Tokens
            total_expected_boost = np.sum(logit_boost_vector[EXPECTED_IDS])
            
            head_boost_results[(layer_idx, head_idx)] = total_expected_boost

    # --- 6. Report Results (Best Head) ---
    best_head, max_boost = max(head_boost_results.items(), key=lambda item: item[1])
    L_best, H_best = best_head
    
    L_best_block = model.transformer.h[L_best]
    W_QKC = L_best_block.attn.c_attn.weight.data 
    W_O_full = L_best_block.attn.c_proj.weight.data 
    W_V_full = W_QKC[:, 2 * d_model : 3 * d_model].cpu().numpy()
    W_O = W_O_full.T.cpu().numpy() 

    start = H_best * d_head
    end = (H_best + 1) * d_head
    
    W_V_h = W_V_full[:, start:end]
    W_O_h = W_O[start:end, :]
    logit_boost_vector_best = E_in.dot(W_V_h).dot(W_O_h).dot(W_U_T)
    
    individual_boosts_WOU = logit_boost_vector_best[EXPECTED_IDS]
    
    details_WOU = []
    for i in range(len(EXPECTED_STRS)):
        details_WOU.append(f"'{EXPECTED_STRS[i]}' ({individual_boosts_WOU[i]:.3f})")
    
    print("\n" + "=" * 60)
    print("🎯 **Targeted Skip-Trigram Head Identification (W^V W^O W_U)** 🎯")
    print("-" * 60)
    print(f"🥇 **Best Head (L, H): Layer {L_best:02d}, Head {H_best:02d}**")
    print(f"  -> W^V W^O W_U Total Logit Boost: **{max_boost:.3f}**")
    print("-" * 60)
    print("Individual Boosts (W^V W^O W_U):")
    print(f"  [{', '.join(details_WOU)}]")
    print("=" * 60)
    
    
    # --- 7. Trace Path to Output (New Task) ---
    total_path_boost, individual_boosts_path = analyze_head_path_to_output(
        model, tokenizer, L_best, H_best, T_in_id, EXPECTED_IDS, E_in
    )
    
    details_path = []
    for i in range(len(EXPECTED_STRS)):
        details_path.append(f"'{EXPECTED_STRS[i]}' ({individual_boosts_path[i]:.3f})")

    print("\n" + "=" * 60)
    print("🔍 **Best Head Path Tracing (E_in @ (W^V W^O)_L @ W_{MLP} @ W_{Attn} @ ... @ W_U)** 🔍")
    print("-" * 60)
    print(f"✨ **Circuit Path Logit Boost from L{L_best}H{H_best} to Output:**")
    print(f"  -> Total Logit Boost for Expected Tokens: **{total_path_boost:.3f}**")
    print("-" * 60)
    print("Individual Boosts along the Full Path:")
    print(f"  [{', '.join(details_path)}]")
    print("=" * 60)


if __name__ == "__main__":
    analyze_trigram_head_contribution()
