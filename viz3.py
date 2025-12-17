from huggingface_hub import hf_hub_download
from IPython.display import display

# ... (rest of the imports and setup remain the same)
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv
import torch
import numpy as np 
# IMPORT FOR NMF
from sklearn.decomposition import NMF 

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
# Use "cuda" if available, otherwise "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu" 
weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

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

model = HookedTransformer(cfg).to(device)
pretrained_weights = torch.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

# Run the model and cache the activations
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
attention_pattern = cache["pattern", 0, "attn"] # Shape: [n_heads, seq_len, seq_len]
raw_attn_scores = cache["attn_scores", 0].detach().cpu().numpy() # Shape: [n_heads, seq_len, seq_len]

str_tokens = model.to_str_tokens(text)

# -------------------------------------------------------------
print("--- Original Attention Pattern Visualization (Head Selection Available) ---")
#display(cv.attention.attention_patterns(
#                            tokens=str_tokens,
#                            attention=attention_pattern.detach().cpu().numpy()))
# -------------------------------------------------------------

# --- 1. Info Weighted Patterns (W_OV Norm Weighted Attention) ---

print("\n--- 1. Info Weighted Patterns (W_OV Norm Weighted Attention) Visualization ---")

# Calculate the L2 norm of the W_O matrix for each head
W_O_norm = torch.linalg.norm(model.blocks[0].attn.W_O, dim=1).detach().cpu().numpy()
W_O_head_norm = W_O_norm.mean(axis=1) 

# Reshape the norms for multiplication: [n_heads] -> [n_heads, 1, 1]
norm_weights = W_O_head_norm[:, np.newaxis, np.newaxis]

# Multiply the attention pattern by the norm weights
norm_weighted_attention = attention_pattern.detach().cpu().numpy() * norm_weights

# Visualize the norm-weighted pattern
#display(cv.attention.attention_patterns(
#                            tokens=str_tokens,
#                            attention=norm_weighted_attention))
# -------------------------------------------------------------

# --- 2. Head Attention Logit Attribution Patterns ---

print("\n--- 2. Head Attention Logit Attribution Patterns Visualization (Softplus and Normalized Attn Scores) ---")

# Apply Softplus and Min-Max Scaling to improve logit visualization contrast
softplus = torch.nn.Softplus()
softplus_scores = softplus(torch.tensor(raw_attn_scores)).numpy()

min_val = softplus_scores.min()
max_val = softplus_scores.max()

if max_val - min_val > 1e-8:
    normalized_attn_scores = (softplus_scores - min_val) / (max_val - min_val)
else:
    normalized_attn_scores = np.ones_like(softplus_scores) * 0.5 

# Display the normalized attention scores.
#display(cv.attention.attention_patterns(
#                            tokens=str_tokens,
#                            attention=normalized_attn_scores))
# -------------------------------------------------------------

# --- 3. Info Weighted Patterns using NMF (Non-negative Matrix Factorization) of X factors ---

print("\n--- 3. Info Weighted Patterns using NMF (Factors as Selectable 'Heads') Visualization ---")

X_FACTORS = 10 
n_heads = cfg.n_heads

# 1. Prepare data for NMF: Flatten to [n_heads, seq_len*seq_len]
attention_matrix = attention_pattern.detach().cpu().numpy().reshape(n_heads, -1)

# 2. Perform NMF
nmf_model = NMF(n_components=X_FACTORS, init='random', random_state=42, max_iter=500)
# W_matrix (Head-Factor weights) and H_matrix (NMF Factors)
W_matrix = nmf_model.fit_transform(attention_matrix)
H_matrix = nmf_model.components_

# 3. Reshape NMF factors for visualization: [X_FACTORS, seq_len, seq_len]
nmf_factors_reshaped = H_matrix.reshape(X_FACTORS, attention_pattern.shape[1], attention_pattern.shape[2])

# 4. Normalize the entire NMF factor matrix for consistent visualization
min_val = nmf_factors_reshaped.min()
max_val = nmf_factors_reshaped.max()

if max_val - min_val > 1e-8:
    normalized_factors = (nmf_factors_reshaped - min_val) / (max_val - min_val)
else:
    normalized_factors = np.zeros_like(nmf_factors_reshaped)

# 5. Display the visualization. We will also print the W-matrix information separately
# for each "Factor" (which circuitsvis will label "Head X").
print(f"Showing {X_FACTORS} NMF factors as selectable 'Heads'. Please refer to the text below for Factor information.")
for i in range(X_FACTORS):
    # Print the relevant NMF factor information (e.g., mean W-weight)
    print(f"Head {i}: NMF Factor {i+1} - Mean W-weight: {W_matrix[:, i].mean():.3f}")

# Display the NMF factors. circuitsvis will label them numerically (Head 0, Head 1, etc.).
display(cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=normalized_factors 
))
