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

# --- MODEL LOADING ---
MODEL_NAME = "gpt2" 
device = "cuda" if torch.cuda.is_available() else "cpu" 

model = HookedTransformer.from_pretrained(
    MODEL_NAME, 
    device=device
)

#text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
text = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere. The Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn't think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's sister, but they hadn't met for several years; in fact, Mrs. Dursley pretended she didn't have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn't want Dudley mixing with a child like that."""

# Run the model and cache the activations
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
attention_pattern = cache["pattern", 0, "attn"] # Shape: [n_heads, seq_len, seq_len]
raw_attn_scores = cache["attn_scores", 0].detach().cpu().numpy() # Shape: [n_heads, seq_len, seq_len]

str_tokens = model.to_str_tokens(text)
n_heads = model.cfg.n_heads
X_FACTORS = 20

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
# --- 3. NMF of ATTENTION PATTERNS (Softmax Probabilities) ---

print("\n--- 3. NMF of ATTENTION PATTERNS (Softmax) - Factors as Selectable 'Heads' Visualization ---")

# 1. Prepare data for NMF: Flatten to [n_heads, seq_len*seq_len]
attention_matrix = attention_pattern.detach().cpu().numpy().reshape(n_heads, -1)

# 2. Perform NMF
nmf_model_attn = NMF(n_components=X_FACTORS, init='random', random_state=42, max_iter=500)
W_matrix_attn = nmf_model_attn.fit_transform(attention_matrix)
H_matrix_attn = nmf_model_attn.components_

# 3. Reshape NMF factors for visualization
nmf_factors_reshaped_attn = H_matrix_attn.reshape(X_FACTORS, attention_pattern.shape[1], attention_pattern.shape[2])

# 4. Normalize the entire NMF factor matrix
min_val = nmf_factors_reshaped_attn.min()
max_val = nmf_factors_reshaped_attn.max()

if max_val - min_val > 1e-8:
    normalized_factors_attn = (nmf_factors_reshaped_attn - min_val) / (max_val - min_val)
else:
    normalized_factors_attn = np.zeros_like(nmf_factors_reshaped_attn)

# 5. Display the visualization
print(f"Showing {X_FACTORS} NMF factors (based on Softmax Attention) as selectable 'Heads'.")
for i in range(X_FACTORS):
    print(f"Head {i}: NMF Factor {i+1} - Mean W-weight: {W_matrix_attn[:, i].mean():.3f}")

#display(cv.attention.attention_patterns(
#    tokens=str_tokens,
#    attention=normalized_factors_attn 
#))
# -------------------------------------------------------------
# --- 4. NMF of LOGIT ATTRIBUTION (Raw Scores) ---

print("\n--- 4. NMF of LOGIT ATTRIBUTION (Raw Scores) - Factors as Selectable 'Heads' Visualization ---")

# NMF requires non-negative input, so we must use the Softplus-transformed scores
# used in step 2, or simply use ReLU(x) to discard negative attribution.
# We will use the non-negative Softplus scores for consistency and stability.
softplus = torch.nn.Softplus()
nmf_input_logits = softplus(torch.tensor(raw_attn_scores)).numpy()

# 1. Prepare data for NMF: Flatten to [n_heads, seq_len*seq_len]
logit_matrix = nmf_input_logits.reshape(n_heads, -1)

# 2. Perform NMF
nmf_model_logit = NMF(n_components=X_FACTORS, init='random', random_state=42, max_iter=500)
W_matrix_logit = nmf_model_logit.fit_transform(logit_matrix)
H_matrix_logit = nmf_model_logit.components_

# 3. Reshape NMF factors for visualization
nmf_factors_reshaped_logit = H_matrix_logit.reshape(X_FACTORS, attention_pattern.shape[1], attention_pattern.shape[2])

# 4. Normalize the entire NMF factor matrix
min_val = nmf_factors_reshaped_logit.min()
max_val = nmf_factors_reshaped_logit.max()

if max_val - min_val > 1e-8:
    normalized_factors_logit = (nmf_factors_reshaped_logit - min_val) / (max_val - min_val)
else:
    normalized_factors_logit = np.zeros_like(nmf_factors_reshaped_logit)

# 5. Display the visualization
print(f"Showing {X_FACTORS} NMF factors (based on Logit Attribution) as selectable 'Heads'.")
for i in range(X_FACTORS):
    print(f"Head {i}: Logit NMF Factor {i+1} - Mean W-weight: {W_matrix_logit[:, i].mean():.3f}")

display(cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=normalized_factors_logit
))
