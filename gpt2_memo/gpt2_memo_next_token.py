"""
Full Python script for Mechanistic Interpretability analysis of Memorization
in gpt2-small, using transformer_lens.

This script implements a multi-step plan:
1.  Setup: Load model, define clean/corrupted data, and define the metric.
2.  Step 2: High-Level Localization (Residual Stream Patching)
3.  Step 3: Component-Level Localization (MLP vs. Attention)
4.  Step 4: Neuron-Level Analysis (Ablation)
5.  Step 5: Finding "Backup" Neurons
6.  Bonus: Attention Pattern Visualization
7.  Bonus: Linear Probing for Memorization Neurons
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from typing import Callable, Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
import circuitsvis as cv
from IPython.display import display, HTML

# ---
# 1. SETUP & CONFIGURATION
# ---

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("Warning: Running on CPU. This will be very slow.")

MODEL_NAME = "gpt2-small"
print(f"Loading model: {MODEL_NAME} to {DEVICE}")
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    device=DEVICE
)
model.eval() # Set to evaluation mode

# --- Data & Metric Definition ---


# The "memorized" context (without the target)
CLEAN_TEXT = "By clicking register, you agree to,"
CORRUPTED_TEXT = "By choosing register, you agree to,"

TARGET_TOKEN_STR = " Etsy"

# --- Automatic TARGET_POS_INDEX Calculation ---

def calculate_target_position(model, text, target_token_str):
    """
    Calculates TARGET_POS_INDEX when predicting the next token after the sequence.
    Accounts for the <|endoftext|> token that GPT-2 adds at the beginning.
    """
    tokens = model.to_str_tokens(text)
    
    print("Tokenization analysis:")
    print(f"Text: '{text}'")
    print("Tokens:")
    for i, token in enumerate(tokens):
        print(f"  Position {i}: '{token}'")
    
    # The target position is where we want to measure the prediction
    # For next token prediction, we want the position BEFORE the prediction
    # If we have tokens [BOS, t1, t2, ..., tn], we predict after tn (position n)
    # But loss is only computed up to position n-1 for predicting tokens 1..n
    
    # So TARGET_POS_INDEX should be the last position where we have a loss value
    # This is position len(tokens)-2 because:
    # - tokens: [BOS, t1, t2, ..., tn] (n+1 tokens total)
    # - loss: computed for predicting [t1, t2, ..., tn] (n loss values at positions 0..n-1)
    # - We want to predict after the last token tn, so we use position n-1
    
    target_pos_index = len(tokens) - 2
    
    print(f"\nWill measure prediction at position {target_pos_index} (predicting token '{tokens[target_pos_index + 1]}')")
    print(f"Expected prediction: '{target_token_str}'")
    
    return target_pos_index

# Calculate TARGET_POS_INDEX for next token prediction
TARGET_POS_INDEX = calculate_target_position(model, CLEAN_TEXT, TARGET_TOKEN_STR)
TARGET_TOKEN_ID = model.to_single_token(TARGET_TOKEN_STR)

print(f"\nTarget position index: {TARGET_POS_INDEX} (token: '{model.to_str_tokens(CLEAN_TEXT)[TARGET_POS_INDEX]}')")
print(f"Prediction target: {TARGET_TOKEN_STR}")

# --- Fix the get_token_loss function to handle BOS token ---


def get_token_loss(
    logits: torch.Tensor,
    target_token_id: int,
    target_pos_index: int
) -> float:
    """
    Calculates the negative log probability (loss) for the target token
    at the specified target position.
    """
    # Get log probabilities for all tokens at the target position
    log_probs = F.log_softmax(logits[0, target_pos_index, :], dim=-1)
    
    # Get the log probability of the target token
    target_log_prob = log_probs[target_token_id].item()
    
    # Convert to negative log probability (loss)
    return -target_log_prob

# --- Metric Helper Functions ---

def calculate_metric_restoration(
    patched_loss: float,
    corrupted_loss: float,
    clean_loss: float
) -> float:
    """
    Measures how much of the "gap" between corrupted and clean loss
    was "restored" by the patch. 100% means full restoration.
    """
    if (corrupted_loss - clean_loss) == 0:
        return 0.0
    return (corrupted_loss - patched_loss) / (corrupted_loss - clean_loss)

# --- Hook Helper Functions ---


def patch_hook_factory(
    clean_act_tensor: torch.Tensor,
    pos_to_patch: int
) -> Callable:
    """
    Returns a hook function that patches in activations from
    the clean run at a specific position.
    
    clean_act_tensor is expected to be 2D ([seq_len, d_features]) due to
    remove_batch_dim=True in run_with_cache.
    """
    def patch_hook(
        act: torch.Tensor,
        hook: HookPoint
    ) -> torch.Tensor:
        # Patch in the activation from the clean cache
        # act shape: [batch=1, pos, d_model] (3D)
        # clean_act_tensor shape: [pos, d_model] (2D)
        
        # The positions in clean_act_tensor already account for BOS token
        act[:, pos_to_patch, :] = clean_act_tensor[pos_to_patch, :] 
        return act
    return patch_hook

def ablate_neuron_hook_factory(
    neuron_indices: torch.Tensor,
    pos_to_ablate: int
) -> Callable:
    """
    Returns a hook function that ablates (zeros out) specific
    neurons at a specific position.
    """
    def ablate_hook(
        act: torch.Tensor,
        hook: HookPoint
    ) -> torch.Tensor:
        # act shape: [batch, pos, d_mlp]
        act[:, pos_to_ablate, neuron_indices] = 0.0
        return act
    return ablate_hook

# --- Plotting Helper Function ---

def plot_heatmap(
    data: torch.Tensor,
    title: str,
    x_labels: List[str],
    y_labels: List[str]
):
    """
    Uses Plotly to create and show an interactive heatmap.
    """
    fig = px.imshow(
        data,
        labels=dict(x="Token Position", y="Layer", color="Loss Restoration %"),
        x=x_labels,
        y=y_labels,
        title=title,
        color_continuous_scale='RdYlGn', # Red (bad) -> Yellow (mid) -> Green (good)
        zmin=0,
        zmax=1.0 # Scale from 0% to 100% restoration
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        width=1000,
        height=600
    )
    # Use fig.show() to work in scripts and notebooks
    fig.show()

# ---
# 2. HIGH-LEVEL LOCALIZATION (Residual Stream Patching)
# ---

def run_residual_stream_patching(
    model: HookedTransformer,
    clean_cache: Dict,
    corrupted_text: str,
    baseline_corrupted_loss: float,
    baseline_clean_loss: float
) -> torch.Tensor:
    """
    Patches the residual stream from clean to corrupted run at every
    (layer, position) and records the metric restoration.
    """
    print("\n--- Running Step 2: Residual Stream Patching ---")
    
    str_tokens = model.to_str_tokens(corrupted_text)
    n_layers = model.cfg.n_layers
    seq_len = len(str_tokens)
    
    # Store results in a matrix: [n_layers, seq_len]
    patching_results = torch.zeros((n_layers, seq_len), device="cpu")

    for layer in range(n_layers):
        for pos in range(seq_len):
            # Get the hook name for the residual stream output
            hook_name = utils.get_act_name("resid_post", layer)
            
            # Create the patching hook
            patch_hook = patch_hook_factory(
                clean_act_tensor=clean_cache[hook_name].to(DEVICE),
                pos_to_patch=pos
            )
            
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(
                corrupted_text,
                fwd_hooks=[(hook_name, patch_hook)],
                return_type="logits"
            )
            
            # Calculate and store the restored metric
            patched_loss = get_token_loss(patched_logits, TARGET_TOKEN_ID, TARGET_POS_INDEX)
            
            patching_results[layer, pos] = calculate_metric_restoration(
                patched_loss,
                baseline_corrupted_loss,
                baseline_clean_loss
            )
            
        print(f"Finished patching layer {layer+1}/{n_layers}")
        
    return patching_results

# ---
# 3. COMPONENT-LEVEL LOCALIZATION (MLP vs. Attention)
# ---

def run_component_patching(
    model: HookedTransformer,
    clean_cache: Dict,
    corrupted_text: str,
    critical_layer: int,
    critical_pos: int,
    baseline_corrupted_loss: float,
    baseline_clean_loss: float
):
    """
    At the critical (layer, pos) found in Step 2, patch
    MLP and Attention components separately.
    """
    print(f"\n--- Running Step 3: Component Patching at (L{critical_layer}, P{critical_pos}) ---")

    # 1. Patch MLP
    mlp_hook_name = utils.get_act_name("mlp_out", critical_layer)
    patch_hook_mlp = patch_hook_factory(
        clean_act_tensor=clean_cache[mlp_hook_name].to(DEVICE),
        pos_to_patch=critical_pos
    )
    patched_logits_mlp = model.run_with_hooks(
        corrupted_text,
        fwd_hooks=[(mlp_hook_name, patch_hook_mlp)],
    )
    mlp_patched_loss = get_token_loss(patched_logits_mlp, TARGET_TOKEN_ID, TARGET_POS_INDEX)
    mlp_restoration = calculate_metric_restoration(
        mlp_patched_loss, baseline_corrupted_loss, baseline_clean_loss
    )

    # 2. Patch Attention
    attn_hook_name = utils.get_act_name("attn_out", critical_layer)
    patch_hook_attn = patch_hook_factory(
        clean_act_tensor=clean_cache[attn_hook_name].to(DEVICE),
        pos_to_patch=critical_pos
    )
    patched_logits_attn = model.run_with_hooks(
        corrupted_text,
        fwd_hooks=[(attn_hook_name, patch_hook_attn)],
    )
    attn_patched_loss = get_token_loss(patched_logits_attn, TARGET_TOKEN_ID, TARGET_POS_INDEX)
    attn_restoration = calculate_metric_restoration(
        attn_patched_loss, baseline_corrupted_loss, baseline_clean_loss
    )

    # Print results
    print(f"Baseline Clean Loss:     {baseline_clean_loss:.3f}")
    print(f"Baseline Corrupted Loss: {baseline_corrupted_loss:.3f}")
    print("-" * 30)
    print(f"MLP Patch Restored Loss:     {mlp_patched_loss:.3f} (Restored {mlp_restoration*100:.1f}%)")
    print(f"Attn Patch Restored Loss:    {attn_patched_loss:.3f} (Restored {attn_restoration*100:.1f}%)")
    
    if mlp_restoration > attn_restoration:
        print("\n=> MLP component appears more important at this position.")
    else:
        print("\n=> Attention component appears more important at this position.")

# ---
# 4. NEURON-LEVEL ANALYSIS (Ablation)
# ---

def run_neuron_ablation(
    model: HookedTransformer,
    clean_text: str,
    clean_cache: Dict,
    critical_layer: int,
    critical_pos: int,
    baseline_clean_loss: float,
    k: int = 20
) -> torch.Tensor:
    """
    Finds the top-k activating neurons in the critical MLP
    and ablates them to see if it breaks memorization.
    """
    print(f"\n--- Running Step 4: Neuron Ablation at (L{critical_layer}, P{critical_pos}) ---")
    
    # Get activations from the MLP's non-linearity. 
    # Cache is 2D: [seq_len, d_mlp].
    mlp_post_acts = clean_cache[utils.get_act_name("post", critical_layer)][critical_pos, :] # [d_mlp]
    
    # Find top k activating neurons
    top_k_neurons_vals, top_k_neurons_indices = torch.topk(mlp_post_acts, k)
    
    print(f"Found Top {k} neurons in L{critical_layer} at P{critical_pos}: {top_k_neurons_indices.cpu().numpy()}")

    # Create the ablation hook
    ablate_hook = ablate_neuron_hook_factory(
        neuron_indices=top_k_neurons_indices,
        pos_to_ablate=critical_pos
    )
    
    # Run on the *clean* text. Does ablating them *break* memorization?
    ablated_logits = model.run_with_hooks(
        clean_text,
        fwd_hooks=[(utils.get_act_name("post", critical_layer), ablate_hook)],
    )
    ablated_loss = get_token_loss(ablated_logits, TARGET_TOKEN_ID, TARGET_POS_INDEX)

    print(f"Clean Loss (original): {baseline_clean_loss:.3f}")
    print(f"Clean Loss (ablated):  {ablated_loss:.3f}")
    
    if ablated_loss > baseline_clean_loss * 1.5: # (Arbitrary threshold for "broken")
        print("\n=> Ablating these neurons significantly *broke* memorization.")
    else:
        print("\n=> Ablating these neurons did not break memorization.")
        
    return top_k_neurons_indices

# ---
# 5. FINDING "BACKUP" NEURONS
# ---

def find_backup_neurons(
    model: HookedTransformer,
    clean_text: str,
    main_neuron_indices: torch.Tensor,
    critical_layer: int,
    critical_pos: int,
    k: int = 20
):
    """
    Finds the top-k activating neurons *after* the main
    neurons have been ablated.
    """
    print(f"\n--- Running Step 5: Finding Backup Neurons ---")
    
    # Create the ablation hook for the *main* neurons
    ablate_hook = ablate_neuron_hook_factory(
        neuron_indices=main_neuron_indices,
        pos_to_ablate=critical_pos
    )
    
    # FIXED: Use run_with_hooks and manually cache the specific activation we need
    hook_name = utils.get_act_name("post", critical_layer)
    
    # We'll manually capture the activation using a second hook
    backup_activations = {}
    def capture_hook(act, hook):
        backup_activations[hook.name] = act.detach().clone()
    
    # Run with both hooks: first ablate, then capture
    _ = model.run_with_hooks(
        clean_text,
        fwd_hooks=[
            (hook_name, ablate_hook),      # First: ablate main neurons
            (hook_name, capture_hook)      # Then: capture the result
        ],
        return_type="logits"
    )
    
    # Get MLP activations from our manual cache
    # Shape will be [batch, seq_len, d_mlp] - we need to remove batch dim if present
    backup_mlp_post_acts = backup_activations[hook_name]
    if backup_mlp_post_acts.dim() == 3:  # [batch, seq_len, d_mlp]
        backup_mlp_post_acts = backup_mlp_post_acts[0]  # remove batch dim
    
    # Get the activations at our critical position
    backup_mlp_post_acts = backup_mlp_post_acts[critical_pos, :]  # [d_mlp]
    
    # Find top k activating neurons *from the ablated run*
    top_k_backup_vals, top_k_backup_indices = torch.topk(backup_mlp_post_acts, k)
    
    print(f"Main circuit neurons:    {main_neuron_indices.cpu().numpy()}")
    print(f"Backup circuit neurons:  {top_k_backup_indices.cpu().numpy()}")
    
    # Check for overlap
    overlap = set(main_neuron_indices.cpu().numpy()) & set(top_k_backup_indices.cpu().numpy())
    if overlap:
        print(f"Warning: Found overlap: {overlap}. This may mean the backup neurons are also part of the main circuit.")
    else:
        print("\n=> Found a distinct set of 'backup' neurons.")
# ---
# 6. BONUS: ATTENTION PATTERN VISUALIZATION
# ---

def visualize_attention(
    clean_cache: Dict,
    clean_text: str,
    layer: int = 0
):
    """
    Uses circuitsvis to display the attention patterns for the clean run.
    NOTE: This will try to open in your browser or display in a notebook.
    """
    print(f"\n--- Bonus: Visualizing Attention Patterns (Layer {layer}) ---")
    str_tokens = model.to_str_tokens(clean_text)
    # Cache stores 2D attention pattern [n_heads, seq_len, seq_len] due to remove_batch_dim=True
    attention_pattern = clean_cache["pattern", layer, "attn"].to("cpu").unsqueeze(0) # Add back batch dimension for circuitsvis
    
    # Create the visualization
    vis_html = cv.attention.attention_patterns(
        tokens=str_tokens,
        attention=attention_pattern
    )
    
    print(f"Displaying attention for L{layer}. (This may open in a browser or notebook cell)")
    # This works best in a Jupyter Notebook
    try:
        # FIX: Use the RenderedHTML object directly or convert to string
        display(HTML(str(vis_html)))
    except Exception as e:
        print(f"Could not display inline. You may need to run this in a Jupyter Notebook or similar environment.")
        print(f"Error: {e}")
        # Fallback to saving to a file
        try:
            with open("attention_L0.html", "w") as f:
                f.write(str(vis_html))
            print("Saved visualization to attention_L0.html")
        except Exception as save_error:
            print(f"Failed to save visualization: {save_error}")

# ---
# 7. BONUS: LINEAR PROBING
# ---

def build_and_run_probe(
    model: HookedTransformer,
    critical_layer: int
):
    """
    Builds a small dataset and trains a linear probe to find
    neurons predictive of memorization.
    """
    print(f"\n--- Bonus: Linear Probing (L{critical_layer}) ---")
    
    # 1. Build a small dataset
    # (In a real experiment, you'd use 1000s of examples)
    memorized_texts = [
        "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
        "He was a big, beefy man with hardly any neck, although he did have a very large mustache.",
        "The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere."
    ]
    non_memorized_texts = [
        "This is a test sentence that is definitely not from a famous book, I hope.",
        "The quick brown fox jumps over the lazy dog, a classic example of a pangram.",
        "My custom script for analyzing neural networks works quite well, I must say."
    ]
    
    X = [] # Activations
    y = [] # Labels (1=memorized, 0=not)
    
    print("Building probe dataset...")
    hook_name = utils.get_act_name("post", critical_layer)
    
    for text in memorized_texts:
        # remove_batch_dim=True makes the cache 2D
        _, cache = model.run_with_cache(text, remove_batch_dim=True)
        # Add *all* token activations from this sentence. Cache is [seq_len, d_mlp]
        acts = cache[hook_name].cpu().numpy()
        X.append(acts)
        y.append(np.ones(acts.shape[0]))

    for text in non_memorized_texts:
        # remove_batch_dim=True makes the cache 2D
        _, cache = model.run_with_cache(text, remove_batch_dim=True)
        # Add *all* token activations from this sentence. Cache is [seq_len, d_mlp]
        acts = cache[hook_name].cpu().numpy()
        X.append(acts)
        y.append(np.zeros(acts.shape[0]))
        
    # Combine into single arrays
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # 2. Train the probe
    print("Training Logistic Regression probe...")
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X, y)
    print(f"Probe accuracy: {probe.score(X, y)*100:.2f}%")
    
    # 3. Analyze coefficients
    # The coefficients (probe.coef_) tell us which neurons
    # were most predictive.
    neuron_coeffs = probe.coef_[0] # Shape [d_mlp]
    
    top_probe_neurons_indices = np.argsort(-neuron_coeffs)[:10]
    top_probe_neurons_coeffs = neuron_coeffs[top_probe_neurons_indices]
    
    print(f"Top 10 neurons found by probe (L{critical_layer}):")
    for i, (idx, val) in enumerate(zip(top_probe_neurons_indices, top_probe_neurons_coeffs)):
        print(f"  {i+1}. Neuron {idx} (Coefficient: {val:.3f})")
        
    return top_probe_neurons_indices

# ---
# MAIN EXECUTION
# ---



def main():
    """
    Runs the full analysis pipeline.
    """
    
    # --- Step 1: Get Baselines ---
    clean_tokens = model.to_str_tokens(CLEAN_TEXT)
    print(f"\nTarget position index: {TARGET_POS_INDEX} (token: '{clean_tokens[TARGET_POS_INDEX]}')")
    print(f"Prediction target: {TARGET_TOKEN_STR}")

    # Get clean logits, cache, and baseline loss
    print("Running clean text...")
    clean_logits, clean_cache = model.run_with_cache(CLEAN_TEXT, remove_batch_dim=True)
    baseline_clean_loss = get_token_loss(clean_logits, TARGET_TOKEN_ID, TARGET_POS_INDEX)
    print(f"Baseline Clean Loss:     {baseline_clean_loss:.3f}")

    # Get corrupted logits and baseline loss
    print("Running corrupted text...")
    corrupted_logits, _ = model.run_with_cache(CORRUPTED_TEXT, remove_batch_dim=True)
    baseline_corrupted_loss = get_token_loss(corrupted_logits, TARGET_TOKEN_ID, TARGET_POS_INDEX)
    print(f"Baseline Corrupted Loss: {baseline_corrupted_loss:.3f}")
    
    if baseline_corrupted_loss <= baseline_clean_loss:
        print("\n*** WARNING ***")
        print("Corrupted loss is not higher than clean loss.")
        print("This memorization metric is not effective for this example.")
        print("Try a different text or perturbation.")
        print("The model might not have memorized this specific association.")
        print("You might need to use a more distinctive example.")
        return

    # --- Step 2: Run Residual Stream Patching ---
    patching_results = run_residual_stream_patching(
        model,
        clean_cache,
        CORRUPTED_TEXT,
        baseline_corrupted_loss,
        baseline_clean_loss
    )
    
    # Visualize the results
    str_tokens = model.to_str_tokens(CORRUPTED_TEXT)
    token_labels = [f"{i}: '{s}'" for i, s in enumerate(str_tokens)]
    layer_labels = [f"L{i}" for i in range(model.cfg.n_layers)]
    
    plot_heatmap(
        patching_results.cpu().numpy(),
        "Residual Stream Patching: Metric Restoration (%)",
        x_labels=token_labels,
        y_labels=layer_labels
    )
    
    # --- Analysis of Patching Results ---
    # Find the hotspot (max restoration)
    max_val, max_idx = torch.max(patching_results.flatten(), 0)
    max_pos = max_idx.item() % patching_results.shape[1]
    max_layer = max_idx.item() // patching_results.shape[1]
    
    print(f"\nPatching Hotspot: Max restoration ({max_val*100:.1f}%) found at:")
    print(f"  CRITICAL_LAYER = {max_layer}")
    print(f"  CRITICAL_POS   = {max_pos} (Token: '{str_tokens[max_pos]}')")

    # --- Step 3: Run Component Patching ---
    # !! Note: We use the CRITICAL_LAYER and CRITICAL_POS found above
    run_component_patching(
        model,
        clean_cache,
        CORRUPTED_TEXT,
        critical_layer=max_layer,
        critical_pos=max_pos,
        baseline_corrupted_loss=baseline_corrupted_loss,
        baseline_clean_loss=baseline_clean_loss
    )
    
    # --- Step 4: Run Neuron Ablation ---
    # We ablate at the *critical_pos* for the prediction, which is TARGET_POS_INDEX
    print("\n(Note: For neuron analysis, we look at the position *of* the prediction)")
    ABLATION_POS = TARGET_POS_INDEX
    
    main_neuron_indices = run_neuron_ablation(
        model,
        CLEAN_TEXT,
        clean_cache,
        critical_layer=max_layer,
        critical_pos=ABLATION_POS,
        baseline_clean_loss=baseline_clean_loss,
        k=20
    )
    
    # --- Step 5: Find Backup Neurons ---
    find_backup_neurons(
        model,
        CLEAN_TEXT,
        main_neuron_indices,
        critical_layer=max_layer,
        critical_pos=ABLATION_POS,
        k=20
    )
    
    # --- Step 6: Visualize Attention ---
    visualize_attention(
        clean_cache,
        CLEAN_TEXT,
        layer=max_layer # Visualize attention at the critical layer
    )
    
    # --- Step 7: Run Linear Probe ---
    # This is slow and just an example
    build_and_run_probe(
        model,
        critical_layer=max_layer
    )

if __name__ == "__main__":
    main()
