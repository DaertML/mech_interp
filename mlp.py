import torch
import numpy as np
from transformer_lens import HookedTransformer

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SIMULATED USER INPUT (Modify these to change the analysis) ---
# Format: layer_id:neuron_id. Layer 0 has 3072 MLP neurons (0 to 3071).
NEURON_TO_STUDY = "0:100"

# The text content to analyze.
CONTENT_TEXT = """
Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck.
"""
# -----------------------------------------------------------------

# --- UTILITY: ANSI COLORING ---

def color_activation(token, activation_value, max_abs_activation, color_steps=8):
    """
    Applies ANSI escape codes to color the token based on activation value.
    Red for positive, Blue for negative. Intensity is based on magnitude.
    """
    if activation_value == 0:
        return token

    # Normalize the activation value to a range [0, 1] for intensity
    normalized_magnitude = abs(activation_value) / max_abs_activation
    
    # Map the normalized magnitude to a specific ANSI color level.
    # We use a limited number of steps for noticeable gradient effect.
    intensity = int(normalized_magnitude * (color_steps - 1)) + 1
    
    # ANSI color codes for 256 color palette (16-231).
    # We choose a range for red (196-202) and blue (21-27)
    if activation_value > 0:
        # Red spectrum (from darker to brighter red)
        base_color = 196  # Darkest red
        color_code = base_color + (intensity * 2)
        # Ensure code stays within a red-ish range
        color_code = min(max(color_code, 196), 202)
    else:
        # Blue spectrum (from darker to brighter blue)
        base_color = 21   # Darkest blue
        color_code = base_color + (intensity * 2)
        # Ensure code stays within a blue-ish range
        color_code = min(max(color_code, 21), 27)

    # ANSI format: \x1b[38;5;<color_code>m<text>\x1b[0m
    return f"\x1b[38;5;{color_code}m{token}\x1b[0m"


def analyze_mlp_neuron(model, text, layer_id, neuron_id):
    """
    Runs the model, extracts neuron activation, and prints the color-coded text.
    """
    print(f"\n--- Analyzing MLP Neuron L{layer_id} N{neuron_id} ---")

    # 1. Run model and cache activations
    try:
        # The 'mlp_out' cache stores the output of the MLP layer (after the GELU).
        # Shape: [seq_len, d_mlp]
        _, cache = model.run_with_cache(text, remove_batch_dim=True)
        mlp_activations = cache[f'mlp_out', layer_id].detach().cpu().numpy()
        str_tokens = model.to_str_tokens(text)
    except Exception as e:
        print(f"Error during model run or cache extraction: {e}")
        return

    # Check if the neuron ID is valid for the model configuration
    d_mlp = mlp_activations.shape[1]
    if neuron_id >= d_mlp:
        print(f"Error: Neuron ID {neuron_id} is out of bounds for Layer {layer_id} (d_mlp={d_mlp}).")
        return

    # 2. Extract specific neuron's activation across the sequence
    # Shape: [seq_len]
    neuron_activations = mlp_activations[:, neuron_id]

    # 3. Find max absolute activation for normalization
    # This helps scale the color intensity relative to the strongest activation.
    max_abs_activation = np.max(np.abs(neuron_activations))
    
    if max_abs_activation < 1e-6:
        print("Warning: All activations for this neuron are near zero. No coloring applied.")
        # Fallback print if all activations are too small
        print("".join(str_tokens))
        return

    # 4. Color-code and print the tokens
    colored_output = []
    
    for token, activation in zip(str_tokens, neuron_activations):
        colored_token = color_activation(token, activation, max_abs_activation)
        colored_output.append(colored_token)

    print(f"Max Absolute Activation: {max_abs_activation:.4f}")
    print("\n" + "="*80)
    print("Color Map: RED (Positive Activation) | BLUE (Negative Activation)")
    print("Intensity: DARKER (Low Magnitude) -> LIGHTER (High Magnitude)")
    print("="*80)
    
    # Join the colored tokens into a single string for display
    print("".join(colored_output))
    print("\n" + "="*80)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Loading model: {MODEL_NAME} on device: {device}...")
    
    # 1. Model Loading
    try:
        model = HookedTransformer.from_pretrained(
            MODEL_NAME,
            device=device,
            # Suppress default print messages during loading
            fold_ln=False, 
            center_writing_weights=False, 
            center_unembed=False
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please ensure transformer_lens is correctly installed.")
        exit()
    
    # 2. Parse User Input
    try:
        layer_str, neuron_str = NEURON_TO_STUDY.split(':')
        target_layer = int(layer_str)
        target_neuron = int(neuron_str)

        if target_layer >= model.cfg.n_layers or target_layer < 0:
            raise ValueError(f"Layer ID {target_layer} is out of model bounds (0 to {model.cfg.n_layers - 1}).")

    except ValueError as e:
        print(f"Invalid NEURON_TO_STUDY format: {e}. Please use 'layer_id:neuron_id' (e.g., '0:100').")
        exit()

    # 3. Perform Analysis
    analyze_mlp_neuron(model, CONTENT_TEXT, target_layer, target_neuron)

