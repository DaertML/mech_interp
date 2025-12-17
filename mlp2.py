import gradio as gr
import torch
import numpy as np
import math
from transformer_lens import HookedTransformer

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model globally once
try:
    print(f"Loading model: {MODEL_NAME} on device: {device}...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False
    )
    # Get model configuration details
    N_LAYERS = model.cfg.n_layers
    D_MLP = model.cfg.d_mlp
except Exception as e:
    print(f"FATAL ERROR: Failed to load model. Please ensure transformer_lens is installed. Error: {e}")
    N_LAYERS = 12
    D_MLP = 3072
    model = None
    
# --- CORE LOGIC: ACTIVATION & COLORING ---

def get_hsl_color(activation_value, max_abs_activation):
    """
    Generates an HSL color string for HTML background based on activation.
    Red (H=0) for positive, Blue (H=240) for negative.
    Lightness (L) is scaled by magnitude: lighter background for stronger activation.
    """
    if abs(activation_value) < 1e-6:
        return "" # Return empty string for near-zero activation

    # Normalize magnitude to [0, 1]
    normalized_magnitude = min(1.0, abs(activation_value) / max_abs_activation)
    
    # Define hue: Red (0) for positive, Blue (240) for negative
    hue = 0 if activation_value > 0 else 240
    saturation = 90 # High saturation
    
    # Define lightness: 95% (lightest) down to 65% (darkest/most intense)
    # Note: Lighter background corresponds to higher magnitude for visualization
    lightness = 95 - (normalized_magnitude * 30)
    
    return f"hsl({hue}, {saturation}%, {lightness}%)"


def analyze_neuron(neuron_id_str: str, text_content: str):
    """
    Calculates neuron activation and generates the color-coded HTML output.
    """
    if model is None:
        return "<p style='color:red;'>Model failed to load. Cannot run analysis.</p>", ""

    # 1. Parse Neuron ID
    try:
        layer_str, neuron_str = neuron_id_str.split(':')
        layer_id = int(layer_str)
        neuron_id = int(neuron_str)

        if not (0 <= layer_id < N_LAYERS):
            return f"<p style='color:red;'>Invalid Layer ID: {layer_id}. Must be between 0 and {N_LAYERS - 1}.</p>", ""
        if not (0 <= neuron_id < D_MLP):
            return f"<p style='color:red;'>Invalid Neuron ID: {neuron_id}. Must be between 0 and {D_MLP - 1} (for d_mlp={D_MLP}).</p>", ""

    except ValueError:
        return "<p style='color:red;'>Invalid Neuron ID format. Use 'Layer:Neuron' (e.g., 0:100).</p>", ""

    if not text_content.strip():
        return "<p style='color:red;'>Please provide text content to analyze.</p>", ""

    # 2. Run Model and Cache Activations
    try:
        # Use a hook to get the activation *before* the second weight matrix (W_out)
        # For TransformerLens, 'mlp_out' is the output *after* the GELU activation.
        # Shape: [1, seq_len, d_mlp] (batch dim removed)
        _, cache = model.run_with_cache(text_content, remove_batch_dim=True)
        mlp_activations = cache[f'mlp_out', layer_id].detach().cpu().numpy()
        str_tokens = model.to_str_tokens(text_content)
    except Exception as e:
        return f"<p style='color:red;'>Error during model run: {e}</p>", ""

    # 3. Extract specific neuron's activation
    # Shape: [seq_len]
    neuron_activations = mlp_activations[:, neuron_id]

    # 4. Find max absolute activation for normalization
    max_abs_activation = np.max(np.abs(neuron_activations))
    
    if max_abs_activation < 1e-6:
        # If activations are near zero, just return the plain text
        output_html = "".join(str_tokens)
        summary = "<p>Warning: All activations for this neuron are near zero. No coloring applied.</p>"
        return summary + f"<div class='token-container'>{output_html}</div>", summary

    # 5. Generate Color-Coded HTML
    colored_tokens = []
    for token, activation in zip(str_tokens, neuron_activations):
        # Escape HTML characters in the token
        safe_token = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        background_color = get_hsl_color(activation, max_abs_activation)
        
        # Use CSS for token styling
        style = f"background-color: {background_color}; border-radius: 4px; padding: 2px 4px; margin: 1px 0; display: inline-block; white-space: pre-wrap; line-height: 1.5; font-weight: 600;"
        
        # Add a title for exact activation value
        title = f"Activation: {activation:.4f}"
        
        # Use <pre> tag for tokens that are just spaces to preserve formatting
        if safe_token.isspace() or safe_token == '':
             # Tokenizer often produces tokens starting with spaces
             html_token = f'<span style="{style}" title="{title}">{safe_token}</span>'
        else:
             html_token = f'<span style="{style}" title="{title}">{safe_token}</span>'
        
        colored_tokens.append(html_token)

    output_html = "".join(colored_tokens)

    # 6. Generate Summary
    summary = f"""
    <div style="margin-top: 20px; padding: 10px; border-top: 1px solid #ddd; background-color: #f9f9f9; border-radius: 6px;">
        <p><strong>Model:</strong> {MODEL_NAME}</p>
        <p><strong>Neuron Target:</strong> L{layer_id}:N{neuron_id} (d_mlp={D_MLP})</p>
        <p><strong>Tokens Analyzed:</strong> {len(str_tokens)}</p>
        <p><strong>Max Absolute Activation:</strong> <span style="font-weight: bold; color: green;">{max_abs_activation:.4f}</span></p>
        <p style="margin-top: 10px; font-weight: bold;">
            <span style="color: hsl(0, 90%, 75%); padding: 0 4px;">RED</span>: Positive Activation | 
            <span style="color: hsl(240, 90%, 75%); padding: 0 4px;">BLUE</span>: Negative Activation
        </p>
    </div>
    """

    return summary, f"<div style='font-family: monospace; font-size: 16px; white-space: pre-wrap;'>{output_html}</div>"


# --- GRADIO INTERFACE SETUP ---

# Define Inputs
neuron_input = gr.Textbox(
    label=f"MLP Neuron ID (Layer:Neuron)",
    value="0:100",
    placeholder=f"e.g., 0:100 (Max Layer: {N_LAYERS - 1}, Max Neuron: {D_MLP - 1})"
)

text_input = gr.Textbox(
    label="Input Text (Content to Analyze)",
    value="Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious.",
    lines=5
)

# Define Outputs
summary_output = gr.HTML(label="Analysis Summary")
visualization_output = gr.HTML(label="Color-Coded Activation Visualization")


# Define the Interface
iface = gr.Interface(
    fn=analyze_neuron,
    inputs=[neuron_input, text_input],
    outputs=[summary_output, visualization_output],
    title="Real GPT-2 MLP Neuron Activation Visualizer",
    description="Enter a layer and neuron ID and text content. The application uses `transformer_lens` to calculate the actual activation of the specified MLP neuron for each token. Tokens are highlighted in Red (Positive) or Blue (Negative) with intensity based on magnitude. Note: Model loading may take a moment.",
    allow_flagging="never",
    examples=[
        ["0:100", "The movie theater was dimly lit and smelled of popcorn and old velvet."],
        ["1:500", "The capital of France is Paris and the Eiffel Tower is a landmark there."],
        ["5:2000", "I bought a carton of milk, a loaf of bread, and a dozen eggs from the store."]
    ]
)

# Launch the app (this line is commented out as it runs automatically in the environment)
iface.launch()
