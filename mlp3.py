import gradio as gr
import torch
import numpy as np
import math
import json
from transformer_lens import HookedTransformer
import aiohttp # <-- NEW: Using aiohttp for asynchronous HTTP requests

# --- CONFIGURATION ---
MODEL_NAME = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ollama API configuration (used for the prediction task)
# NOTE: This endpoint is a placeholder. To make this work locally,
# you must have an Ollama server running on port 11434 with 'llama3' pulled.
OLLAMA_MODEL = "llama3" 
OLLAMA_API_URL = "http://localhost:11434/api/generate" 

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

# --- OLLAMA INTEGRATION ---

def create_llm_data_string(str_tokens, neuron_activations):
    """
    Creates a structured string of tokens and activations for the LLM to analyze.
    """
    data_points = []
    for token, activation in zip(str_tokens, neuron_activations):
        # Escape newlines and format activation
        clean_token = token.replace('\n', '\\n').replace('\r', '\\r')
        data_points.append(f"TOKEN: '{clean_token}' | ACTIVATION: {activation:.4f}")
    
    return "\n".join(data_points)

async def call_llm_for_prediction(data_string, user_prompt):
    """
    Calls the Ollama LLM to predict neuron activity based on the activation data.
    Uses aiohttp for the asynchronous POST request.
    """
    
    system_instruction = (
        "You are an expert neuroscience analyst for Large Language Models. "
        "Your task is to analyze the provided list of tokens and their corresponding "
        "activation values for a single MLP neuron. A high positive activation "
        "(>0) means the neuron is 'firing' on that token. A high negative activation "
        "(<0) means the neuron is suppressed. "
        "Based on the pattern of high/low activation across the sequence, try to describe "
        "the linguistic function or concept this neuron might be detecting. "
        "Provide a concise summary and a clear hypothesis."
    )
    
    full_prompt = (
        f"--- NEURON ACTIVATION DATA ---\n"
        f"The following shows the token and its activation value (GELU output) for an MLP neuron across a text sequence. Analyze this data:\n"
        f"{data_string}\n"
        f"-------------------------------\n"
        f"USER REQUEST: {user_prompt}"
    )

    # Ollama Payload Structure
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_instruction}\n\n{full_prompt}",
        "stream": False, # Request non-streaming response
        "options": {
            "temperature": 0.1
        }
    }

    try:
        # NEW: Using aiohttp.ClientSession for the asynchronous request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OLLAMA_API_URL, 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                
                # Check for successful HTTP status
                if response.status != 200:
                    error_text = await response.text()
                    return (f"OLLAMA HTTP ERROR: Server responded with status {response.status}. "
                            f"Response body snippet: {error_text[:200]}")
                
                result = await response.json()
                
                # Ollama's non-streaming response has the text in the 'response' field.
                text = result.get('response', "Ollama response not found or failed to parse.")
                
                return text

    except aiohttp.ClientConnectorError as e:
        # Handle connection-specific errors (server not running or connection refused)
        return (f"OLLAMA CONNECTION ERROR: Could not connect to the Ollama server at {OLLAMA_API_URL}. "
                f"Please ensure Ollama is running locally and the model '{OLLAMA_MODEL}' is available. "
                f"Detailed error: {e}")
    except Exception as e:
        # Handle other general exceptions (JSON parsing, etc.)
        return f"OLLAMA GENERAL ERROR: An unexpected error occurred. Detailed error: {e}"


# --- GRADIO WRAPPER FUNCTION ---

async def analyze_and_predict_neuron(neuron_id_str: str, text_content: str, llm_prompt: str):
    """
    Main function to analyze neuron activation and call the LLM for prediction.
    """
    if model is None:
        return "<p style='color:red;'>Model failed to load. Cannot run analysis.</p>", "", "<p style='color:red;'>Prediction failed.</p>"

    # 1. Input Validation
    try:
        layer_str, neuron_str = neuron_id_str.split(':')
        layer_id = int(layer_str)
        neuron_id = int(neuron_str)

        if not (0 <= layer_id < N_LAYERS):
            return f"<p style='color:red;'>Invalid Layer ID: {layer_id}. Must be between 0 and {N_LAYERS - 1}.</p>", "", ""
        if not (0 <= neuron_id < D_MLP):
            return f"<p style='color:red;'>Invalid Neuron ID: {neuron_id}. Must be between 0 and {D_MLP - 1} (for d_mlp={D_MLP}).</p>", "", ""

    except ValueError:
        return "<p style='color:red;'>Invalid Neuron ID format. Use 'Layer:Neuron' (e.g., 0:100).</p>", "", ""

    if not text_content.strip():
        return "<p style='color:red;'>Please provide text content to analyze.</p>", "", ""

    # 2. Run Model and Cache Activations
    try:
        _, cache = model.run_with_cache(text_content, remove_batch_dim=True)
        mlp_activations = cache[f'mlp_out', layer_id].detach().cpu().numpy()
        str_tokens = model.to_str_tokens(text_content)
    except Exception as e:
        return f"<p style='color:red;'>Error during model run: {e}</p>", "", ""

    # 3. Extract specific neuron's activation
    neuron_activations = mlp_activations[:, neuron_id]

    # 4. Find max absolute activation for normalization
    max_abs_activation = np.max(np.abs(neuron_activations))
    
    # Generate LLM data string early for LLM call
    data_string = create_llm_data_string(str_tokens, neuron_activations)
    
    # 5. Generate Activation Visualization HTML
    colored_tokens = []
    if max_abs_activation < 1e-6:
        output_html = "".join(str_tokens)
        summary = "<p>Warning: All activations for this neuron are near zero. No coloring applied.</p>"
    else:
        # Generate color-coded tokens
        for token, activation in zip(str_tokens, neuron_activations):
            safe_token = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            background_color = get_hsl_color(activation, max_abs_activation)
            style = f"background-color: {background_color}; border-radius: 4px; padding: 2px 4px; margin: 1px 0; display: inline-block; white-space: pre-wrap; line-height: 1.5; font-weight: 600;"
            title = f"Activation: {activation:.4f}"
            html_token = f'<span style="{style}" title="{title}">{safe_token}</span>'
            colored_tokens.append(html_token)
        
        output_html = "".join(colored_tokens)
        
        # Generate summary
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

    # 6. Call LLM for Prediction (Ollama)
    llm_prediction = await call_llm_for_prediction(data_string, llm_prompt)
    
    llm_html = f"""
    <div style="padding: 15px; border: 1px solid #007bff; border-radius: 8px; background-color: #e9f7ff;">
        <p style="font-weight: bold; color: #007bff; margin-bottom: 10px;">LLM Interpretation (Ollama - {OLLAMA_MODEL}):</p>
        <p>{llm_prediction}</p>
    </div>
    """

    return summary, f"<div style='font-family: monospace; font-size: 16px; white-space: pre-wrap;'>{output_html}</div>", llm_html


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

llm_prompt_input = gr.Textbox(
    label="LLM Prediction Prompt (Ollama)",
    value="Based on the highlighted text, what linguistic function or type of concept is this neuron firing on? Give a one-paragraph hypothesis.",
    placeholder="e.g., 'Is this neuron detecting names, verbs, or sentiment?'",
    lines=2
)


# Define Outputs
summary_output = gr.HTML(label="Analysis Summary")
visualization_output = gr.HTML(label="Color-Coded Activation Visualization")
llm_output = gr.HTML(label="Neuron Function Prediction")


# Define the Interface
iface = gr.Interface(
    fn=analyze_and_predict_neuron,
    inputs=[neuron_input, text_input, llm_prompt_input],
    outputs=[summary_output, visualization_output, llm_output],
    title="GPT-2 Neuron Activation & Ollama Interpretation Tool",
    description=f"Visualize real GPT-2 MLP neuron activations and send the results to an Ollama instance for functional hypothesis generation. **Ollama URL: {OLLAMA_API_URL}** (Must be running locally). Tokens are colored by activation strength (Red=Positive, Blue=Negative).",
    allow_flagging="never",
    examples=[
        ["0:100", "The movie theater was dimly lit and smelled of popcorn and old velvet. The attendant checked our tickets.", "What is this neuron attending to right before punctuation?"],
        ["1:500", "The capital of France is Paris and the Eiffel Tower is a landmark there.", "Does this neuron detect geographical entities or facts?"],
        ["5:2000", "I bought a carton of milk, a loaf of bread, and a dozen eggs from the store.", "Does this neuron activate for items in a list or common nouns?"],
    ]
)
iface.launch()

