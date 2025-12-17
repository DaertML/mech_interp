# 3 tokens: a value, b value, "="
# Model learns to move data from a, b tokens to =
# model doesnt move data from = to itself or any other places

#A <= K(a), K(b), Q(=)
#A predicts a value for a and * "x" to "=" and b value * "1-x" to "="
#= <= K(a)*x + K(b)*(1-x) and Q 

# neural nets are so good at linear algebra
# and they just wanna do lin_alg, so we add non
# linearities to avoid them only doing lin_alg XD

# If a certain token is always repeated at a certain
# pos; no need to add it as a matrix mult. Just multiply
# by the toke vectors. 

import torch
import numpy as np
import os
from transformer_lens import HookedTransformer, HookedTransformerConfig

# --- Configuration (MUST match the training script) ---
p = 113 # Modulus
artifacts_dir = "modular_addition_artifacts"
final_model_path = os.path.join(artifacts_dir, "final_trained_model.pt")

# Define the model configuration again to correctly initialize the structure
cfg = HookedTransformerConfig(
    n_layers = 1,
    n_heads = 4,
    d_model = 128,
    d_head = 32,
    d_mlp = 512,
    act_fn = "relu",
    normalization_type=None,
    d_vocab = p+1, # 114 tokens (0..112 for numbers, 113 for '=')
    d_vocab_out=p, # 113 results (0..112)
    n_ctx = 3,
    init_weights=True,
    device='cuda' # Ensure device matches training if running on GPU
)

def load_and_initialize_model():
    """Initializes the model and loads the trained weights."""
    if not os.path.exists(final_model_path):
        print(f"Error: Model file not found at {final_model_path}")
        print("Please ensure the training script ran successfully.")
        return None
    
    # 1. Initialize the model structure
    model = HookedTransformer(cfg).to(cfg.device)
    
    # 2. Disable biases (as done during training)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False

    # 3. Load the state dictionary
    print(f"Loading model weights from: {final_model_path}")
    state_dict = torch.load(final_model_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    print("Model loaded successfully.")
    return model

def run_inference(model, a: int, b: int) -> int:
    """
    Runs inference for modular addition: (a + b) % p
    a and b must be in the range [0, p-1].
    """
    if a < 0 or a >= p or b < 0 or b >= p:
        raise ValueError(f"Input numbers a and b must be between 0 and {p-1}.")

    # The input sequence is [a, b, '=']
    # The token for '=' is p (113 in this case)
    input_tokens = torch.tensor([[a, b, p]], device=model.cfg.device) 
    
    # Run the model without gradient tracking
    with torch.inference_mode():
        # model(input) returns logits of shape [batch_size, seq_len, d_vocab_out]
        logits = model(input_tokens)
    
    # We are interested in the prediction for the last token position (index 2)
    last_token_logits = logits[0, -1, :] # Shape [d_vocab_out]
    
    # Get the predicted output token (the result of the addition)
    prediction = last_token_logits.argmax().item()
    
    return prediction

def verify_inference(model):
    """Generates a few random inputs and checks the prediction against the ground truth."""
    print("\n--- Running Verification Tests ---")
    
    test_cases = [
        (10, 5),     # Easy case
        (p-1, 1),    # Wrap around case 1
        (p-1, p-1),  # Wrap around case 2 (large numbers)
        (50, 63)     # Mid-range case
    ]

    correct_count = 0
    
    for a, b in test_cases:
        ground_truth = (a + b) % p
        predicted = run_inference(model, a, b)
        
        is_correct = predicted == ground_truth
        print(f"Input: {a} + {b} = ? (mod {p}) | Ground Truth: {ground_truth} | Predicted: {predicted} | Correct: {is_correct}")
        if is_correct:
            correct_count += 1
            
    print(f"\nVerification complete. {correct_count}/{len(test_cases)} tests passed.")


if __name__ == "__main__":
    # Load the model
    loaded_model = load_and_initialize_model()

    if loaded_model:
        # Run the verification tests
        verify_inference(loaded_model)
        
        # Example of running a single prediction
        a_val, b_val = 100, 50
        try:
            result = run_inference(loaded_model, a_val, b_val)
            print(f"\nSingle Prediction: {a_val} + {b_val} (mod {p}) -> Model Predicts: {result}")
            print(f"Actual Result: {(a_val + b_val) % p}")
        except ValueError as e:
            print(e)