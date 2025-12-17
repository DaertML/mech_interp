import torch
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer

def load_model_and_test():
    """
    Loads a small GPT-2 model using HookedTransformer and runs a simple forward pass.
    
    If you were passing an argument like `token="hf_..."`, remove it, as it causes
    conflicts with the older, necessary version of the 'transformers' library.
    """
    try:
        print("Attempting to load model 'gpt2-small'...")

        # FIX: Ensure you are NOT passing 'token=...' here if you see the error.
        model: HookedTransformer = HookedTransformer.from_pretrained(
            "gpt2-small",
            # Set the device to CUDA if available, otherwise CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Model successfully loaded on device: {model.cfg.device}")
        print(f"Model name: {model.cfg.model_name}")

        # Simple test forward pass
        test_prompt = "The quick brown fox jumps over the lazy"
        tokens = model.to_tokens(test_prompt)
        logits = model(tokens)

        print("\nTest prompt:", test_prompt)
        print("Output logits shape:", logits.shape)
        print("Loading successful!")

    except Exception as e:
        print(f"\nAn error occurred during model loading or test run: {e}")
        print("Please verify that your 'HookedTransformer.from_pretrained' call does not include 'token='")

if __name__ == "__main__":
    load_model_and_test()

