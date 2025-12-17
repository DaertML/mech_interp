from huggingface_hub import hf_hub_download
from IPython.display import display

# ... (rest of the imports and setup remain the same)
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import circuitsvis as cv
import torch

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"
device = "cuda"
weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)

model = HookedTransformer(cfg)
pretrained_weights = torch.load(weights_path, map_location=device)
model.load_state_dict(pretrained_weights)

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
attention_pattern = cache["pattern", 0, "attn"]

str_tokens = model.to_str_tokens(text)

# FIX: Changed cv.attention.from_cache to cv.attention.attention_patterns_from_cache
display(cv.attention.attention_patterns(
                            tokens=str_tokens,
                            attention=attention_pattern))
