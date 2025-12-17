import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class CharacterTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(f"Vocabulary built. Total unique characters (including special tokens): {self.vocab_size}")

    def encode(self, s):
        return [self.char_to_idx[char] for char in s]

    def decode(self, l):
        return ''.join([self.idx_to_char[idx] for idx in l])

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Causal mask for decoder-only transformer
        self.register_buffer("bias", torch.tril(torch.ones(max_seq_len, max_seq_len))
                                     .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension

        # Project query, key, value
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Scaled Dot-Product Attention
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)

        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        output = attn_weights @ v 
        
        output = output.transpose(1, 2).contiguous().view(B, T, C) # Reassemble heads
        
        output = self.o_proj(output)

        return output

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_seq_len
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False) # Language modeling head

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.eye_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Token embeddings
        x = self.token_embedding(idx) # (B, T, embed_dim)

        # Attention
        x = x + self.attn(x)

        # Language modeling head
        logits = self.fc(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape for CrossEntropyLoss: (N, C, ...) where C is num_classes
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # Crop input to max_seq_len if it exceeds
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            logits, _ = self(idx_cond)
            print(logits[:, -1, :].cpu().numpy())

            # Take logits of the last token
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        self.train()
        return tokenizer.decode(idx[0].tolist())


# --- Training Configuration ---
class Config:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 128 # Max sequence length for training
        self.epochs = 100
        self.learning_rate = 5e-2
        self.embed_dim = 6
        self.num_heads = 2
        self.eval_interval = 1
        self.max_grad_norm = 1.0
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.save_dir = 'model_checkpoints'
        self.model_name = 'decoder_only_transformer.pth'

# --- Training Function ---
def train_model(config, model, tokenizer, data):
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Prepare data
    data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # Create directories for saving model if they don't exist
    os.makedirs(config.save_dir, exist_ok=True)

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config.eval_interval) # Smaller number of batches for estimation
            for k in range(losses.shape[0]):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    print(f"Starting training on {config.device}...")
    model_save_path = os.path.join(config.save_dir, 'cp_000_'+config.model_name)
    print(f"Saving model checkpoint to {model_save_path}")   
    torch.save(model.state_dict(), model_save_path)

    for epoch in range(config.epochs):
        X, Y = get_batch('train')

        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm) # Gradient clipping
        optimizer.step()

        if epoch % config.eval_interval == 0:
            losses = estimate_loss()
            print(f"Epoch {epoch+1}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            
            # Save model parameters
            model_save_path = os.path.join(config.save_dir, 'cp_'+str(epoch+1).zfill(3)+'_'+config.model_name)
            print(f"Saving model checkpoint to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':

    text_data = "xxxxxbacxxxxxdae"*200
    
    # 1. Tokenize
    tokenizer = CharacterTokenizer(text_data)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample encode/decode: '{text_data[:20]}' -> {tokenizer.encode(text_data[:20])} -> '{tokenizer.decode(tokenizer.encode(text_data[:20]))}'")

    # 2. Initialize Config and Model
    config = Config()
    config.vocab_size = tokenizer.vocab_size # Update vocab size based on data
    config.max_seq_len = config.block_size # Max sequence length for RoPE and causal mask

    model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len
    )

    # --- Counting all parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (trainable + frozen): {total_params:,}")

    # --- Counting ONLY trainable parameters ---
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # 3. Train the model
    train_model(config, model, tokenizer, text_data)

    # 4. Load and Test the trained model
    print("\n--- Testing Trained Model ---")
    
    # Instantiate a new model instance
    loaded_model = DecoderOnlyTransformer(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len
    )
    
    # Load the saved state_dict
    model_save_path = os.path.join(config.save_dir, 'cp_'+str(config.epochs).zfill(3)+'_'+config.model_name)
    loaded_model.load_state_dict(torch.load(model_save_path, map_location=config.device))
    loaded_model.to(config.device)

    prompt = "ba"
    generated_text = loaded_model.generate(tokenizer, prompt, max_new_tokens=10, temperature=0.1, top_k=3)
    print(f"Generated: \"{generated_text}\"")

    prompt = "da"
    generated_text = loaded_model.generate(tokenizer, prompt, max_new_tokens=10, temperature=0.1, top_k=3)
    print(f"Generated: \"{generated_text}\"")
