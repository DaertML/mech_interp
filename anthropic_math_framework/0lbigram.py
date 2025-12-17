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
        print(f"Vocabulary built. Total unique characters: {self.vocab_size}")

    def encode(self, s):
        return [self.char_to_idx[char] for char in s]

    def decode(self, l):
        return ''.join([self.idx_to_char[idx] for idx in l])

class BigramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_seq_len

        # Set embeddings to one-hot
        w = torch.eye(vocab_size, embed_dim)
        print(f"shape of pre-trained embeddings: {w.shape}")
        self.token_embedding = nn.Embedding.from_pretrained(w, freeze=True)

        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # Token embeddings
        x = self.token_embedding(idx) # (B, T, embed_dim)

        # Fully connected layer
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
        self.block_size = 200 # Max sequence length for training
        self.epochs = 10
        self.learning_rate = 1e-2
        self.embed_dim = 4
        self.eval_interval = 1
        self.max_grad_norm = 1.0
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.save_dir = 'model_checkpoints'
        self.model_name = 'bigram.pth'


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
        ix = torch.arange(0,len(data)-config.block_size,config.block_size//20)
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()

        losses = torch.zeros(config.eval_interval)
        for k in range(losses.shape[0]):
            X, Y = get_batch('val')
            _, loss = model(X, Y)
            losses[k] = loss.item()
        
        out = losses.mean()
        
        model.train()
        return out

    print(f"Starting training on {config.device}...")
    for epoch in range(config.epochs):
        X, Y = get_batch('train')

        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm) # Gradient clipping
        optimizer.step()

        if epoch % config.eval_interval == 0:
            eval_loss = estimate_loss()
            print(f"Epoch {epoch+1}: Train Loss {loss.item():.4f}, Val Loss {eval_loss:.4f}")
            
            # Save model parameters
            model_save_path = os.path.join(config.save_dir, 'cp_'+str(epoch).zfill(3)+'_'+config.model_name)
            print(f"Saving model checkpoint to {model_save_path}")
            torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    text_data = 'abcd'*1000

    # 1. Toeknize
    tokenizer = CharacterTokenizer(text_data)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sample encode/decode: '{text_data[:20]}' -> {tokenizer.encode(text_data[:20])} -> '{tokenizer.decode(tokenizer.encode(text_data[:20]))}'")

    # 2. Initialize Config and Model
    config = Config()
    config.vocab_size = tokenizer.vocab_size # Update vocab size based on data
    config.max_seq_len = config.block_size # Max sequence length for RoPE and causal mask

    model = BigramModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
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
    loaded_model = BigramModel(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_seq_len=config.max_seq_len
    )
    
    # Load the saved state_dict
    model_save_path = os.path.join(config.save_dir, 'cp_'+str(config.epochs-1).zfill(3)+'_'+config.model_name)
    if os.path.exists(model_save_path):
        loaded_model.load_state_dict(torch.load(model_save_path, map_location=config.device))
        loaded_model.to(config.device)
        print(f"Model loaded successfully from {model_save_path}")

        prompt = "c"
        print(f"Prompt: \"{prompt}\"")
        generated_text = loaded_model.generate(tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=3)
        print(f"Generated: \"{generated_text}\"")
    else:
        print(f"No saved model found at {model_save_path}. Please train the model first.")
