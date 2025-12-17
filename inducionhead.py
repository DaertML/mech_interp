import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# --- 1. Model Definition (Same as before, focuses on 2 attention heads) ---

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding used in the original Transformer paper.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, Max_Len, D_Model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, D_Model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TwoHeadAttentionCircuit(nn.Module):
    """
    A simple autoregressive model using a single block with two attention heads.
    """
    def __init__(self, vocab_size, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert num_heads == 2, "This circuit is hardcoded for exactly 2 heads."
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Multi-Head Self-Attention Layer
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Final Output Layer
        self.linear_out = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        """Creates a square, causal mask for self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device

        # Input Processing (Token + Positional Embedding)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Generate Causal Mask
        attn_mask = self.generate_causal_mask(seq_len, device)

        # Self-Attention Block
        attn_output, _ = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            is_causal=False
        )
        x = self.norm1(x + attn_output) # Add & Norm

        # Feed-Forward Block
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output) # Add & Norm

        # Output
        logits = self.linear_out(x)
        return logits


# --- 2. Induction Data Generation ---

def generate_induction_data(batch_size, seq_len, vocab_size, device, prefix_len=4, pattern_len=2):
    """
    Generates synthetic data designed to encourage induction head behavior.
    Sequences follow the structure: [Random Prefix | Pattern | Pattern]
    E.g., SEQ_LEN=10, Prefix=4, Pattern=3: [R1, R2, R3, R4, A, B, C, A, B, C]
    The model should predict 'A' after the first 'C', 'B' after the second 'A',
    and 'C' after the second 'B'.

    We use the simplest form: [R1..R_p, A, B, A, B]
    """
    data = []
    
    # Calculate required padding for the pattern
    required_len_for_pattern = 2 * pattern_len
    if seq_len < prefix_len + required_len_for_pattern:
        raise ValueError("Sequence length is too short to contain prefix and two patterns.")

    pattern_start_index = prefix_len
    
    for _ in range(batch_size):
        sequence = torch.zeros(seq_len, dtype=torch.long)
        
        # 1. Generate Random Prefix (R...R)
        sequence[:prefix_len] = torch.randint(
            low=0, 
            high=vocab_size, 
            size=(prefix_len,)
        )
        
        # 2. Generate a random bigram Pattern (A, B)
        # We ensure the pattern tokens are *different* from the padding token (0) and 
        # potentially different from the prefix tokens, although here we just use high=vocab_size
        pattern = torch.randint(
            low=1, # Ensure pattern tokens are > 0 to stand out from potential padding if any
            high=vocab_size, 
            size=(pattern_len,)
        )
        
        # 3. Insert Pattern A, B (P1)
        sequence[pattern_start_index : pattern_start_index + pattern_len] = pattern

        # 4. Insert Pattern A, B again (P2)
        sequence[pattern_start_index + pattern_len : pattern_start_index + 2 * pattern_len] = pattern
        
        # 5. Fill remaining space (if any) with random tokens
        remaining_len = seq_len - (prefix_len + 2 * pattern_len)
        if remaining_len > 0:
            # We want the induction to happen right after the second pattern, so fill the rest randomly
            sequence[prefix_len + 2 * pattern_len:] = torch.randint(
                low=1, 
                high=vocab_size, 
                size=(remaining_len,)
            )

        data.append(sequence)

    return torch.stack(data).to(device)


# --- 3. Training Function ---

def train_autoregressive_circuit():
    # Configuration
    VOCAB_SIZE = 16    # Slightly larger vocab to make pattern tokens unique
    D_MODEL = 64
    NUM_HEADS = 2
    BATCH_SIZE = 64    # Larger batch size helps stabilize learning
    SEQ_LEN = 12       # Sequence length: e.g., 4 prefix + 4 pattern + 4 random
    NUM_EPOCHS = 2500   # Increased epochs to ensure convergence on the pattern
    LEARNING_RATE = 1e-3
    
    # Specific Induction Data Parameters
    PREFIX_LEN = 4
    PATTERN_LEN = 2 # The pattern is always a bigram (A, B)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Goal: Train the model to perform induction on [R..R, A, B, A, B, ...]")

    # Initialize Model
    model = TwoHeadAttentionCircuit(VOCAB_SIZE, D_MODEL, NUM_HEADS).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        # Generate new data for every epoch to ensure generalization of the pattern logic
        # and not just memorizing one specific sequence.
        induction_data = generate_induction_data(
            BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, device, 
            prefix_len=PREFIX_LEN, pattern_len=PATTERN_LEN
        )

        optimizer.zero_grad()

        # Get logits
        logits = model(induction_data) # (B, S, V)

        # Autoregressive loss calculation: predict token i+1 from input token i
        # Target sequence is the input sequence shifted left by one
        targets = induction_data[:, 1:].reshape(-1)
        
        # Logits for the tokens 0 to S-2 (used to predict 1 to S-1)
        predictions = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    print("\nTraining finished.")

    # --- 4. Validation and Analysis ---
    model.eval()
    print("\n--- Validation: Testing Induction ---")
    
    # Create a specific test sequence: [R, R, R, R, 10, 11, 10, 11, 12, 12, 12, 12]
    # Prediction target is '11' after the second '10'.
    # Sequence length is 12.
    # Token 10 is at index 4 and 6. Token 11 is at index 5 and 7.
    
    # Sequence to test:
    test_seq_list = [1, 2, 3, 4] # Prefix
    test_seq_list += [10, 11]    # First Pattern (A=10, B=11)
    test_seq_list += [10, 0]     # Second Pattern start: A=10, ?=0 (Model must predict 11)
    test_seq_list += [0] * (SEQ_LEN - len(test_seq_list)) # Padding the rest
    
    # The crucial position is index 7 (the '0' after the second '10'). 
    # The input to the model will be up to index 6 (the second '10').
    test_seq = torch.tensor([test_seq_list]).to(device)

    print(f"Test Input (A=10, B=11): {test_seq_list[:7]}...")
    
    with torch.no_grad():
        logits = model(test_seq)
        
        # We check the prediction at position 7 (index 7), which is based on the input up to position 6 (index 6, the second '10').
        # Input: [1, 2, 3, 4, 10, 11, 10]
        # Prediction index: 7 (should be 11)
        
        prediction_idx = 7 # The index of the token we want to predict
        predicted_token_logits = logits[0, prediction_idx - 1, :] # Logits from the position *before* the prediction
        
        predicted_token_index = torch.argmax(predicted_token_logits, dim=-1).item()
        
        # Check the probability for the correct token (11)
        probabilities = F.softmax(predicted_token_logits, dim=-1)
        correct_prob = probabilities[11].item() # Check prob for token 11

    print(f"Prediction made at sequence index: {prediction_idx} (after the second '10')")
    print(f"Predicted next token: {predicted_token_index}")
    print(f"Confidence for token 11 (the correct token): {correct_prob*100:.2f}%")
    
    if predicted_token_index == 11 and correct_prob > 0.5:
        print("\nSuccess: The model appears to have learned the induction pattern!")
    else:
        print("\nNote: The model did not strongly predict the induction token (11). More training or a different architecture might be needed.")


if __name__ == "__main__":
    # Ensure reproducibility for testing
    torch.manual_seed(42)
    train_autoregressive_circuit()
