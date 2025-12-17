import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set a random seed for reproducible results
np.random.seed(42)

def generate_matrices(d_model, d_head):
    """Generates mock Q, K, V, and O weight matrices."""
    print("--- 1. Generating Mock Weights ---")
    # W_Q, W_K, W_V: d_model x d_head
    W_Q = np.random.randn(d_model, d_head) / np.sqrt(d_model)
    W_K = np.random.randn(d_model, d_head) / np.sqrt(d_model)
    W_V = np.random.randn(d_model, d_head) / np.sqrt(d_model)
    # W_O: d_head x d_model (output projection)
    W_O = np.random.randn(d_head, d_model) / np.sqrt(d_head)
    
    # Apply a structure to simulate positional focus (e.g., an Induction Head pattern):
    # We bias the matrices so W_Q @ X_i aligns with W_K @ X_{i-1} for a skip connection.
    # This simplified pattern encourages tokens to attend to the *previous* position.
    
    # W_Q: Enhance diagonal features related to position
    W_Q[:5, :5] += 2.0
    # W_K: Enhance features that should align with W_Q after a shift
    W_K[10:15, 10:15] += 1.5
    # W_V: Enhance features that carry the information to be copied
    W_V[20:25, 20:25] -= 1.0

    return W_Q, W_K, W_V, W_O

def visualize_matrices(matrices, titles):
    """Visualizes the weight matrices as heatmaps."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(18, 5))
    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes[i]
        sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=True, square=True,
                    xticklabels=False, yticklabels=False)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"d_head={matrix.shape[1]}", fontsize=10)
        ax.set_ylabel(f"d_model={matrix.shape[0]}", fontsize=10)
    
    fig.suptitle("Weight Matrix Heatmaps: Subspace Projection Structures", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_attention_weights(weights):
    """Visualizes the attention weight matrix, highlighting the QK focus."""
    context_length = weights.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Mask future tokens for a causal model simulation
    mask = np.triu(np.ones_like(weights, dtype=bool), k=1)
    weights_masked = np.ma.masked_where(mask, weights)
    
    sns.heatmap(weights_masked, ax=ax, cmap="Reds", cbar=True, square=True,
                linewidths=.5, linecolor='lightgray',
                xticklabels=[f"Key T{i}" for i in range(context_length)], 
                yticklabels=[f"Query T{i}" for i in range(context_length)])
    
    ax.set_title("QK Circuit Output: Attention Weights (Causal Masked)", fontsize=14)
    ax.set_xlabel("Attended-to Position (Key Index)", fontsize=10)
    ax.set_ylabel("Query Position (Token Index)", fontsize=10)
    
    # Highlight the "induction" diagonal (Token i attends to Token i-1)
    for i in range(1, context_length):
        ax.add_patch(plt.Rectangle((i-1, i), 1, 1, fill=False, edgecolor='blue', lw=3, alpha=0.8))
        
    plt.tight_layout()
    plt.show()

def visualize_ov_circuit_intermediates(V, head_output, weights, current_token_idx):
    """
    Visualizes the Value matrix (V) and the resulting Head Output for a given token.
    This shows what information is available to be copied (V) and the result of the OV aggregation.
    """
    context_length = V.shape[0]
    d_head = V.shape[1]

    # Get the attention weights for the current token (row i of the weights matrix)
    current_weights = weights[current_token_idx, :]
    
    # Identify the top 3 attended tokens (excluding masked/zero weight tokens)
    sorted_indices = np.argsort(current_weights)[::-1]
    
    # Filter for indices with weight > 1e-6 and are valid (i < current_token_idx)
    attended_indices = [i for i in sorted_indices if current_weights[i] > 1e-6 and i < current_token_idx] 
    
    top_attended = attended_indices[:3] # Focus on the top 3 contributors

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Plot 1: Value Matrix (V) ---
    ax1 = axes[0]
    
    sns.heatmap(V, ax=ax1, cmap="coolwarm", cbar=True, square=False,
                xticklabels=False, 
                yticklabels=[f"T{i}" for i in range(context_length)],
                vmin=V.min(), vmax=V.max())
    
    ax1.set_title(f"Value Matrix (V) - Information Source (d_head)", fontsize=14)

    # Highlight the row corresponding to the Query Token T_i 
    ax1.add_patch(plt.Rectangle((0, current_token_idx), d_head, 1, 
                                    fill=False, edgecolor='blue', lw=2, alpha=0.5, linestyle='--'))
    
    # Annotate/Highlight the contributing Value vectors
    for i, idx in enumerate(top_attended):
        weight_val = current_weights[idx]
        
        # Color the box based on contribution strength
        edge_color = 'green' if i == 0 else 'gray'
        
        # Draw box around the contributing row
        ax1.add_patch(plt.Rectangle((0, idx), d_head, 1, 
                                    fill=False, edgecolor=edge_color, lw=3, alpha=0.8))
                                    
        # Add annotation text (Attention Weight * V)
        ax1.text(d_head / 2, idx + 0.5, f"Weight: {weight_val:.2f}",
                 ha='center', va='center', color='white', fontsize=10, weight='bold')

    # Add legend annotations outside the heatmap
    ax1.text(d_head + 2, current_token_idx + 0.5, f"Query T{current_token_idx} (Blue Dashed)", va='center', color='blue', fontsize=10)
    if top_attended:
        ax1.text(d_head + 2, top_attended[0] + 0.5, f"Strongest Contribution (Green Box)", va='center', color='green', fontsize=10)
    
    ax1.set_xlabel(f"d_head dimensions ({d_head})", fontsize=10)
    ax1.set_ylabel("Token Position", fontsize=10)
    
    # --- Plot 2: Head Output (Weighted V) ---
    ax2 = axes[1]
    
    # This vector is the result of SUMMING (Weight_j * V_j) for all j tokens
    current_head_output = head_output[current_token_idx:current_token_idx+1, :]
    
    sns.heatmap(current_head_output, ax=ax2, cmap="magma", cbar=True, square=False,
                xticklabels=False, 
                yticklabels=[f"T{current_token_idx}"],
                vmin=head_output.min(), vmax=head_output.max())
    
    # Add an arrow/label to imply the summation process
    ax2.text(d_head / 2, -0.5, f"<- SUM (Weight_j * V_j) across all previous tokens",
             ha='center', va='center', color='black', fontsize=12, weight='bold')
    
    ax2.set_title(f"OV Circuit Result: Aggregated Vector for Query T{current_token_idx}", fontsize=14)
    ax2.set_xlabel(f"d_head dimensions ({d_head})", fontsize=10)
    ax2.set_ylabel("Output Token Position", fontsize=10)
    
    fig.suptitle("OV Circuit Flow: Aggregation of Value Vectors based on QK Weights", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def softmax(x):
    """Computes softmax along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def simulate_attention(X, W_Q, W_K, W_V, W_O):
    """Simulates a single attention head computation."""
    
    # 1. QK Circuit: Calculates query and key vectors and computes attention scores.
    # X (Context_Length x d_model) -> now includes token and position embeddings
    
    # Q (Context_Length x d_head): Query vector projection
    Q = X @ W_Q
    print(f"2. Q: X @ W_Q -> Shape: {Q.shape}")
    
    # K (Context_Length x d_head): Key vector projection
    K = X @ W_K
    print(f"3. K: X @ W_K -> Shape: {K.shape}")
    
    # Attention Scores (Context_Length x Context_Length)
    # Score = Q @ K.T / sqrt(d_head)
    scores = Q @ K.T
    d_head = Q.shape[1]
    scores = scores / np.sqrt(d_head)
    print(f"4. Scores: Q @ K.T / sqrt(d_head) -> Shape: {scores.shape}")
    
    # Apply Causal Mask (look only at previous tokens)
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
    scores[mask] = -np.inf # Set scores for future tokens to negative infinity
    
    # Attention Weights (Context_Length x Context_Length)
    weights = softmax(scores)
    print(f"5. Weights (Softmax + Causal Mask): Shape: {weights.shape}")
    
    # 2. OV Circuit: Calculates value vectors and aggregates information.
    
    # V (Context_Length x d_head): Value vector projection
    V = X @ W_V
    print(f"6. V: X @ W_V -> Shape: {V.shape}")
    
    # Weighted V (Context_Length x d_head): Aggregated information per token
    # Head Output = Attention Weights @ V
    head_output = weights @ V
    print(f"7. Head Output: Weights @ V -> Shape: {head_output.shape}")

    # 3. Output Projection
    # Z (Context_Length x d_model): Final output of the head (in model space)
    Z = head_output @ W_O
    print(f"8. Final Output Z: Head_Output @ W_O -> Shape: {Z.shape}")
    
    return Q, K, V, head_output, Z, scores, weights

def visualize_subspaces(X, Q, K, V, head_output, Z, current_token_idx):
    """
    Visualizes the movement of a single token vector through the different attention subspaces.
    Highlights the interaction between the Query of the current token (Q_i) and the
    Key and Value of the previous token (K_{i-1}, V_{i-1}) as part of the QK and OV circuits.
    
    Colors:
    - Blue: Input Token T_i
    - Orange: Query T_i (matching vector)
    - Green: Key T_{i-1} (target to match)
    - Orange/Yellow: Value T_{i-1} (information source)
    - Red: Aggregated Head Output (result of OV circuit)
    - Cyan: Final Model Output (Z)
    """
    
    if current_token_idx == 0:
        print("Cannot visualize QK interaction for T0 as it has no previous token. Using T1.")
        current_token_idx = 1
    
    x_vec = X[current_token_idx]
    q_vec = Q[current_token_idx]
    
    # T_{i-1} vectors, which are the source of information (Key and Value)
    k_vec_prev = K[current_token_idx - 1] 
    v_vec_prev = V[current_token_idx - 1]
    
    head_output_vec = head_output[current_token_idx]
    z_vec = Z[current_token_idx]
    
    # Collect the vectors for visualization (using the first 3 dimensions)
    vectors = {
        f"1. Input Vector (T{current_token_idx}) [Blue]": x_vec[:3],
        f"2. Query Subspace (T{current_token_idx}) [Orange]": q_vec[:3],
        f"3. Key Subspace (T{current_token_idx - 1}) [Green]": k_vec_prev[:3], # QK Interaction
        f"4. Value Subspace (T{current_token_idx - 1}) [Yellow/Orange]": v_vec_prev[:3], # OV Information Source
        "5. Aggregated Head Output (Weighted V) [Red]": head_output_vec[:3],
        "6. Final Model Output (Z) [Cyan]": z_vec[:3]
    }
    
    # Calculate QK dot product for context
    qk_dot_product = np.dot(q_vec, k_vec_prev)
    print(f"\nQK Interaction Focus: Dot Product (Query T{current_token_idx} vs Key T{current_token_idx - 1}) = {qk_dot_product:.2f}")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#f58231', '#d62728', '#17becf']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(f"Information Flow and Subspace Evolution (Focus on T{current_token_idx})", fontsize=14)
    ax.set_xlabel("Dimension 1", fontsize=10)
    ax.set_ylabel("Dimension 2", fontsize=10)
    ax.set_zlabel("Dimension 3", fontsize=10)

    # Plot vectors
    origin = [0, 0, 0]
    for i, (label, vec) in enumerate(vectors.items()):
        # Plot vector as an arrow
        ax.quiver(*origin, vec[0], vec[1], vec[2], 
                  color=colors[i], length=1.0, arrow_length_ratio=0.1, 
                  label=label, linewidth=2)
        
        # Plot the endpoint as a sphere (marker)
        ax.scatter(vec[0], vec[1], vec[2], color=colors[i], s=50)

    # Note on QK alignment
    ax.text2D(0.05, 0.95, f"QK Dot Product (T{current_token_idx} vs T{current_token_idx - 1}): {qk_dot_product:.2f}", 
              transform=ax.transAxes, color='black', fontsize=10)

    ax.legend(loc='best')
    ax.view_init(elev=20, azim=45) # Set initial view angle

    # Set limits based on all points to ensure everything is visible
    all_coords = np.array(list(vectors.values()))
    max_val = np.max(np.abs(all_coords)) * 1.2
    
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    
    # Define hyper-parameters
    D_MODEL = 32         # Dimensionality of the model's embeddings
    D_HEAD = 16          # Dimensionality of the attention head's internal space
    CONTEXT_LENGTH = 20  # Increased for better visualization of the attention matrix
    
    # Set the token index for focused visualizations (e.g., T5)
    TOKEN_FOCUS_IDX = 5 
    
    print(f"Configuration: d_model={D_MODEL}, d_head={D_HEAD}, Context_Length={CONTEXT_LENGTH}\n")
    
    # 1. Generate Input Embeddings (X)
    # Token Embeddings (content information)
    Token_Embeddings = np.random.randn(CONTEXT_LENGTH, D_MODEL)
    
    # Positional Embeddings (positional information) - Sinusoidal structure is common
    positions = np.arange(CONTEXT_LENGTH)[:, np.newaxis]
    i = np.arange(0, D_MODEL, 2)
    Position_Embeddings = np.zeros((CONTEXT_LENGTH, D_MODEL))
    Position_Embeddings[:, 0::2] = np.sin(positions / (10000**(i / D_MODEL)))
    Position_Embeddings[:, 1::2] = np.cos(positions / (10000**(i / D_MODEL)))
    
    # Combined Input Embedding (X = TE + PE)
    X = Token_Embeddings + Position_Embeddings
    
    print(f"0. Input Embeddings X (TE + PE): Shape: {X.shape}\n")
    
    # 2. Generate Weight Matrices
    W_Q, W_K, W_V, W_O = generate_matrices(D_MODEL, D_HEAD)
    
    # 3. Visualize Weight Matrices
    visualize_matrices(
        [W_Q, W_K, W_V, W_O.T], 
        ["W_Q (Query Matrix)", "W_K (Key Matrix)", "W_V (Value Matrix)", "W_O.T (Output Projection)"]
    )

    # 4. Simulate the Attention Mechanism
    Q, K, V, head_output, Z, scores, weights = simulate_attention(X, W_Q, W_K, W_V, W_O)
    
    # 5. Visualize QK Circuit Output (Attention Weights)
    visualize_attention_weights(weights)
    
    # 6. Visualize OV Circuit Intermediates: V and Weighted V
    print("\n--- 11. Visualizing OV Circuit Intermediates ---")
    visualize_ov_circuit_intermediates(V, head_output, weights, TOKEN_FOCUS_IDX)
    
    print("\n--- 9. QK Attention Scores (Focus Token) ---")
    print(f"Token {TOKEN_FOCUS_IDX} Attention Scores (to all previous keys):\n{scores[TOKEN_FOCUS_IDX]}")
    
    print("\n--- 10. Attention Weights (Focus Token) ---")
    print(f"Token {TOKEN_FOCUS_IDX} Attention Weights (to all previous keys, sums to 1):\n{weights[TOKEN_FOCUS_IDX]}")
    
    # 7. Visualize Subspace Evolution, highlighting the QK interaction
    print("\n--- 12. Visualizing Subspace Evolution ---")
    visualize_subspaces(X, Q, K, V, head_output, Z, TOKEN_FOCUS_IDX)

