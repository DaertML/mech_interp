import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set a random seed for reproducible results
np.random.seed(42)

def generate_matrices(d_model, d_head):
    """Generates mock Q, K, V, and O weight matrices with simulated structure."""
    print("--- 1. Generating Mock Weights ---")
    # W_Q, W_K, W_V: d_model x d_head
    W_Q = np.random.randn(d_model, d_head) / np.sqrt(d_model)
    W_K = np.random.randn(d_model, d_head) / np.sqrt(d_model)
    W_V = np.random.randn(d_model, d_head) / np.sqrt(d_model)
    # W_O: d_head x d_model (output projection)
    W_O = np.random.randn(d_head, d_model) / np.sqrt(d_head)
    
    # Introduce a mock structure to simulate a useful circuit (like an Induction Head)
    # This biases the system to find positional alignment (T_i attends to T_{i-1})
    W_Q[:5, :5] += 2.0  # Encourage specific features to become the Query signal
    W_K[10:15, 10:15] += 1.5 # Encourage Key vectors to align with the Query signal
    W_V[20:25, 20:25] -= 1.0 # Bias the Value vector to contain specific information
    
    return W_Q, W_K, W_V, W_O

def visualize_matrices(matrices, titles):
    """Visualizes the weight matrices as heatmaps (Plot 1)."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(18, 5))
    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        ax = axes[i]
        sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=True, square=True,
                    xticklabels=False, yticklabels=False)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"d_head={matrix.shape[1]}", fontsize=10)
        ax.set_ylabel(f"d_model={matrix.shape[0]}", fontsize=10)
    
    fig.suptitle("Plot 1: Weight Matrix Heatmaps (Projection Subspaces)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_attention_weights(weights):
    """Visualizes the attention weight matrix (QK Circuit Output) (Plot 2)."""
    context_length = weights.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Mask future tokens for a causal model simulation
    mask = np.triu(np.ones_like(weights, dtype=bool), k=1)
    weights_masked = np.ma.masked_where(mask, weights)
    
    sns.heatmap(weights_masked, ax=ax, cmap="Reds", cbar=True, square=True,
                linewidths=.5, linecolor='lightgray',
                xticklabels=[f"Key T{i}" for i in range(context_length)], 
                yticklabels=[f"Query T{i}" for i in range(context_length)])
    
    ax.set_title("Plot 2: QK Circuit Output - Attention Weights (Causal Masked)", fontsize=14)
    ax.set_xlabel("Attended-to Position (Key Index)", fontsize=10)
    ax.set_ylabel("Query Position (Token Index)", fontsize=10)
    
    # Highlight the "induction" diagonal (Token i attends to Token i-1)
    for i in range(1, context_length):
        ax.add_patch(plt.Rectangle((i-1, i), 1, 1, fill=False, edgecolor='blue', lw=3, alpha=0.8))
        
    plt.tight_layout()
    plt.show()

def visualize_ov_circuit_intermediates(V, head_output, weights, current_token_idx):
    """
    Visualizes the OV Circuit Flow (Plot 3): QK Weight -> V Value -> Head Output.
    """
    context_length = V.shape[0]
    d_head = V.shape[1]

    # Get the attention weights for the current token (row i of the weights matrix)
    current_weights = weights[current_token_idx, :]
    
    # Identify the strongest attended token (T_{i-1} is expected here)
    attended_indices = [i for i in np.argsort(current_weights)[::-1] if current_weights[i] > 1e-6 and i < current_token_idx] 
    
    top_attended_idx = attended_indices[0] if attended_indices else None
    
    # Use a figure with 3 columns, where the middle one is just for the flow arrow
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), gridspec_kw={'width_ratios': [1, 0.2, 1]})
    ax1, ax_flow, ax2 = axes
    
    # --- Plot 1: Value Matrix (V) - The Information Source ---
    sns.heatmap(V, ax=ax1, cmap="coolwarm", cbar=True, square=False,
                xticklabels=False, 
                yticklabels=[f"T{i}" for i in range(context_length)],
                vmin=V.min(), vmax=V.max())
    
    ax1.set_title(f"3a. Value Matrix (V) - Information Source (d_head)", fontsize=14)
    ax1.set_xlabel(f"d_head dimensions ({d_head})", fontsize=10)
    ax1.set_ylabel("Token Position", fontsize=10)

    if top_attended_idx is not None:
        weight_val = current_weights[top_attended_idx]
        
        # Highlight the strongest contributing row (Value vector being copied)
        ax1.add_patch(plt.Rectangle((0, top_attended_idx), d_head, 1, 
                                    fill=False, edgecolor='green', lw=3, alpha=0.9))
                                    
        # Annotate the QK Weight next to the V vector
        ax1.text(d_head + 0.5, top_attended_idx + 0.5, 
                 f"Weight={weight_val:.2f} (from QK match)\nSource T{top_attended_idx} V", 
                 va='center', color='green', fontsize=12, weight='bold')

    # --- Plot 2: Head Output (Weighted V) - The Information Result ---
    # This vector is the result of SUMMING (Weight_j * V_j) for all j tokens
    current_head_output = head_output[current_token_idx:current_token_idx+1, :]
    
    sns.heatmap(current_head_output, ax=ax2, cmap="magma", cbar=True, square=False,
                xticklabels=False, 
                yticklabels=[f"T{current_token_idx}"],
                vmin=head_output.min(), vmax=head_output.max())
    
    ax2.set_title(f"3c. Aggregated Head Output Vector for Query T{current_token_idx}", fontsize=14)
    ax2.set_xlabel(f"d_head dimensions ({d_head})", fontsize=10)
    ax2.set_ylabel("Output Token Position", fontsize=10)

    # --- Plot 3: Flow Arrow and Operation (in the middle axis) ---
    ax_flow.axis('off') # Hide axis for flow diagram
    
    # Draw a conceptual arrow representing the linear combination flow
    ax_flow.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                     arrowprops={'arrowstyle': '->', 'lw': 4, 'color': 'red'},
                     xycoords='axes fraction', textcoords='axes fraction')
    
    # Label the operation
    ax_flow.text(0.5, 0.6, r'$\mathbf{HeadOutput}_i = \sum_{j=0}^{i-1} \mathbf{Weight}_{i,j} \cdot \mathbf{V}_j$', 
                 fontsize=14, ha='center', va='bottom', color='red', weight='bold')
    ax_flow.text(0.5, 0.45, r'Matrix Multiplication (OV Circuit)', 
                 fontsize=12, ha='center', va='top', color='black')
    
    fig.suptitle("Plot 3: OV Circuit Flow - Information Copying and Aggregation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def softmax(x):
    """Computes softmax along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def simulate_attention(X, W_Q, W_K, W_V, W_O):
    """Simulates a single attention head computation."""
    
    # 1. QK Circuit: Calculates query and key vectors and computes attention scores.
    Q = X @ W_Q
    print(f"2. Q: X @ W_Q -> Shape: {Q.shape}")
    
    K = X @ W_K
    print(f"3. K: X @ W_K -> Shape: {K.shape}")
    
    # Attention Scores (Context_Length x Context_Length)
    scores = Q @ K.T
    d_head = Q.shape[1]
    scores = scores / np.sqrt(d_head)
    print(f"4. Scores: Q @ K.T / sqrt(d_head) -> Shape: {scores.shape}")
    
    # Apply Causal Mask
    mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
    scores[mask] = -np.inf 
    
    # Attention Weights (QK Circuit Output)
    weights = softmax(scores)
    print(f"5. Weights (Softmax + Causal Mask): Shape: {weights.shape}")
    
    # 2. OV Circuit: Calculates value vectors and aggregates information.
    V = X @ W_V
    print(f"6. V: X @ W_V -> Shape: {V.shape}")
    
    # Weighted V (OV Circuit Result)
    head_output = weights @ V
    print(f"7. Head Output: Weights @ V -> Shape: {head_output.shape}")

    # 3. Output Projection
    Z = head_output @ W_O
    print(f"8. Final Output Z: Head_Output @ W_O -> Shape: {Z.shape}")
    
    return Q, K, V, head_output, Z, scores, weights

def visualize_subspaces(X, Q, K, V, head_output, Z, current_token_idx):
    """
    Visualizes the movement of a single token vector through the different attention subspaces (Plot 4).
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
        f"3. Key Subspace (T{current_token_idx - 1}) [Green]": k_vec_prev[:3], # QK Match
        f"4. Value Subspace (T{current_token_idx - 1}) [Yellow/Orange]": v_vec_prev[:3], # OV Source
        "5. Aggregated Head Output (Weighted V) [Red]": head_output_vec[:3],
        "6. Final Model Output (Z) [Cyan]": z_vec[:3]
    }
    
    # Calculate QK dot product for context
    qk_dot_product = np.dot(q_vec, k_vec_prev)
    print(f"\nQK Interaction Focus: Dot Product (Query T{current_token_idx} vs Key T{current_token_idx - 1}) = {qk_dot_product:.2f}")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#f58231', '#d62728', '#17becf']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(f"Plot 4: Information Flow and Subspace Evolution (Focus on T{current_token_idx})", fontsize=14)
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
    CONTEXT_LENGTH = 20  # Length of the simulated token sequence
    
    # Set the token index for focused visualizations (e.g., T5)
    TOKEN_FOCUS_IDX = 5 
    
    print(f"Configuration: d_model={D_MODEL}, d_head={D_HEAD}, Context_Length={CONTEXT_LENGTH}\n")
    
    # 1. Generate Input Embeddings (X = TE + PE)
    Token_Embeddings = np.random.randn(CONTEXT_LENGTH, D_MODEL)
    positions = np.arange(CONTEXT_LENGTH)[:, np.newaxis]
    i = np.arange(0, D_MODEL, 2)
    Position_Embeddings = np.zeros((CONTEXT_LENGTH, D_MODEL))
    Position_Embeddings[:, 0::2] = np.sin(positions / (10000**(i / D_MODEL)))
    Position_Embeddings[:, 1::2] = np.cos(positions / (10000**(i / D_MODEL)))
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
    
    # Print numerical results for context
    print("\n--- 9. QK Attention Scores (Focus Token) ---")
    print(f"Token {TOKEN_FOCUS_IDX} Attention Scores (to all previous keys):\n{scores[TOKEN_FOCUS_IDX]}")
    
    print("\n--- 10. Attention Weights (Focus Token) ---")
    print(f"Token {TOKEN_FOCUS_IDX} Attention Weights (to all previous keys, sums to 1):\n{weights[TOKEN_FOCUS_IDX]}")
    
    # 7. Visualize Subspace Evolution
    print("\n--- 12. Visualizing Subspace Evolution ---")
    visualize_subspaces(X, Q, K, V, head_output, Z, TOKEN_FOCUS_IDX)

