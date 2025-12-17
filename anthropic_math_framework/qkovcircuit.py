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
    
    # Apply a small structure to the matrices for visualization purposes
    W_Q[:5, :5] += 2.0
    W_K[10:15, 10:15] += 1.5
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

def softmax(x):
    """Computes softmax along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def simulate_attention(X, W_Q, W_K, W_V, W_O):
    """Simulates a single attention head computation."""
    
    # 1. QK Circuit: Calculates query and key vectors and computes attention scores.
    # X (Context_Length x d_model)
    
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
    
    # Attention Weights (Context_Length x Context_Length)
    weights = softmax(scores)
    print(f"5. Weights (Softmax): Shape: {weights.shape}")
    
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

def visualize_subspaces(X, Q, V, head_output, Z):
    """
    Visualizes the movement of a single token vector (the first token X[0])
    through the different attention subspaces (Q, V) and back to the model space (Z).
    We use the first three dimensions for 3D visualization.
    """
    
    # Use the first token vector as an example
    x_vec = X[0]
    q_vec = Q[0]
    v_vec = V[0]
    head_output_vec = head_output[0]
    z_vec = Z[0]
    
    # Normalize vectors for clearer plotting (optional, but helps with visuals)
    # Scaling factor to make all vectors roughly the same visible length
    scale_factor = 2.0 / np.linalg.norm(x_vec[:3])
    
    # Collect the vectors for visualization (using the first 3 dimensions)
    vectors = {
        "1. Input Vector (X)": x_vec[:3],
        "2. Query Subspace (Q)": q_vec[:3] * scale_factor,
        "3. Value Subspace (V)": v_vec[:3] * scale_factor,
        "4. Aggregated Head Output (Weighted V)": head_output_vec[:3] * scale_factor,
        "5. Final Model Output (Z)": z_vec[:3]
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title("Information Flow and Subspace Evolution (First 3 Dimensions)", fontsize=14)
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
    CONTEXT_LENGTH = 10  # Number of tokens in the input sequence (e.g., 10 words)
    
    print(f"Configuration: d_model={D_MODEL}, d_head={D_HEAD}, Context_Length={CONTEXT_LENGTH}\n")
    
    # 1. Generate Input Embeddings (X)
    X = np.random.randn(CONTEXT_LENGTH, D_MODEL)
    print(f"0. Input Embeddings X (Context_Length x d_model): Shape: {X.shape}\n")
    
    # 2. Generate Weight Matrices
    W_Q, W_K, W_V, W_O = generate_matrices(D_MODEL, D_HEAD)
    
    # 3. Visualize Weight Matrices
    visualize_matrices(
        [W_Q, W_K, W_V, W_O.T], # W_O.T is plotted as d_model x d_head for visual consistency
        ["W_Q (Query Matrix)", "W_K (Key Matrix)", "W_V (Value Matrix)", "W_O.T (Output Projection)"]
    )

    # 4. Simulate the Attention Mechanism
    Q, K, V, head_output, Z, scores, weights = simulate_attention(X, W_Q, W_K, W_V, W_O)
    
    print("\n--- 9. QK Attention Scores (First Token) ---")
    print(f"Token 0 Attention Scores (to all keys):\n{scores[0]}")
    
    print("\n--- 10. Attention Weights (First Token) ---")
    print(f"Token 0 Attention Weights (to all keys, sums to 1):\n{weights[0]}")
    
    # 5. Visualize Subspace Evolution
    print("\n--- 11. Visualizing Subspace Evolution ---")
    visualize_subspaces(X, Q, V, head_output, Z)

