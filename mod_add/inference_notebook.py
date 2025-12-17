import torch
import numpy as np
import os
from transformer_lens import HookedTransformer, HookedTransformerConfig
from neel.imports import *
import matplotlib
# Set the backend to 'Agg' (for non-interactive file generation)
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

def loss_fn(logits,labels):
    if len(logits.shape)==3:
        logits = logits[:,-1]

    # set logits to float 64 before softmax
    logits = logits.to(torch.float64)

    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1,index=labels[:,None])[:,0]
    return -correct_log_probs.mean()


# --- Configuration (MUST match the training script) ---
p = 113 # Modulus
artifacts_dir = "modular_addition_artifacts"
final_model_path = os.path.join(artifacts_dir, "final_trained_model.pt")

# hyperparams
frac_train = 0.3
lr = 1e-3
wd = 1 # the greater, the greater the incentives to be simpler for the model
betas = (0.9, 0.98) # beta2 affects the model behavior during grokking
num_epochs = 15000
checkpoint_every = 5000

## Define dataset
a_vector = einops.repeat(torch.arange(p),"i -> (i j)", j=p)
b_vector = einops.repeat(torch.arange(p),"j -> (i j)", i=p)
equals_vector = einops.repeat(torch.tensor(113), "-> (i j)", i=p,j=p)

dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).cuda()
print(dataset)

labels = (dataset[:,0]+dataset[:,1])%p 
indices = torch.randperm(p*p)
cutoff = int(p*p*frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]

train_data = dataset[train_indices]
train_labels = labels[train_indices]
test_data = dataset[test_indices]
test_labels = labels[test_indices]


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

def plot_attention_pattern_heatmap(cache):
    """
    Plots the attention pattern TO THE FINAL TOKEN (=) 
    from all source tokens, for all heads, in a single heatmap.
    Heads are on the y-axis, source tokens (A, B, =) are on the x-axis.
    """
    print("\n--- Plotting Layer 0 Attention to Final Token (=) across All Heads (Single Heatmap) ---")

    try:
        # Data: [n_heads, src_seq_len] - attention to the final token from all source tokens, for all heads
        attention_pattern = cache["pattern", 0][5][:, -1, :] 
    except KeyError:
        print("Error: Could not find 'pattern, 0' in cache. Caching may be off.")
        return
    except IndexError:
        print("Error: Cache structure seems incorrect.")
        return

    # 'attention_pattern' is already the correct shape for a single heatmap: (n_heads, src_seq_len)
    n_heads = attention_pattern.shape[0]
    src_seq_len = attention_pattern.shape[1] 
    
    # Source tokens for the x-axis
    source_token_labels = ['A', 'B', '='] 
    # Head indices for the y-axis
    head_labels = [f"Head {h}" for h in range(n_heads)]
    
    # Convert to numpy for plotting, if it's not already
    attention_pattern_np = attention_pattern.cpu().numpy()
    
    # --- Setup the single plot ---
    # Adjust figure size for better aspect ratio
    fig_width = 5 
    fig_height = max(4, n_heads * 0.4) 
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height)) 

    # Plot the heatmap
    # Use 'origin="lower"' so Head 0 is at the bottom, or 'origin="upper"' 
    # (default) so Head 0 is at the top. 'upper' is often clearer for indices.
    im = ax.imshow(attention_pattern_np, cmap='viridis', aspect='auto') 
    
    # Set X-axis (Source Tokens)
    ax.set_xticks(np.arange(src_seq_len))
    ax.set_xticklabels(source_token_labels)
    ax.set_xlabel("Source Token Position")
    
    # Set Y-axis (Heads)
    ax.set_yticks(np.arange(n_heads)) 
    ax.set_yticklabels(head_labels)
    ax.set_ylabel("Attention Head")
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax, label="Attention Weight")
    
    # Add text labels on the heatmap cells
    vmin = attention_pattern_np.min().item()
    vmax = attention_pattern_np.max().item()
    threshold = (vmax - vmin) * 0.5 + vmin
    
    for i in range(n_heads): # Rows (Heads)
        for j in range(src_seq_len): # Columns (Source Tokens)
            weight = attention_pattern_np[i, j].item() 
            text_color = "white" if weight > threshold else "black"
            ax.text(j, i, f"{weight:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.suptitle("Layer 0: Attention to Final Token (=) across All Heads", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle and labels
    plt.show()

    # Optional: uncomment to save and close
    # plt.savefig('attention_single_heatmap.png')
    # plt.close(fig)
    
    print("\nSingle heatmap plot successfully displayed.")
    
    return # End of function

def plot_attention_pattern_all_heads_byrows(cache):
    """
    Plots the attention pattern TO THE FINAL TOKEN (=) 
    from all source tokens, for each head, in a single row per head.
    """
    print("\n--- Plotting Layer 0 Attention to Final Token (=) by Head ---")

    try:
        attention_pattern = cache["pattern", 0][4][:, -1, :] 
    except KeyError:
        print("Error: Could not find 'pattern, 0' in cache. Caching may be off.")
        return
    except IndexError:
        print("Error: Cache structure seems incorrect.")
        return

    n_heads = attention_pattern.shape[0]
    src_seq_len = attention_pattern.shape[1] 
    
    source_token_labels = ['A', 'B', '=']
    dest_token_label = ['=']
    
    vmin = attention_pattern.min().item()
    vmax = attention_pattern.max().item()
    
    # --- Setup the subplot grid ---
    ncols = 2
    nrows = int(math.ceil(n_heads / ncols))
    
    # Check if the calculated figure size is reasonable
    # You might consider reducing the figsize if n_heads is large.
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 2.5), squeeze=False) 

    im = None
    
    # ... (Plotting loop remains unchanged) ...
    for h in range(n_heads):
        row = h // ncols
        col = h % ncols
        ax = axes[row, col]
        
        head_pattern_1d = attention_pattern[h].cpu().numpy()
        head_pattern_2d = head_pattern_1d.reshape(1, src_seq_len)
        
        im = ax.imshow(head_pattern_2d, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax, aspect='auto') 
        
        ax.set_xticks(np.arange(len(source_token_labels)))
        ax.set_xticklabels(source_token_labels)
        
        ax.set_yticks(np.arange(len(dest_token_label))) 
        ax.set_yticklabels(dest_token_label)
        
        ax.set_xlabel("Source Token Position")
        ax.set_ylabel("Dest. Token")
        ax.set_title(f"Head {h}")
        
        threshold = (vmax - vmin) * 0.5 + vmin
        
        for j in range(src_seq_len):
            weight = head_pattern_1d[j].item() 
            text_color = "white" if weight > threshold else "black"
            ax.text(j, 0, f"{weight:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    # Hide any unused subplots
    #for i in range(n_heads, nrows * ncols):
    #    row = i // ncols
    #    col = i % ncols
    #    axes[row, col].axis('off')

    # Add a single, shared colorbar
    #fig.colorbar(im, ax=axes.ravel().tolist(), label="Attention Weight", shrink=0.8)
    
    plt.suptitle("Layer 0: Attention to Final Token (=) by Head", fontsize=16)
    plt.show()
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 1. Save the figure (essential for 'Agg' backend)
    #plt.savefig('attention_plot.png')
    
    # 2. Explicitly close the figure object to free memory immediately
    #plt.close(fig)
    
    # 3. Force garbage collection to ensure related Matplotlib objects are deleted
    #gc.collect() 
    
    print("\nPlot successfully saved as 'attention_plot.png'.")


def plot_attention_pattern_all_heads(cache):
    """Plots the attention pattern for each head in the single layer."""
    
    print("\n--- Plotting Layer 0 Attention Patterns by Head ---")

    # Cache key for attention pattern in layer 0
    attention_pattern = cache["pattern", 0].mean(dim=0) # Shape: [n_heads, dest_seq_len, src_seq_len]
    
    n_heads = attention_pattern.shape[0]
    
    # Labels for source and destination tokens: [A, B, =]
    token_labels = ['A', 'B', '=']
    
    # Find global min/max for a consistent color scale across all heads
    vmin = attention_pattern.min().item()
    vmax = attention_pattern.max().item()
    
    # Setup the subplot grid
    # We'll default to 2 columns for a nice layout
    ncols = 2
    # Calculate rows needed
    nrows = int(math.ceil(n_heads / ncols))
    
    # Create the figure. squeeze=False ensures 'axes' is always a 2D array
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)

    im = None # To store the mappable object for the colorbar
    
    for h in range(n_heads):
        # Calculate the row and column index for this head's subplot
        row = h // ncols
        col = h % ncols
        ax = axes[row, col]
        
        # Get the pattern for this specific head
        head_pattern = attention_pattern[h].cpu().numpy()
        
        # Plot the attention matrix for this head
        im = ax.imshow(head_pattern, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(np.arange(len(token_labels)))
        ax.set_xticklabels(token_labels)
        ax.set_yticks(np.arange(len(token_labels)))
        ax.set_yticklabels(token_labels)
        
        ax.set_xlabel("Source Token Position")
        ax.set_ylabel("Destination Token Position")
        ax.set_title(f"Head {h}")
        
        # Text annotations
        # Threshold for text color is based on the global scale
        threshold = (vmax - vmin) * 0.5 + vmin
        
        for i in range(head_pattern.shape[0]):
            for j in range(head_pattern.shape[1]):
                weight = head_pattern[i, j].item()
                # Choose text color based on background intensity
                text_color = "white" if weight > threshold else "black"
                ax.text(j, i, f"{weight:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    # Hide any unused subplots
    for i in range(n_heads, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('off')

    # Add a single, shared colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Attention Weight", shrink=0.8)
    
    # Add an overall title
    plt.suptitle("Layer 0: Attention Patterns by Head", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

from matplotlib.colors import LinearSegmentedColormap, Normalize

def plot_neuron_activations(neuron_acts, n):
    pattern = neuron_acts[:,:n]
    pattern_rearranged = einops.rearrange(
        pattern, 
        "(a b) neuron -> neuron a b", 
        a=p, b=p
    ).cpu()

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    
    # Set up the color format
    colors = [
        (1, 0, 0),  # Red
        (1, 1, 1),  # White
        (0, 0, 1)   # Blue
    ]
    
    # Calculate global normalization parameters
    max_abs_val = np.abs(pattern_rearranged).max()
    vmin = -max_abs_val
    vmax = max_abs_val
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("RedWhiteBlue", colors, N=256)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    token_labels = ['A', 'B', '=']
    
    for i in range(n):
        ax = axes[i]
        im = ax.imshow(pattern_rearranged[i], cmap=cmap, norm=norm, origin='upper')
        
        ax.set_xticks(np.arange(len(token_labels)))
        ax.set_yticks(np.arange(len(token_labels)))
        ax.set_xticklabels(token_labels)
        ax.set_yticklabels(token_labels)
        
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        ax.set_title(f"Neuron {i} from a -> =")
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Neuron Weight")
    
    plt.show()

def plot_attention_pattern_multiple_heads(cache):
    """Plots attention patterns for all heads in a grid layout."""
    
    # Get attention pattern and rearrange using einops
    attention_pattern = cache["pattern", 0]
    
    # Rearrange using einops to get [head, a, b]
    pattern_rearranged = einops.rearrange(
        attention_pattern[:, :, -1, 0], 
        "(a b) head -> head a b", 
        a=p, b=p
    ).cpu()
    
    n_heads = pattern_rearranged.shape[0]
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_heads)))
    n_rows = int(np.ceil(n_heads / n_cols))
    
    # Set up the color format
    colors = [
        (1, 0, 0),  # Red
        (1, 1, 1),  # White
        (0, 0, 1)   # Blue
    ]
    
    # Calculate global normalization parameters
    max_abs_val = np.abs(pattern_rearranged).max()
    vmin = -max_abs_val
    vmax = max_abs_val
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("RedWhiteBlue", colors, N=256)
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    token_labels = ['A', 'B', '=']
    
    for i in range(n_heads):
        ax = axes[i]
        im = ax.imshow(pattern_rearranged[i], cmap=cmap, norm=norm, origin='upper')
        
        ax.set_xticks(np.arange(len(token_labels)))
        ax.set_yticks(np.arange(len(token_labels)))
        ax.set_xticklabels(token_labels)
        ax.set_yticklabels(token_labels)
        
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        ax.set_title(f"Head {i} from a -> =")
    
    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Attention Weight")
    
    plt.show()

# daert version
def plot_attention_pattern_h0_a_to_equal(cache):
    """Plots the attention pattern averaged across all heads in the single layer."""
    
    print("\n--- Plotting Layer 0 Attention Pattern ---")

    # Cache key for attention pattern in layer 0
    # Shape is likely [batch_size, n_heads, dest_seq_len, src_seq_len] = [1, 4, 3, 3]
    attention_pattern = cache["pattern", 0]
    
    # *** THE FIX IS HERE ***
    # 1. Select the first (and only) item from the batch
    # Shape becomes [n_heads, dest_seq_len, src_seq_len] = [4, 3, 3]
    pattern_batch_item_0 = attention_pattern[:, 0, -1, 0].reshape(p,p).cpu()
    
    # 2. Average across all heads (dim=0)
    # Shape becomes [dest_seq_len, src_seq_len] = [3, 3]
    #avg_attention_pattern = pattern_batch_item_0.mean(dim=0).cpu().numpy()
    
    # Calculate threshold once outside the loop and ensure it is a scalar float
    max_attention = pattern_batch_item_0.max().item()
    threshold = 0.5 * max_attention
    
    # Plotting
    plt.figure(figsize=(6, 6))

    colors = [
        (1, 0, 0),  # Red
        (1, 1, 1),  # White
        (0, 0, 1)   # Blue
    ]
    max_abs_val = np.abs(pattern_batch_item_0).max()
    vmin = -max_abs_val
    vmax = max_abs_val
    
    # Normalize the colormap to center at 0
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create the colormap
    # 'N' is the number of discrete segments (optional, good practice for continuous data)
    cmap = LinearSegmentedColormap.from_list("RedWhiteBlue", colors, N=256)
    
    
    # Using imshow to visualize the attention matrix
    plt.imshow(pattern_batch_item_0, cmap=cmap, norm=norm, origin='upper')
    
    # Labels for source and destination tokens: [A, B, =]
    token_labels = ['A', 'B', '=']
    
    plt.xticks(np.arange(len(token_labels)), token_labels)
    plt.yticks(np.arange(len(token_labels)), token_labels)
    
    plt.xlabel("b")
    plt.ylabel("a")
    plt.title("Attention for Head 0 from a -> =")
    
    # Add color bar to show the scale of attention weights
    cbar = plt.colorbar(label="Attention Weight")
    
    # Add text annotations for attention weights
    #for i in range(pattern_batch_item_0.shape[0]):
    #    for j in range(pattern_batch_item_0.shape[1]):
    #        # This should now be a scalar, but we use .item() just to be safe
    #        weight = pattern_batch_item_0[i, j].item()
            
            # This comparison is now between two plain Python floats
    #        text_color = "white" if weight > threshold else "black"
    #        plt.text(j, i, f"{weight:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    plt.show()

def plot_attention_pattern(cache):
    """Plots the attention pattern averaged across all heads in the single layer."""
    
    print("\n--- Plotting Layer 0 Attention Pattern ---")

    # Cache key for attention pattern in layer 0
    # Shape is likely [batch_size, n_heads, dest_seq_len, src_seq_len] = [1, 4, 3, 3]
    attention_pattern = cache["pattern", 0]
    
    # *** THE FIX IS HERE ***
    # 1. Select the first (and only) item from the batch
    # Shape becomes [n_heads, dest_seq_len, src_seq_len] = [4, 3, 3]
    pattern_batch_item_0 = attention_pattern[0, :, :, :]
    
    # 2. Average across all heads (dim=0)
    # Shape becomes [dest_seq_len, src_seq_len] = [3, 3]
    avg_attention_pattern = pattern_batch_item_0.mean(dim=0).cpu().numpy()
    
    # Calculate threshold once outside the loop and ensure it is a scalar float
    max_attention = avg_attention_pattern.max().item()
    threshold = 0.5 * max_attention
    
    # Plotting
    plt.figure(figsize=(6, 6))
    
    # Using imshow to visualize the attention matrix
    plt.imshow(avg_attention_pattern, cmap='viridis', origin='upper')
    
    # Labels for source and destination tokens: [A, B, =]
    token_labels = ['A', 'B', '=']
    
    plt.xticks(np.arange(len(token_labels)), token_labels)
    plt.yticks(np.arange(len(token_labels)), token_labels)
    
    plt.xlabel("Source Token Position")
    plt.ylabel("Destination Token Position")
    plt.title("Layer 0: Mean Attention Pattern (Heads 0-3)")
    
    # Add color bar to show the scale of attention weights
    cbar = plt.colorbar(label="Attention Weight")
    
    # Add text annotations for attention weights
    for i in range(avg_attention_pattern.shape[0]):
        for j in range(avg_attention_pattern.shape[1]):
            # This should now be a scalar, but we use .item() just to be safe
            weight = avg_attention_pattern[i, j].item()
            
            # This comparison is now between two plain Python floats
            text_color = "white" if weight > threshold else "black"
            plt.text(j, i, f"{weight:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    plt.show()

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

def plot_heatmap(tensor_2d, title=None, xaxis=None, yaxis=None, xticks=None, yticks=None):
    heatmap_data = tensor_2d.cpu().detach().numpy()
    # 3. Generate the Heatmap Plot
    plt.figure(figsize=(8, 7))

    colors = [
        (1, 0, 0),  # Red
        (1, 1, 1),  # White
        (0, 0, 1)   # Blue
    ]
    max_abs_val = np.abs(heatmap_data).max()
    vmin = -max_abs_val
    vmax = max_abs_val
    
    # Normalize the colormap to center at 0
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create the colormap
    # 'N' is the number of discrete segments (optional, good practice for continuous data)
    cmap = LinearSegmentedColormap.from_list("RedWhiteBlue", colors, N=256)

    # The imshow function displays data as an image, where values are mapped to colors.
    # 'cmap' sets the color map (e.g., 'viridis', 'hot', 'coolwarm').
    # 'interpolation' is set to 'nearest' to ensure clear, defined squares for each data point.
    img = plt.imshow(heatmap_data, cmap=cmap,norm=norm, interpolation='nearest')

    # 4. Add a Colorbar for Scale Interpretation
    # The color bar shows the mapping between the numeric value and the color intensity.
    plt.colorbar(img, label='Tensor Value (e.g., 0.0 to 1.0)')

    # 5. Add Labels and Title
    if not xaxis:
        xaxis = 'Column Index'
    if not yaxis:
        yaxis = 'Row Index'
    if not title:
        title = 'PyTorch Tensor as a Heatmap (10x10 Random Data)'
    if not xticks:
        xticks = np.arange(10)
    if not yticks:
        yticks = np.arange(10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xaxis, fontsize=12)
    plt.ylabel(yaxis, fontsize=12)

    # Optional: Add axis ticks for clarity
    plt.xticks(range(len(xticks)),xticks)
    plt.yticks(range(len(yticks)),yticks)

    # 6. Display the plot
    plt.show()

def plot_lines(y_tensor,xaxis=None,yaxis=None,xticks=None,yticks=None,title=None):
    y_data = y_tensor.cpu().detach().numpy()

    # 3. Create the X-axis Data (Indices/Steps)
    # The X-axis will simply be the index of each element in the vector.
    x_data = np.arange(len(y_data))

    # 4. Generate the Plot
    plt.figure(figsize=(10, 6))
    # Use the numpy array data for both X and Y
    plt.plot(x_data, y_data, marker='', linestyle='-', color='teal', linewidth=2)

    # 5. Add Labels, Title, and Grid for clarity
    if not title:
        title='PyTorch Vector Visualization (Y-values vs. Index)'
    
    if not xaxis:
        xaxis='Vector Index / Time Step (X-axis)'
    
    if not yaxis:
        yaxis='Tensor Value (Y-axis)'

    if not xticks:
        xticks = np.arange(10)
    if not yticks:
        yticks = np.arange(10)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xaxis, fontsize=12)
    plt.ylabel(yaxis, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    # Optional: Add axis ticks for clarity
    plt.xticks(range(len(xticks)),xticks,rotation=90)
    plt.yticks(range(len(yticks)),yticks)

    # 6. Display the plot
    plt.show()


def plot_svd(y_tensor):
    y_data = y_tensor.cpu().detach().numpy()

    # 3. Create the X-axis Data (Indices/Steps)
    # The X-axis will simply be the index of each element in the vector.
    x_data = np.arange(len(y_data))

    # 4. Generate the Plot
    plt.figure(figsize=(10, 6))
    # Use the numpy array data for both X and Y
    plt.plot(x_data, y_data, marker='', linestyle='-', color='teal', linewidth=2)

    # 5. Add Labels, Title, and Grid for clarity
    plt.title('PyTorch Vector Visualization (Y-values vs. Index)', fontsize=16, fontweight='bold')
    plt.xlabel('Vector Index / Time Step (X-axis)', fontsize=12)
    plt.ylabel('Tensor Value (Y-axis)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 6. Display the plot
    plt.show()


model = load_and_initialize_model()

original_logits, cache = model.run_with_cache(dataset)

W_E = model.embed.W_E[:-1]

# W_neur: input to MLP
# W_logit: output of the whole model
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U

# does the model work?
original_loss = loss_fn(original_logits,labels).item()
print("Original loss:",original_loss)

# Print the shapes
for param_name, param in cache.items():
    print(param_name,param.shape)

# Print average attention pattern
#plot_attention_pattern_all_heads_byrows(cache)
#plot_attention_pattern_heatmap(cache)

#plot_attention_pattern_h0_a_to_equal(cache)

#npx.imshow(
#    einops.rearrange(cache["pattern",0][:,:,-1,0], "(a b) head -> head a b",a=p,b=p),
#    title="Attention for Head 0 from a -> =",xaxis="b",yaxis="a",facet_col=0
#)
#plot_attention_pattern_multiple_heads(cache)

# Plot neuron activations
pattern_a = cache["pattern",0,"attn"][:,:,-1,0]
pattern_b = cache["pattern",0,"attn"][:,:,-1,1]
neuron_acts = cache["post",0,"mlp"][:,-1,:]
neuron_pre_acts = cache["pre",0,"mlp"][:,-1,:]

#plot_neuron_activations(neuron_acts,5)

# single value decomposition of mlp
U, S, Vh = torch.svd(W_E)
#plot_svd(S)

# plot principal components
#plot_heatmap(U)

# control code: get svd of gaussian matrix
#U, S, Vh = torch.svd(torch.rand_like(W_E))
#plot_svd(S)
#plot_heatmap(U)

## Analyse embedding
#plot_lines(U[:,:2])

fourier_basis = []
fourier_basis_names = []
fourier_basis.append(torch.ones(p))
fourier_basis_names.append("Constant")
for freq in range(1,p//2+1):
    fourier_basis.append(torch.sin(torch.arange(p)*2*torch.pi*freq/p))
    fourier_basis_names.append(f"Sin {freq}")
    fourier_basis.append(torch.cos(torch.arange(p)*2*torch.pi*freq/p))
    fourier_basis_names.append(f"Cos {freq}")

fourier_basis = torch.stack(fourier_basis, dim=0).cuda()
fourier_basis = fourier_basis/fourier_basis.norm(dim=-1)
#plot_heatmap(fourier_basis)

#plot_lines(fourier_basis[:8])

# all vectors in the fourier basis are orthogonal,
# this the diag matrix appears when multiplied by transpose
#plot_heatmap(fourier_basis @ fourier_basis.T)

# Plot the embedding in the fourier basis
#plot_heatmap(fourier_basis @ W_E,xticks=W_E.shape,yticks=fourier_basis_names,xaxis='Residual Stream',yaxis='Fourier Component',title='Embedding in Fourier basis')

#plot_lines((fourier_basis @ W_E).norm(dim=-1),xaxis="Fourier Component",yaxis="y",title="norms of embedding in fourier basis",xticks=fourier_basis_names)

# check that the learned directions in the embedding state are orthogonal
#TODO
#key_freqs = []
#key_freq_indices = []
#fourier_embed = fourier_basis @ W_E
#key_fourier_embed = fourier_embed[key_fourier_indices]
#print("key_fourier_embed",key_fourier_embed.shape)
#plot_heatmap(key_fourier_embed @ key_fourier_embed.T)

# Constructive Interference: mean of all the different key freqs

# cos(wa)*cos(wb)
#plot_heatmap(fourier_basis[94][None,:]*fourier_basis[94][:,None])

# 2d fourier transformer of neuron 0
# this let us know what neuros learn
# the activated places are the coefficients of the different products
plot_heatmap(fourier_basis@neuron_acts[:,0].reshape(p,p)@fourier_basis.T,title="2D fourier transformer of neuron 0",yticks=fourier_basis_names,xticks=fourier_basis_names)

# get neuron clusters