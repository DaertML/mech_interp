from neel.imports import *

artifacts_dir = "modular_addition_artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

p = 113

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

## Define model
cfg = HookedTransformerConfig(
    n_layers = 1,
    n_heads = 4,
    d_model = 128,
    d_head = 32,
    d_mlp = 512,
    act_fn = "relu",
    normalization_type=None,
    d_vocab = p+1,
    d_vocab_out=p,
    n_ctx = 3,
    init_weights=True,
    device='cuda'
)

model = HookedTransformer(cfg)

# Disable biases. Not necessary for the task
# init to 0 as it is disabled
for name, param in model.named_parameters():
    if "b_" in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=wd, betas=betas)

def loss_fn(logits,labels):
    if len(logits.shape)==3:
        logits = logits[:,-1]

    # set logits to float 64 before softmax
    logits = logits.to(torch.float64)

    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1,index=labels[:,None])[:,0]
    return -correct_log_probs.mean()

# check if the loss is equal to the uniform loss before training
train_logits = model(train_data)
train_loss = loss_fn(train_logits, train_labels)
print("Train loss:",train_loss)

test_logits = model(test_data)
test_loss = loss_fn(test_logits,test_labels)
print("Test loss:",test_loss)

print("Uniform loss:",np.log(p))


## Train model
#### Train with a single backward update for the whole batch.
#### Objective: make the training as clean as possible in terms of choices

train_losses = []
test_losses = []
model_checkpoints = []
checkpoint_epochs = []
for epoch in tqdm.tqdm(range(num_epochs)):
    train_logits = model(train_data)
    train_loss = loss_fn(train_logits, train_labels)
    train_loss.backward()
    train_losses.append(train_loss.item())

    optimizer.step()
    optimizer.zero_grad()

    with torch.inference_mode():
        test_logits = model(test_data)
        test_loss = loss_fn(test_logits, test_labels)
        test_losses.append(test_loss.item())

    if (epoch+1)%checkpoint_every:
        checkpoint_epochs.append(epoch)
        model_checkpoints.append(copy.deepcopy(model.state_dict()))
        print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")


## Save final model and history
print("--- Training complete. Saving final artifacts ---")

# 1. Save the final trained model
final_model_path = os.path.join(artifacts_dir, "final_trained_model.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# 2. Save the loss and checkpoint history lists
history = {
    "train_losses": train_losses,
    "test_losses": test_losses,
    "checkpoint_epochs": checkpoint_epochs,
    "model_checkpoints_list": model_checkpoints # Note: This list might be large
}
history_path = os.path.join(artifacts_dir, "training_history.pt")
torch.save(history, history_path)
print(f"Training history saved to {history_path}")

#import neel_plotly.plot as npx
#npx.line([train_losses[::100],test_losses[::100]], x=np.arange(0,len(train_losses),100),xaxis="Epoch",yaxis="Loss",log_y=True,title="Training curve Modular Addition",line_labels=['train','test'],toggle_x=True,toggle_y=True)

# Plotting the results using Matplotlib
print("--- Generating loss plot using Matplotlib ---")

# Data preparation for plotting (using every 100th point)
import matplotlib.pyplot as plt
plot_epochs = np.arange(0, len(train_losses), 100)
plot_train_losses = np.array(train_losses)[::100]
plot_test_losses = np.array(test_losses)[::100]

plt.figure(figsize=(10, 6))
plt.plot(plot_epochs, plot_train_losses, label='Train Loss', color='blue')
plt.plot(plot_epochs, plot_test_losses, label='Test Loss', color='red')

plt.xlabel("Epoch (Every 100th)")
plt.ylabel("Loss (Log Scale)")
plt.yscale('log')
plt.title("Training Curve Modular Addition")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
# In a non-interactive environment, the plot needs to be explicitly saved or shown.
# Since we are in an execution environment, we will use plt.show()
plt.show()