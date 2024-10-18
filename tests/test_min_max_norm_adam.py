#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path) 
from src.lazy_adamw import LazyAdamW
#%%

import torch
import torch.nn as nn
import torch.optim

# Define the LazyAdamW optimizer (include the corrected code here or import it)

# Define the simple model
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pnorm=None):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.pnorm = pnorm
        self.init_embedding()

    def init_embedding(self):
        if self.pnorm is None:
            return
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight)
                    norm = m.weight.data.norm(p=2, dim=1, keepdim=True)
                    m.weight.data.copy_(m.weight.data * (self.pnorm / norm))


    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        output = self.linear(embedded)
        return output

# Hyperparameters
vocab_size = 10
embedding_dim = 100
output_dim = 1
batch_size = 16
sequence_length = 10

# Instantiate the model
model = SimpleModel(vocab_size, embedding_dim, output_dim, pnorm=0.5)
# list(model.modules())
#%%
print("Initial Embedding Norms", model.embedding.weight.norm(p=2, dim=1))

#%%
# Define optimizer with separate parameter groups
optimizer = LazyAdamW(
    [
        {
            'params': model.embedding.parameters(),
            'lr': 1e-2,
            'weight_decay': 0.00,
            'l1_coeff': 0.0,
            'min_norm': 0.5,
            'max_norm': 0.89,
        },
        {
            'params': model.linear.parameters(),
            'lr': 1e-2,
            'weight_decay': 0.01,
            'l1_coeff': 0.05,
            # No min_norm and max_norm specified here
        },
    ]
)




criterion = nn.MSELoss()

# Generate random input data and targets
inputs = torch.randint(0, vocab_size, (batch_size, sequence_length), dtype=torch.long)
targets = torch.randn(batch_size, output_dim)

# Initial sum of absolute parameter values
with torch.no_grad():
    initial_param_sum = sum(p.abs().sum().item() for p in model.parameters())

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    loss.backward()

    # Perform optimization step
    optimizer.step()

    # Monitor training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Check norms of embedding parameters
    with torch.no_grad():
        # Embedding layer norms
        embedding_weights = model.embedding.weight
        updated_indices = inputs.unique()
        updated_embeddings = embedding_weights[updated_indices]
        embedding_norms = updated_embeddings.norm(p=2, dim=1)
        min_norm_value = embedding_norms.min().item()
        max_norm_value = embedding_norms.max().item()
        mean_norm_value = embedding_norms.mean().item()
        print(f"Embedding norms - min: {min_norm_value:.4f}, max: {max_norm_value:.4f}, mean: {mean_norm_value:.4f}")
        # Assert that norms are within the specified range
        assert min_norm_value >= optimizer.param_groups[0]['min_norm'] - 1e-4, "Min norm constraint violated in embeddings"
        assert max_norm_value <= optimizer.param_groups[0]['max_norm'] + 1e-4, "Max norm constraint violated in embeddings"

        # Linear layer norms (no constraints)
        linear_weights = model.linear.weight
        linear_norms = linear_weights.norm(p=2, dim=1)
        min_linear_norm = linear_norms.min().item()
        max_linear_norm = linear_norms.max().item()
        mean_linear_norm = linear_norms.mean().item()
        print(f"Linear layer norms - min: {min_linear_norm:.4f}, max: {max_linear_norm:.4f}, mean: {mean_linear_norm:.4f}")
        # No assertions here since norms are not constrained

# Final sum of absolute parameter values
with torch.no_grad():
    final_param_sum = sum(p.abs().sum().item() for p in model.parameters())

print(f"Initial sum of absolute parameter values: {initial_param_sum:.4f}")
print(f"Final sum of absolute parameter values: {final_param_sum:.4f}")
assert final_param_sum < initial_param_sum, "L1 regularization did not decrease parameter values as expected"


# %%
