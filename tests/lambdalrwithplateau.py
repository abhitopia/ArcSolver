#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory
from src.lr_scheduler import LambdaLRWithReduceOnPlateau
#%%
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

# Simulate an optimizer with dummy parameters
dummy_param = torch.nn.Parameter(torch.zeros(1))
optimizer_lambda = SGD([dummy_param], lr=0.1)
optimizer_hybrid = SGD([dummy_param], lr=0.1)

# Define lambda function
lambda_func = lambda epoch: 0.95 ** epoch  # Decay LR by 5% every epoch

# Initialize LambdaLR
lambda_scheduler = LambdaLR(optimizer_lambda, lr_lambda=lambda_func)

# Initialize HybridLRScheduler with only lambda-based scheduling
hybrid_scheduler = LambdaLRWithReduceOnPlateau(optimizer_hybrid, lr_lambda=lambda_func)

# Simulate epochs
num_epochs = 10
print("Epoch\tLambdaLR\tHybridLR")
for epoch in range(num_epochs):
    # LambdaLR scheduler step
    lambda_scheduler.step()
    lr_lambda = optimizer_lambda.param_groups[0]['lr']

    # HybridLRScheduler step
    hybrid_scheduler.step()
    lr_hybrid = optimizer_hybrid.param_groups[0]['lr']

    print(f"{epoch+1}\t{lr_lambda:.6f}\t{lr_hybrid:.6f}")

#%%

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Simulate an optimizer with dummy parameters
dummy_param = torch.nn.Parameter(torch.zeros(1))
optimizer_plateau = SGD([dummy_param], lr=0.1)
optimizer_hybrid = SGD([dummy_param], lr=0.1)

# Initialize ReduceLROnPlateau
plateau_scheduler = ReduceLROnPlateau(optimizer_plateau, mode='max', factor=0.1, patience=2, verbose=True)

# Initialize HybridLRScheduler with only plateau-based scheduling
hybrid_scheduler = LambdaLRWithReduceOnPlateau(
    optimizer_hybrid,
    lr_lambda=lambda epoch: 1.0,  # Lambda function that returns 1 (no change)
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True
)

# Simulated validation metrics
metrics = [0.8, 0.82, 0.81, 0.81, 0.81, 0.83, 0.83, 0.82, 0.85, 0.85]

print("Epoch\tReduceLR\tHybridLR")
for epoch, metric in enumerate(metrics):
    # ReduceLROnPlateau step
    plateau_scheduler.step(metric)
    lr_plateau = optimizer_plateau.param_groups[0]['lr']

    # HybridLRScheduler step_metric()
    hybrid_scheduler.step_metric(metric)
    lr_hybrid = optimizer_hybrid.param_groups[0]['lr']

    print(f"{epoch+1}\t{lr_plateau:.6f}\t{lr_hybrid:.6f}")

# %%
# Simulate an optimizer with dummy parameters
dummy_param = torch.nn.Parameter(torch.zeros(1))
optimizer_hybrid = SGD([dummy_param], lr=0.1)

# Define lambda function
lambda_func = lambda epoch: 0.95 ** epoch  # Decay LR by 5% every epoch

# Initialize HybridLRScheduler with both scheduling methods
hybrid_scheduler = LambdaLRWithReduceOnPlateau(
    optimizer_hybrid,
    lr_lambda=lambda_func,
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True
)

# Simulated validation metrics
metrics = [0.8, 0.82, 0.81, 0.81, 0.81, 0.83, 0.83, 0.82, 0.82, 0.85, 0.85]

print("Epoch\tHybridLR\tMetric")
for epoch, metric in enumerate(metrics):
    # HybridLRScheduler step()
    hybrid_scheduler.step()

    # HybridLRScheduler step_metric()
    hybrid_scheduler.step_metric(metric)
    lr_hybrid = optimizer_hybrid.param_groups[0]['lr']


    print(f"{epoch+1}\t{lr_hybrid:.6f}\t{metric:.2f}")

# %%
# Save the state
# scheduler_state = hybrid_scheduler.state_dict()
# optimizer_state = optimizer_hybrid.state_dict()

# # Create new optimizer and scheduler
# optimizer_new = SGD([dummy_param], lr=0.1)
# scheduler_new = LambdaLRWithReduceOnPlateau(
#     optimizer_new,
#     lr_lambda=lambda_func,
#     mode='max',
#     factor=0.1,
#     patience=2,
#     verbose=True
# )

# # Load the state
# scheduler_new.load_state_dict(scheduler_state)
# optimizer_new.load_state_dict(optimizer_state)

# # Continue training
# metrics = [0.85, 0.84, 0.84]
# print("\nAfter loading state:")
# print("Epoch\tHybridLR\tMetric")
# for epoch, metric in enumerate(metrics, start=11):
#     scheduler_new.step()
#     scheduler_new.step_metric(metric)
#     lr_new = optimizer_new.param_groups[0]['lr']
#     print(f"{epoch}\t{lr_new:.6f}\t{metric:.2f}")

# %%
# Save the state
scheduler_state = hybrid_scheduler.state_dict()
optimizer_state = optimizer_hybrid.state_dict()

# Simulate training interruption and create a new optimizer and scheduler
optimizer_new = SGD([dummy_param], lr=0.1)
scheduler_new = LambdaLRWithReduceOnPlateau(
    optimizer_new,
    lr_lambda=lambda_func,
    mode='max',
    factor=0.1,
    patience=2,
    verbose=True
)

# Load the state
scheduler_new.load_state_dict(scheduler_state)
optimizer_new.load_state_dict(optimizer_state)

# Continue training
metrics = [0.85, 0.84, 0.84]
print("\nAfter loading state:")
print("Epoch\tLR\t\tMetric")
for epoch, metric in enumerate(metrics, start=11):
    scheduler_new.step()
    scheduler_new.step_metric(metric)
    lr = optimizer_new.param_groups[0]['lr']
    print(f"{epoch}\t{lr:.6f}\t{metric:.2f}")
# %%
