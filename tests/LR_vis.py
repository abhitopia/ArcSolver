#%%
import math
import matplotlib.pyplot as plt

def noam_schedule(step, warmup_steps, decay_steps, min_lr_scale=0.1):
    """
    Computes the learning rate at a given step based on the adjusted Noam scheduler.

    Args:
        step (int): Current training step.
        warmup_steps (int): Number of warmup steps.
        decay_steps (int): Number of steps for cosine decay.
        min_lr_scale (float): Scaling factor for the minimum learning rate.

    Returns:
        float: Learning rate at the current step.
    """
    max_lr = 1.0
    min_lr = max_lr * min_lr_scale
    step_until_decay = warmup_steps + decay_steps

    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    elif step <= step_until_decay:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (step_until_decay - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    else:
        # Inverse square root decay
        return min_lr * (step_until_decay / step) ** 0.5


def plot_lr_schedule(warmup_steps, decay_steps, min_lr_scale=0.0, total_steps=40000):
    """
    Plots the learning rate schedule over a specified number of steps.

    Args:
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Number of steps for cosine decay.
        min_lr_scale (float, optional): Scaling factor for the minimum learning rate. Defaults to 0.2.
        total_steps (int, optional): Total number of training steps to plot. Defaults to 40000.
    """
    steps = list(range(1, total_steps + 1))
    lrs = [noam_schedule(step, warmup_steps, decay_steps, min_lr_scale) for step in steps]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, lrs, label='Learning Rate')
    plt.axvline(x=warmup_steps, color='orange', linestyle='--', label='End of Warmup')
    plt.axvline(x=warmup_steps + decay_steps, color='green', linestyle='--', label='End of Cosine Decay')
    plt.title('Adjusted Noam Learning Rate Schedule')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Parameters
warmup_steps = 10000      # Number of warmup steps
decay_steps = 30000      # Number of steps for cosine decay
min_lr_scale = 0.1      # Minimum learning rate scaling factor
total_steps = 100000      # Total number of steps to visualize

# Plot the learning rate schedule
plot_lr_schedule(warmup_steps, decay_steps, min_lr_scale, total_steps)

# %%
def plot_multiple_schedulers(warmup_steps, max_steps, min_lr_scale, total_steps=40000):
    """
    Plots multiple learning rate schedules for comparison.

    Args:
        warmup_steps (int): Number of warmup steps.
        max_steps (int): Number of steps for cosine decay.
        min_lr_scale (float): Scaling factor for the minimum learning rate.
        total_steps (int, optional): Total number of training steps to plot. Defaults to 40000.
    """
    steps = list(range(1, total_steps + 1))
    
    # Scheduler 1: Original Adjusted Noam
    lrs_noam = [adjusted_noam_schedule(step, warmup_steps, max_steps, min_lr_scale) for step in steps]
    
    # Scheduler 2: Flat after decay
    def no_flat_schedule(step):
        return adjusted_noam_schedule(step, warmup_steps, max_steps, min_lr_scale=0.2) if step <= max_steps else 0.1
    
    lrs_flat = [no_flat_schedule(step) for step in steps]
    
    # Scheduler 3: Exponential Decay
    def exponential_decay_schedule(step, warmup_steps, max_steps, min_lr_scale=0.2, decay_rate=0.0001):
        max_lr = 1.0
        min_lr = max_lr * min_lr_scale
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        elif step <= max_steps:
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)
        else:
            return max(min_lr, min_lr * math.exp(-decay_rate * (step - max_steps)))
    
    lrs_exp = [exponential_decay_schedule(step, warmup_steps, max_steps, min_lr_scale, decay_rate=0.0001) for step in steps]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(steps, lrs_noam, label='Adjusted Noam Schedule (Inverse Sqrt Decay)')
    plt.plot(steps, lrs_flat, label='Flat Schedule After Decay', linestyle='--')
    plt.plot(steps, lrs_exp, label='Exponential Decay Schedule', linestyle='-.')
    plt.axvline(x=warmup_steps, color='orange', linestyle='--', label='End of Warmup')
    plt.axvline(x=max_steps, color='green', linestyle='--', label='End of Cosine Decay')
    plt.title('Comparison of Learning Rate Schedules')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot multiple schedulers
plot_multiple_schedulers(warmup_steps, max_steps, min_lr_scale, total_steps)

# %%
