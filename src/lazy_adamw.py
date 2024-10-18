#%%
from typing import Optional
import torch
from torch import Tensor
import torch.optim._functional as F
from torch.optim import AdamW
from torch.optim.adamw import adamw
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable, _get_scalar_dtype, _get_fused_kernels_supported_devices


class LazyAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        l1_coeff=0.0,
        min_norm=None,
        max_norm=None,
        amsgrad=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= l1_coeff:
            raise ValueError("Invalid l1_coeff value: {}".format(l1_coeff))
        if min_norm is not None and not 0.0 <= min_norm:
            raise ValueError(f"Invalid min_norm value: {min_norm}")
        if max_norm is not None and not 0.0 <= max_norm:
            raise ValueError(f"Invalid max_norm value: {max_norm}")
        if min_norm is not None and max_norm is not None and min_norm > max_norm:
            raise ValueError(f"min_norm should be less than or equal to max_norm, got min_norm={min_norm} and max_norm={max_norm}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            l1_coeff=l1_coeff,
            min_norm=min_norm,
            max_norm=max_norm,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            fused_supported_devices = _get_fused_kernels_supported_devices()
            if not all(
                p.device.type in fused_supported_devices and
                torch.is_floating_point(p)
                for pg in self.param_groups for p in pg['params']
            ):
                raise RuntimeError("`fused=True` requires all the params to be floating point Tensors of "
                                   f"supported devices: {fused_supported_devices}.")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if not all(
                p.is_cuda and torch.is_floating_point(p)
                for pg in self.param_groups for p in pg['params']
            ):
                raise RuntimeError("`fused=True` requires all the params to be CUDA, floating point Tensor")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        max_exp_avg_sqs=None,
        parse_sparse=False
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            is_sparse = p.grad.is_sparse
            if parse_sparse != is_sparse:
                continue

            params_with_grad.append(p)
      
            # if p.grad.is_sparse:
            #     raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                if is_sparse:
                    state['step'] = torch.tensor(0.0)  # Initialize step as tensor
                else:
                    state["step"] = (
                        torch.zeros((), dtype=torch.float, device=p.device)
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0)
                    )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad and not is_sparse:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if amsgrad and not is_sparse:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            # For dense params, step is incremnted inside the adamw function
            # but for sparse params, step is incremented here
            if is_sparse:
                state['step'] += 1
                weight_decay = group['weight_decay']

                # Apply weight decay (L2 regularization) to sparse embeddings
                if weight_decay != 0:
                    # p.data.add_(p.data, alpha=-group['lr'] * weight_decay)

                    p.grad = p.grad.coalesce()  # Ensure the sparse gradient is in coalesced form
                    grad_indices = p.grad.indices()

                    # print(f"LazyAdamW: Updated {grad_indices.size(1)} embeddings")

                    # For each accessed index, apply weight decay
                    for i in range(grad_indices.size(1)):
                        index = grad_indices[0, i]
                        # p.data[index] -= weight_decay * p.data[index]
                        p.data[index].add_(p.data[index], alpha=-group['lr'] * weight_decay)

            state_steps.append(state["step"])


    def _apply_norm_constraints(self, group):
        # Apply min_norm and max_norm weight correction
        with torch.no_grad():
            min_norm = group.get('min_norm', None)
            max_norm = group.get('max_norm', None)
            if min_norm is not None or max_norm is not None:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    assert p.dim() == 2, "Weight norm correction is only supported for 2D parameters"
                    if p.grad.is_sparse:
                        # For sparse parameters, adjust only the accessed embeddings
                        p.grad = p.grad.coalesce()
                        indices = p.grad.indices()[0]  # Get indices as a 1D tensor
                        param_vectors = p.data[indices]
                        param_norms = param_vectors.norm(p=2, dim=1)
                        desired_norms = param_norms.clone()
                        if min_norm is not None:
                            desired_norms = torch.max(desired_norms, torch.full_like(desired_norms, min_norm))
                        if max_norm is not None:
                            desired_norms = torch.min(desired_norms, torch.full_like(desired_norms, max_norm))
                        scales = desired_norms / (param_norms + 1e-7)
                        p.data[indices] = param_vectors * scales.unsqueeze(1)
                    else:
                        # For dense parameters, adjust the entire parameter tensor
                        param_norms = p.data.norm(p=2, dim=1, keepdim=True)
                        desired_norms = param_norms.clone()
                    
                        desired_norms = torch.clamp(param_norms, min=min_norm, max=max_norm)
                        scales = desired_norms / (param_norms + 1e-7)
                        p.data = p.data * scales

    def _apply_l1_reg(self, group):
        # Apply L1 regularization **after** the AdamW or Sparse Adam updates
        l1_coeff = group['l1_coeff']
        if l1_coeff == 0:
            return

        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                p.grad = p.grad.coalesce()  # Ensure the sparse gradient is coalesced
                grad_indices = p.grad.indices()

                # Apply L1 regularization to sparse parameters (only for accessed embeddings)
                for i in range(grad_indices.size(1)):
                    index = grad_indices[0, i]
                    p.data[index] -= l1_coeff * torch.sign(p.data[index])

            else:
                # Apply L1 regularization to dense parameters (all parameters)
                p.data -= l1_coeff * torch.sign(p.data)



    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            dense_params_with_grad, sparse_params_with_grad = [], []
            dense_grads, sparse_grads = [], []
            dense_exp_avgs, sparse_exp_avgs = [], []
            dense_exp_avg_sqs, sparse_exp_avg_sqs = [], []
            dense_max_exp_avg_sqs = []
            dense_state_steps, sparse_state_steps = [], []

            amsgrad = group["amsgrad"]
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            self._init_group(
                group,
                dense_params_with_grad,
                dense_grads,
                amsgrad,
                dense_exp_avgs,
                dense_exp_avg_sqs,
                dense_state_steps,
                dense_max_exp_avg_sqs,
                parse_sparse=False
            )

            self._init_group(
                group,
                sparse_params_with_grad,
                sparse_grads,
                amsgrad,
                sparse_exp_avgs,
                sparse_exp_avg_sqs,
                sparse_state_steps,
                None,
                parse_sparse=True
            )

            if len(dense_params_with_grad) > 0:
                adamw(
                    dense_params_with_grad,
                    dense_grads,
                    dense_exp_avgs,
                    dense_exp_avg_sqs,
                    dense_max_exp_avg_sqs,
                    dense_state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=weight_decay,
                    eps=group["eps"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                )

            if len(sparse_params_with_grad) > 0:
                F.sparse_adam(
                    sparse_params_with_grad,
                    sparse_grads,
                    sparse_exp_avgs,
                    sparse_exp_avg_sqs,
                    sparse_state_steps, # This is now a list of tensors
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    eps=group['eps'],
                    maximize=False,
                )

            # Apply L1 regularization
            self._apply_l1_reg(group)

            # Renorm the embeddings to be within a range of min_norm and max_norm
            self._apply_norm_constraints(group)

        return loss

# %%
