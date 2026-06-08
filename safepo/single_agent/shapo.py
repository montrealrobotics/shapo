import wandb
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable
import numpy as np
from safepo.common.model import ActorVCritic




def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, "No gradients were found in model parameters."
    return torch.cat(flat_params)


def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p, policy, fvp_obs)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"


def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, "No gradients were found in model parameters."
    return torch.cat(grads)


def fvp(
    params: torch.Tensor,
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
) -> torch.Tensor:
    policy.actor.zero_grad()
    current_distribution = policy.actor(fvp_obs)
    with torch.no_grad():
        old_distribution = policy.actor(fvp_obs)
    kl = torch.distributions.kl.kl_divergence(
        old_distribution, current_distribution
    ).mean()

    grads = torch.autograd.grad(kl, tuple(policy.actor.parameters()), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_p = (flat_grad_kl * params).sum()
    grads = torch.autograd.grad(
        kl_p,
        tuple(policy.actor.parameters()),
        retain_graph=False,
    )

    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    return flat_grad_grad_kl + params * 0.1


def compute_sam_gradients_critic(critic, data, target_values, rho=0.05, num_samples=10):
    """Compute Sharpness Aware Minimization gradients for critic.
    
    Args:
        critic: The critic network (reward or cost)
        data: Dictionary containing observations, risk values
        target_values: Target values for the critic
        rho: Perturbation radius for SAM
        
    Returns:
        sam_grads: Dictionary mapping parameter names to their SAM gradients
        perturbed_params: The perturbed parameters
    """
    # Store original parameters
    original_params = []
    for param in critic.parameters():
        if param.requires_grad:
            original_params.append(param.data.clone())
    
    # First compute the base loss and gradients
    critic.zero_grad()
    value_pred = critic(data["obs"])
    base_loss = nn.functional.mse_loss(value_pred, target_values)
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    
    # Get gradients and compute norm
    grad_norm = 0.0
    for param in critic.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    # Compute perturbation
    scale = rho / (grad_norm + 1e-12)
    perturbed_params = []
    for param in critic.parameters():
        if param.grad is None:
            continue
        e_w = param.grad * scale
        perturbed_params.append(e_w)
        param.data.add_(e_w)
    
    # Compute loss and gradients at perturbed point
    critic.zero_grad()
    value_pred = critic(data["obs"])
    perturbed_loss = nn.functional.mse_loss(value_pred, target_values)
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = {}
    for name, param in critic.named_parameters():
        if param.grad is not None:
            sam_grads[name] = param.grad.clone()
    
    # Restore original parameters
    for param, orig_param in zip(critic.parameters(), original_params):
        if param.requires_grad:
            param.data.copy_(orig_param)
    
    return sam_grads, perturbed_params, None, None, None




def compute_kl_constrained_perturbation(policy, data, step_direction, target_kl, max_search_steps=10):
    """Compute a perturbation that satisfies the KL constraint using line search.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        step_direction: The direction to perturb parameters
        target_kl: Target KL divergence constraint
        max_search_steps: Maximum number of line search steps
        
    Returns:
        step_frac: The step fraction that satisfies KL constraint
        final_kl: The final KL divergence achieved
        accepted_step: Whether a suitable step was found
    """
    theta_old = get_flat_params_from(policy.actor)
    
    # Store old distribution for KL computation
    with torch.no_grad():
        old_distribution = policy.actor(data["obs"])
    
    step_frac = 1.0
    final_kl = 0.0
    accepted_step = False
    
    for step in range(max_search_steps):
        # Compute perturbed parameters
        theta_perturbed = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_perturbed)
        
        # Compute KL divergence between old and perturbed policy
        with torch.no_grad():
            current_distribution = policy.actor(data["obs"])
            kl = torch.distributions.kl.kl_divergence(
                old_distribution, current_distribution
            ).mean().item()
        
        if kl <= target_kl:
            final_kl = kl
            accepted_step = True
            break
        else:
            step_frac *= 0.8
    else:
        # If no suitable step found, use a very small perturbation
        step_frac = 0.1
        theta_perturbed = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_perturbed)
        
        with torch.no_grad():
            current_distribution = policy.actor(data["obs"])
            final_kl = torch.distributions.kl.kl_divergence(
                old_distribution, current_distribution
            ).mean().item()
        accepted_step = True
    
    return step_frac, final_kl, accepted_step


def compute_natural_gradient_direction(fvp, policy, data, grads, target_kl, conjugate_gradient_iters=10):
    """Compute the natural gradient direction using conjugate gradients.
    
    Args:
        fvp: Fisher vector product function
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        grads: The policy gradients
        target_kl: Target KL divergence for step size computation
        
    Returns:
        step_direction: The natural gradient step direction
        x: The conjugate gradient solution
        xHx: The quadratic form x^T H x
    """
    x = conjugate_gradients(fvp, policy, data["fvp_obs"], grads, conjugate_gradient_iters)
    assert torch.isfinite(x).all(), "x is not finite"
    xHx = torch.dot(x, fvp(x, policy, data["fvp_obs"]))
    assert xHx.item() >= 0, "xHx is negative"
    
    # Initial step size based on TRPO
    alpha = torch.sqrt(2 * target_kl / (xHx + 1e-8))
    step_direction = x * alpha
    
    return step_direction, x, xHx


def log_gradient_statistics(grads, step_direction, x):
    """Log gradient statistics for debugging and monitoring.
    
    Args:
        grads: The original policy gradients
        step_direction: The natural gradient step direction
        x: The conjugate gradient solution
        logger: Logger instance for logging
    """
    grad_norm = torch.norm(grads)
    
    # Compute cosine similarity between natural gradient and original gradient
    cos_sim = torch.nn.functional.cosine_similarity(x.view(1,-1), grads.view(1,-1))
    
    # Compute effective rho as the norm of the step direction
    effective_rho = torch.norm(step_direction).item()

    # Calculate scale by projecting step_direction onto original gradient direction
    grad_direction = grads / (grad_norm + 1e-12)  # Normalize gradient
    scale_along_grad = torch.dot(step_direction, grad_direction)

    return cos_sim, effective_rho, scale_along_grad


def compute_shapo_gradients_actor(fvp, policy, data, advantage_lag, advantage_cost, advantage_reward, perturbation_target_kl=0.01, max_search_steps=10):
    """Compute Sharpness Aware Minimization gradients with KL constraint.
    
    Args:
        policy: The policy network
        data: Dictionary containing observations, actions, etc.
        advantage: Advantage values
        rho: Perturbation radius for SAM
        target_kl: Target KL divergence constraint
        max_search_steps: Maximum number of line search steps
        
    Returns:
        sam_grads: The gradients computed at the perturbed point
        perturbed_params: The perturbed parameters
    """

    theta_old = get_flat_params_from(policy.actor)
    
    # First compute the base loss and gradients with respect to cost.
    temp_distribution = policy.actor(data["obs"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    base_loss = -(ratio * advantage_lag).mean()
    
    # Compute gradients
    base_loss.backward(retain_graph=True)
    grads = get_flat_gradients_from(policy.actor)
    
    # Compute natural gradient direction
    step_direction, x, xHx = compute_natural_gradient_direction(fvp, policy, data, grads, perturbation_target_kl, max_search_steps)

    # Log gradient statistics
    cos_sim, effective_rho, scale_along_grad = log_gradient_statistics(grads, step_direction, x)

    # Find KL-constrained perturbation in the direction of increasing cost.
    step_frac, final_kl, accepted_step = compute_kl_constrained_perturbation(
        policy, data, step_direction, perturbation_target_kl, max_search_steps
    )
    
    # Store the final perturbation
    perturbed_params = step_frac * step_direction
    
    # Compute loss and gradients at perturbed point
    policy.actor.zero_grad()
    temp_distribution = policy.actor(data["obs"])
    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
    ratio = torch.exp(log_prob - data["log_prob"])
    perturbed_loss = -(ratio * advantage_lag).mean()
    perturbed_loss.backward()
    
    # Get gradients at perturbed point
    sam_grads = get_flat_gradients_from(policy.actor)
    
    # Restore original parameters
    set_param_values_to_model(policy.actor, theta_old)
    
    return sam_grads, perturbed_params, cos_sim, effective_rho, scale_along_grad
