# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Lagrange."""

from __future__ import annotations

from collections import deque
import math

import torch


def is_valid_number(x) -> bool:
    """Check if a value is a valid finite number (not nan or inf)."""
    if not isinstance(x, (int, float)):
        return False
    if isinstance(x, float):
        return not (math.isnan(x) or math.isinf(x))
    return True


class Lagrange:
    """Lagrange multiplier for constrained optimization.
    
    Args:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier

    Attributes:
        cost_limit: the cost limit  
        lagrangian_multiplier_lr: the learning rate of the lagrangian multiplier
        lagrangian_upper_bound: the upper bound of the lagrangian multiplier
        _lagrangian_multiplier: the lagrangian multiplier
        lambda_range_projection: the projection function of the lagrangian multiplier
        lambda_optimizer: the optimizer of the lagrangian multiplier    
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lagrangian_multiplier_lr: float,
        lagrangian_upper_bound: float | None = None,
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        self.cost_limit: float = cost_limit
        self.lagrangian_multiplier_lr: float = lagrangian_multiplier_lr
        self.lagrangian_upper_bound: float | None = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.0)
        self._lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        self.lambda_optimizer: torch.optim.Optimizer = torch.optim.Adam(
            [
                self._lagrangian_multiplier,
            ],
            lr=lagrangian_multiplier_lr,
        )

    @property
    def lagrangian_multiplier(self) -> torch.Tensor:
        """The lagrangian multiplier.
        
        Returns:
            the lagrangian multiplier
        """
        return self.lambda_range_projection(self._lagrangian_multiplier).detach().item()

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        """Compute the loss of the lagrangian multiplier.
        
        Args:
            mean_ep_cost: the mean episode cost
            
        Returns:
            the loss of the lagrangian multiplier
        """
        return -self._lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, Jc: float) -> None:
        """Update the lagrangian multiplier.
        
        Args:
            Jc: the mean episode cost
            
        Returns:
            the loss of the lagrangian multiplier
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(Jc)
        
        # Add debugging information
        constraint_violation = Jc - self.cost_limit
        current_lambda = self._lagrangian_multiplier.item()
        
        lambda_loss.backward()
        
        # Check for gradient issues
        if self._lagrangian_multiplier.grad is not None:
            grad_norm = self._lagrangian_multiplier.grad.norm().item()
            if grad_norm > 10.0:  # Clip large gradients
                self._lagrangian_multiplier.grad.clamp_(-10.0, 10.0)
        
        self.lambda_optimizer.step()
        
        # More careful clamping - only clamp to upper bound if specified
        if self.lagrangian_upper_bound is not None:
            self._lagrangian_multiplier.data.clamp_(
                0.0,
                self.lagrangian_upper_bound,
            )
        else:
            # Only ensure non-negativity, don't clamp to 0 aggressively
            self._lagrangian_multiplier.data.clamp_(min=0.0)
        
        # Debug output (you can remove this in production)
        new_lambda = self._lagrangian_multiplier.item()
        if new_lambda == 0.0 and constraint_violation > 0.0:
            print(f"WARNING: Lambda became 0 despite constraint violation!")
            print(f"  Cost: {Jc:.4f}, Limit: {self.cost_limit:.4f}, Violation: {constraint_violation:.4f}")
            print(f"  Old lambda: {current_lambda:.6f}, New lambda: {new_lambda:.6f}")
            print(f"  Loss: {lambda_loss.item():.6f}, Grad norm: {grad_norm if 'grad_norm' in locals() else 'N/A'}")

    def get_debug_info(self, current_cost: float) -> dict:
        """Get debug information about the current state of the Lagrange multiplier.
        
        Args:
            current_cost: the current episode cost
            
        Returns:
            dictionary containing debug information
        """
        constraint_violation = current_cost - self.cost_limit
        lambda_value = self.lagrangian_multiplier
        
        return {
            'cost': current_cost,
            'cost_limit': self.cost_limit,
            'constraint_violation': constraint_violation,
            'lambda_value': lambda_value,
            'lambda_parameter': self._lagrangian_multiplier.item(),
            'lambda_upper_bound': self.lagrangian_upper_bound,
            'lambda_lr': self.lagrangian_multiplier_lr,
            'should_increase': constraint_violation > 0.0 and lambda_value == 0.0,
        }


class PIDLagrangian:

    """PID Lagrangian multiplier for constrained optimization.

    Args:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        pid_kp: the proportional gain of the PID controller
        pid_ki: the integral gain of the PID controller
        pid_kd: the derivative gain of the PID controller
        pid_d_delay: the delay of the derivative term
        pid_delta_p_ema_alpha: the exponential moving average alpha of the delta_p
        pid_delta_d_ema_alpha: the exponential moving average alpha of the delta_d
        sum_norm: whether to normalize the sum of the PID output
        diff_norm: whether to normalize the difference of the PID output
        penalty_max: the maximum value of the penalty

    Attributes:
        cost_limit: the cost limit
        lagrangian_multiplier_init: the initial value of the lagrangian multiplier
        pid_kp: the proportional gain of the PID controller
        pid_ki: the integral gain of the PID controller
        pid_kd: the derivative gain of the PID controller
        pid_d_delay: the delay of the derivative term
        pid_delta_p_ema_alpha: the exponential moving average alpha of the delta_p
        pid_delta_d_ema_alpha: the exponential moving average alpha of the delta_d
        sum_norm: whether to normalize the sum of the PID output
        diff_norm: whether to normalize the difference of the PID output
        penalty_max: the maximum value of the penalty

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: `CPPOPID <https://arxiv.org/abs/2007.03964>`_
    """
    
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float=0.005,
        pid_kp: float=0.1,
        pid_ki: float=0.01,
        pid_kd: float=0.01,
        pid_d_delay: int=10,
        pid_delta_p_ema_alpha: float=0.95,
        pid_delta_d_ema_alpha: float=0.95,
        sum_norm: bool=True,
        diff_norm: bool=False,
        penalty_max: int=100.0,
    ) -> None:
        """Initialize an instance of :class:`PIDLagrangian`."""
        self._pid_kp: float = pid_kp
        self._pid_ki: float = pid_ki
        self._pid_kd: float = pid_kd
        self._pid_d_delay = pid_d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._pid_i: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self._pid_d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._cost_penalty: float = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        return self._cost_penalty

    def update_lagrange_multiplier(self, ep_cost_avg: float) -> None:
        # Handle nan input - this is the root cause of the issue
        if not is_valid_number(ep_cost_avg):
            print(f"WARNING: Invalid ep_cost_avg received: {ep_cost_avg}, using previous penalty value")
            # Keep the previous penalty value and don't update anything
            return
        
        delta = float(ep_cost_avg - self._cost_limit)
        
        # Store previous values for debugging
        prev_pid_i = self._pid_i
        prev_cost_penalty = self._cost_penalty
        
        # Update integral term - be more careful about clamping
        integral_update = delta * self._pid_ki
        self._pid_i += integral_update
        
        # Ensure integral term doesn't become nan
        if not is_valid_number(self._pid_i):
            print(f"WARNING: pid_i became invalid: {self._pid_i}, resetting to 0")
            self._pid_i = 0.0
        
        # Only clamp to 0 if the integral term becomes negative AND we're not violating constraints
        if self._pid_i < 0.0 and delta <= 0.0:
            self._pid_i = 0.0
        
        if self._diff_norm:
            self._pid_i = max(0.0, min(1.0, self._pid_i))
        
        a_p = self._pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        
        # Ensure delta_p doesn't become nan
        if not is_valid_number(self._delta_p):
            print(f"WARNING: delta_p became invalid: {self._delta_p}, resetting to 0")
            self._delta_p = 0.0
        
        a_d = self._pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        
        # Ensure cost_d doesn't become nan
        if not is_valid_number(self._cost_d):
            print(f"WARNING: cost_d became invalid: {self._cost_d}, resetting to 0")
            self._cost_d = 0.0
        
        pid_d = max(0.0, self._cost_d - self._cost_ds[0])
        
        # Ensure pid_d doesn't become nan
        if not is_valid_number(pid_d):
            print(f"WARNING: pid_d became invalid: {pid_d}, resetting to 0")
            pid_d = 0.0
            
        pid_o = self._pid_kp * self._delta_p + self._pid_i + self._pid_kd * pid_d

        # Check for nan in PID output and handle gracefully
        if not is_valid_number(pid_o):
            print(f"WARNING: PID output is nan/inf: {pid_o}, using previous penalty value")
            # Keep the previous penalty value
            return

        print(f"PID output: {pid_o:.6f}, Final penalty: {self._cost_penalty:.6f}, Delta: {delta:.6f}, delta_p: {self._delta_p:.6f}, cost_d: {self._cost_d:.6f}, cost_ds: {self._cost_ds[0]:.6f}, pid_d: {pid_d:.6f}, pid_i: {self._pid_i:.6f}, pid_kp: {self._pid_kp:.6f}, pid_ki: {self._pid_ki:.6f}, pid_kd: {self._pid_kd:.6f}")
        
        # More careful clamping of the final penalty
        if pid_o < 0.0 and delta > 0.0:
            # If we have a constraint violation, don't clamp to 0
            self._cost_penalty = 0.001  # Small positive value instead of 0
        else:
            self._cost_penalty = max(0.0, pid_o)
        
        if self._diff_norm:
            self._cost_penalty = min(1.0, self._cost_penalty)
        if not (self._diff_norm or self._sum_norm):
            self._cost_penalty = min(self._cost_penalty, self._penalty_max)
        
        # Only append valid cost_d values to the deque
        if is_valid_number(self._cost_d):
            self._cost_ds.append(self._cost_d)
        else:
            print(f"WARNING: Not appending invalid cost_d {self._cost_d} to deque")
        
        # Debug output (you can remove this in production)
        if self._cost_penalty == 0.0 and delta > 0.0:
            print(f"WARNING: PID Lambda became 0 despite constraint violation!")
            print(f"  Cost: {ep_cost_avg:.4f}, Limit: {self._cost_limit:.4f}, Delta: {delta:.4f}")
            print(f"  PID components - P: {self._pid_kp * self._delta_p:.6f}, I: {self._pid_i:.6f}, D: {self._pid_kd * pid_d:.6f}")
            print(f"  PID output: {pid_o:.6f}, Final penalty: {self._cost_penalty:.6f}")
            print(f"  Previous penalty: {prev_cost_penalty:.6f}, Previous I: {prev_pid_i:.6f}")

    def get_debug_info(self, current_cost: float) -> dict:
        """Get debug information about the current state of the PID Lagrange multiplier.
        
        Args:
            current_cost: the current episode cost
            
        Returns:
            dictionary containing debug information
        """
        delta = current_cost - self._cost_limit
        
        return {
            'cost': current_cost,
            'cost_limit': self._cost_limit,
            'constraint_violation': delta,
            'lambda_value': self._cost_penalty,
            'pid_i': self._pid_i,
            'pid_kp': self._pid_kp,
            'pid_ki': self._pid_ki,
            'pid_kd': self._pid_kd,
            'delta_p': self._delta_p,
            'cost_d': self._cost_d,
            'sum_norm': self._sum_norm,
            'diff_norm': self._diff_norm,
            'penalty_max': self._penalty_max,
            'should_increase': delta > 0.0 and self._cost_penalty == 0.0,
        }
