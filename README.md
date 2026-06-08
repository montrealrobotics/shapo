# SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration

We've implemented the SHAPO versions of three algorithms: CPO, PIDLag and CRPO. 

## Installation

For detailed installation instructions please see [Installation.md](Installation.md).

## Running SHAPO Algorithms

We provide scripts to run the SHAPO-enhanced algorithms (CPO, PIDLag, and CRPO) on supported environments.

### Quick Start

#### 1. SHAPO-CPO

Run SHAPO-CPO with default settings:
```bash
python safepo/single_agent/cpo.py --task SafetyPointGoal1-v0 \
    --use-shapo-actor-reward True \
    --use-shapo-actor-cost True \
    --perturbation-target-kl 0.00001 \
    --cost-limit 10.0 \
    --seed 0
```

#### 2. SHAPO-CRPO

Run SHAPO-CRPO with default settings:
```bash
python safepo/single_agent/crpo.py --task SafetyPointGoal1-v0 \
    --use-shapo-actor True \
    --perturbation-target-kl 0.00001 \
    --crpo-distance 0.0 \
    --crpo-lambda-c 1.0 \
    --cost-limit 10.0 \
    --seed 0
```

#### 3. SHAPO-PIDLag (PIDLag)

Run SHAPO-PIDLag with default settings:
```bash
python safepo/single_agent/trpo_pid.py --task SafetyPointGoal1-v0 \
    --use-shapo-actor True \
    --perturbation-target-kl 0.00001 \
    --cost-limit 10.0 \
    --seed 0
```
## Environments

The code supports all the environments shown in the paper: SafetyPointGoal1-v0, SafetyPointButton1-v0, Ant-v4 and Walker2d-v4.

### Important Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | str | `SafetyPointGoal1-v0` | Environment to run (e.g., SafetyPointGoal1-v0, SafetyCarGoal1-v0) |
| `--use-shapo-actor` | bool | `False` | Enable SHAPO for actor updates (CRPO, PIDLag) |
| `--use-shapo-actor-reward` | bool | `False` | Enable SHAPO for reward objective in actor (CPO only) |
| `--use-shapo-actor-cost` | bool | `False` | Enable SHAPO for cost objective in actor (CPO only) |
| `--sam-rho` | float | `0.05` | SAM perturbation radius for the critic |
| `--perturbation-target-kl` | float | `0.01` | Target KL divergence for SHAPO perturbation for the actor |
| `--use-sam-cost-critic` | bool | `False` | Enable SAM for cost critic updates |
| `--use-sam-reward-critic` | bool | `False` | Enable SAM for reward critic updates |
| `--cost-limit` | float | `10.0` | Cost constraint threshold |
| `--num-envs` | int | `5` | Number of parallel environments |
| `--seed` | int | `0` | Random seed for reproducibility |

**Note**: CPO computes two separate SHAPO gradients (one for reward, one for cost) using `--use-shapo-actor-reward` and `--use-shapo-actor-cost`. CRPO and PIDLag use a single SHAPO gradient with `--use-shapo-actor`.




This repository is a modification of the original [Safe-Policy-Optimization](https://github.com/PKU-Alignment/Safe-Policy-Optimization) repository developed by the PKU-Alignment team.

## Citation

If you use this code or the original Safe-Policy-Optimization repository in your research, please cite:

```bibtex
@article{ji2023safety,
  title={Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark},
  author={Ji, Jiaming and Zhang, Borong and Zhou, Jiayi and Pan, Xuehai and Huang, Weidong and Sun, Ruiyang and Geng, Yiran and Zhong, Yifan and Dai, Juntao and Yang, Yaodong},
  journal={arXiv preprint arXiv:2310.12567},
  year={2023}
}
```

## Original Repository

For the original implementation and documentation, please visit:
- GitHub: https://github.com/PKU-Alignment/Safe-Policy-Optimization
- Paper: https://arxiv.org/abs/2310.12567

## Acknowledgments

We thank the PKU-Alignment team for developing and maintaining the original Safe-Policy-Optimization benchmark, which serves as the foundation for this work.
