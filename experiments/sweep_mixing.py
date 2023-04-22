import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'mixing sweep',
    'entity': 'yaoji',
    'program': 'main.py',
    'metric': {
        'goal': 'minimize', 
        'name': 'iter_loss (log scale)'
    },
    'parameters': {
        'num_samples': {'values': [220]},
        'num_dimensions':{'values':[360]},
        'sparsity':{'values':[5]},
        'num_nodes': {'values':[5]},
        'network': {'values':['complete_graph']},
        "mixing": {'values':['metropolis_hasting', 'lazy_metropolis']},
        # 'net_arg': {'values':[0.1]},
        # 'connectivity': {'values':[0.872]},
        "radius_const": {'values': [1.01]},
        'constraint': {'values': ['lagrangian']},
        "lmda": {'values': [0.033]},
        'gamma': {'values': [8e-3]},
        'max_iter': {'values': [5000]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_deterministic")
wandb.agent(sweep_id)