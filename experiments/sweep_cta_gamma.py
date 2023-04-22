import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'deterministic graph',
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
        'network': {'values':['complete_graph','star_graph', 'path_graph']},
        "mixing": {'values':['metropolis_hasting']},
        # 'net_arg': {'values':[0.1]},
        # 'connectivity': {'values':[0.872]},
        "radius_const": {'values': [1.01]},
        'constraint': {'values': ['lagrangian']},
        "lmda": {'values': [0.27]},
        'gamma': {'values': [2e-3, 4e-3, 8e-3]},
        'max_iter': {'values': [10000]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_deterministic")
wandb.agent(sweep_id)