import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'centralized baseline',
    'entity': 'yaoji',
    'program': 'main.py',
    'metric': {
        'goal': 'minimize', 
        'name': 'iter_loss (log scale)'
    },
    'parameters': {
        'num_samples': {'values': [200]},
        'num_dimensions':{'values':[400]},
        'sparsity':{'values':[5]},
        'num_nodes': {'values':[1]},
        'network': {'values':['complete_graph']},
        "mixing": {'values':['lazy_metropolis']},
        # 'net_arg': {'values':[0.1]},
        # 'connectivity': {'values':[0.872]},
        "radius_const": {'values': [1.01]},
        'constraint': {'values': ['lagrangian']},
        "lmda": {'values': [0.027,0.030,0.033,0.0036]},
        'gamma': {'values': [0.005, 0.01, 0.02, 0.05]},
        'max_iter': {'values': [1000]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_deterministic")
wandb.agent(sweep_id)
