import wandb
import numpy as np

sweep_configuration = {
    'method': 'grid',
    'name': 'star_path_graph m50 d400',
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
        'num_nodes': {'values':[50]},
        'network': {'values':['path_graph', 'star_graph']},
        "mixing": {'values':['lazy_metropolis']},
        # 'net_arg': {'values':[0.1]},
        # 'connectivity': {'values':[0.872]},
        "solver": {"values": ["cta"]},
        "radius_const": {'values': [1.01]},
        'constraint': {'values': ['lagrangian']},
        "lmda": {'values': [0.025]},
        'gamma': {'values': [0.000008, 0.00001, 0.000015, 0.00002, 0.000025, 0.00003]},
        'centerized_loss': {'values': [-4.541]},
        'max_iter': {'values': [1500000]}
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="distributed_deterministic")
wandb.agent(sweep_id)
