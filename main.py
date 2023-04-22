import time
import argparse
import pickle
import os
from generator import Generator
from network import ExistedNetwork
from solver import *
from scheduler import *
import wandb


# configuration
parser = argparse.ArgumentParser(description='distributed optimization')
parser.add_argument('--storing_filepath', default='', type=str, help='storing_file_path')
parser.add_argument('--storing_filename', default='', type=str, help='storing_file_name')
## data
parser.add_argument("-N", "--num_samples", type=int)
parser.add_argument("-d", "--num_dimensions", type=int)
parser.add_argument("-s", "--sparsity", type=int)
parser.add_argument("-k", "--k", type=float, default=0.25)
parser.add_argument("--sigma", type=float, default=0.5)
## network
parser.add_argument("--network", default='fast_gnp_random_graph', choices=['fast_gnp_random_graph', 'random_geometric_graph', 'complete_graph','star_graph', 'path_graph'])
parser.add_argument("--mixing", default='metropolis_hasting')
parser.add_argument("-m", "--num_nodes", default=5, type=int)
parser.add_argument("-p", "--probability", default=1, type=float)
parser.add_argument("--rand_geo_radius", default=1, type=float)
parser.add_argument("-rho", "--connectivity", default=0, type=float)
## solver
parser.add_argument("--solver", default='cta')
parser.add_argument("--radius_const", type=float, default=1.01)
parser.add_argument("--constraint", choices=("lagrangian", "projected"))
parser.add_argument("--lmda", default=1, type=float)
parser.add_argument("--centerized_loss", type=float, default=-1e10)
parser.add_argument("--max_iter", type=int, default=10000)
# parser.add_argument("--iter_type", choices=("lagrangian", "projected"))
parser.add_argument("--gamma", type=float)
parser.add_argument("--communication", type=int, default=1)
parser.add_argument("--local_computation", type=int, default=1)
parser.add_argument("--scheduler", choices=("const","diminish"), default="const")

## others
parser.add_argument("--seed", type=int, default=8989)
args = parser.parse_args()


def main():
    # register wandb
    wandb.init(project="distributed_deterministic",
                   entity="yaoji",
                   config=vars(args))
    wandb.run.log_code()
    # preprocessing data

    ## processing data"))
    generator = Generator(args.num_samples, args.num_nodes, args.num_dimensions, args.sparsity, args.k, args.sigma, args.seed)
    ## processing network
    G = ExistedNetwork(args.network, args.num_nodes, args.mixing, args.probability, args.connectivity, seed=args.seed)
    ## process stepsize
    w = G.w

    if args.scheduler == "const":
        gamma = ConstScheduler(args.gamma)
    elif args.scheduler == "diminish":
        gamma = DiminishScheduler(args.gamma)

    # solver run
    NetworkGD(generator, w, gamma, args).fit()
    # elif args.solver == 'pcta':
    #     print("projected_cta")
    #     solver = PCTA(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    # elif args.solver == 'patc':
    #     print("projected_atc")
    #     solver = PATC(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    # elif args.solver == 'netlasso':
    #     print("netlasso")
    #     solver = NetLasso(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    # elif args.solver == 'pgextra':
    #     print("pgextra")
    #     solver = PGExtra(args.max_iter, gamma, r, w, args.communication, args.local_computation)
    # elif args.solver == 'primaldual':
    #     if args.betascheduler == "const":
    #         beta = ConstScheduler(args.beta)
    #     elif args.betascheduler == "diminish":
    #         beta = DiminishScheduler(args.beta)
    #     print("primal_dual")
    #     solver = PrimalDual(args.max_iter, gamma, beta, r, w, args.communication, args.local_computation)
    # else:
    #     raise NotImplementedError("solver mode currently only support centralized or distributed")



if __name__ == "__main__":
    main()
