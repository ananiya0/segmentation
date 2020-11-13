import sys

import torch
from data_loader import get_data_loaders
from mpi4py import MPI
from network import gen_dist_net
