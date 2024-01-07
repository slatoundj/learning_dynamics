from math import (floor, exp)
import numpy as np
import torch
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0" # For amd gpu with rocm

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(device)


def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0
    

def fermi(beta, fx, fy):
    return 1 / (1 + exp(beta * (fx - fy)))


def V(i_config, Zr):
    ir, ip = i_config
    return int(ir + ip + ip * Zr)


def rec_V(x, Zr):
    ip = floor(x/Zr)
    ir = x - ip * Zr - ip
    while ir < 0:
        ip -= 1
        ir = x - ip * Zr - ip
    return (int(ir), int(ip))   


def null_cpu(a, rtol=1e-8):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:].T.copy()


def null(a, rtol=1e-6):
    A = torch.Tensor(a).to(device)
    U, S, Vh = torch.linalg.svd(A)
    v_cpu = Vh.cpu()
    v = v_cpu.numpy()
    rank = (S > rtol*S[0]).sum()
    return rank, v[rank:].T.copy()


def distrib_to_matrix(distribution, x_size, y_size):
    numStates = x_size * y_size
    matrix = np.zeros((y_size, x_size))
    for i in range(numStates):
        ir, ip = rec_V(i, x_size-1)
        matrix[ip, ir] = distribution[i]
    return matrix