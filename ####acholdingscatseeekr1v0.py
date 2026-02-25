"""
Pure Python 3.14 CATSEEKR0.0B Realistic 4B LLM + o1 Reasoning Chip
NO FILES • NO DEPENDENCIES • THREAD-SAFE GUI • CODEBASE CAT R1
"""

import math
import random
import sys
import time
import threading
import tkinter as tk
from tkinter import font

# =============================================================================
# 0. Pure Python Math Engine
# =============================================================================
def zeros(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]

def rand_matrix(rows, cols, std=0.02):
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def vec_add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def vec_mul_scalar(v, scalar):
    return [a * scalar for a in v]

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def mat_vec_mul(mat, vec):
    return [dot_product(row, vec) for row in mat]

def silu(v):
    res = []
    for x in v:
        try:
            res.append(x / (1.0 + math.exp(-x)))
        except OverflowError:
            res.append(0.0)
    return res

def softmax(v):
    if not v:
        return []
    max_val = max(v)
    exps = [math.exp(x - max_val) for x in v]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def rms_norm(v, weight, eps=1e-6):
    mean_sq = sum(x * x for x in v) / len(v)
    inv_std = 1.0 / math.sqrt(mean_sq + eps)
#
