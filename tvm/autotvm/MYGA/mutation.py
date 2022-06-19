import random

from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from enum import Enum
import numpy as np

class GAMutation:
    @staticmethod
    def rm(gene, dims, rate):
        for j, dim in enumerate(dims):
            if np.random.random() < rate:
                gene[j] = np.random.randint(dim)
        return gene


    @staticmethod
    def cm(gene, dims, rate):
        l = len(gene)
        idx1 = np.random.randint(l)
        idx2 = np.random.randint(l)
        while idx2 == idx1:
            idx2 = np.random.randint(l)
        gene[idx1] , gene[idx2] = gene[idx2] , gene[idx1]
        gene[idx1] %= dims[idx1]
        gene[idx2] %= dims[idx2]
        return gene


    @staticmethod
    def sm(gene, dims, rate):
        l = len(dims)
        idx1 = np.random.randint(l)
        idx2 = np.random.randint(l)
        while idx2 == idx1:
            idx2 = np.random.randint(l)
        if idx1 > idx2 :
            idx1, idx2 = idx2, idx1
        gene[idx1:idx2 + 1] = gene[idx2::-1]
        for i in range(idx1, idx2 + 1):
            gene[i] %= dims[i]
        return gene

    @staticmethod
    def im(gene, dims, rate):
        l = len(gene)
        idx1 = np.random.randint(l)
        idx2 = np.random.randint(l)
        while idx2 == idx1:
            idx2 = np.random.randint(l)
        if idx1 > idx2 :
            idx1, idx2 = idx2, idx1
        tmp = gene[idx1:idx2+1]
        random.shuffle(tmp)
        gene[idx1:idx2 + 1] = tmp
        for i in range(idx1, idx2+1):
            gene[i] %= dims[i]
        return gene

    @staticmethod
    def Rm(gene, dims, rate):
        gene = []
        for i in range(len(dims)):
            gene.append(np.random.randint(dims[i]))
        return gene


class EnumMutation(Enum):
    rm = 0 # random mutation
    cm = 1 # crossover mutation
    sm = 2 # swap mutation
    im = 3 # inversion mutation
    Rm = 4

    @staticmethod
    def randomMutation():
        l = len(EnumMutation)
        idx = np.random.randint(l)
        mutation = EnumMutation(idx)
        return mutation