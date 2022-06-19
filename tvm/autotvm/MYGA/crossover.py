from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from enum import Enum
import math
import numpy as np

class GACrossover:
    @staticmethod
    def spc(g1, g2, size):
        if size >= 1:
            point = np.random.randint(size)
            tmp_gene = g1[:point] + g2[point:]
            return tmp_gene
        return g1

    @staticmethod
    def tpc(g1, g2, size):
        if size >= 2:
            p1 = np.random.randint(size)
            p2 = np.random.randint(size)
            while p1 == p2 :
                p2 = np.random.randint(size)
            p1 , p2 = min(p1,p2), max(p1,p2)
            tmp_gene = g1[:p1] + g2[p1:p2] + g1[p2:]
            return tmp_gene
        else:
            return GACrossover.spc(g1, g2 ,size)

    @staticmethod
    def mpc(g1, g2, size):
        if size >= 3:
            p1 = np.random.randint(size)
            p2 = np.random.randint(size)
            while p1 == p2 :
                p2 = np.random.randint(size)
            p3 = np.random.randint(size)
            while p3 == p1 or p3 == p2:
                p3 = np.random.randint(size)
            p = [p1, p2, p3]
            p = np.array(p)
            p = np.sort(p)
            tmp_gene = g1[:p[0]] + g2[p[0]:p[1]] + g1[p[1]:p[2]] + g2[p[2]:]
            return tmp_gene
        else:
            return GACrossover.tpc(g1, g2, size)

    @staticmethod
    def ec(g1, g2, size):
        gene = []
        for i in range(size):
            if np.random.randint(1) == 0:
                gene.append(g1[i])
            else:
                gene.append(g2[i])
        return gene

    @staticmethod
    def etpc(g1, g2, size):
        if size >= 2:
            p1 = np.random.randint(size)
            p2 = np.random.randint(size)
            while p1 == p2:
                p2 = np.random.randint(size)
            p1, p2 = min(p1, p2), max(p1, p2)
            judge = np.random.randint(2)
            if judge == 0:
                tmp_gene = g1[:p1] + g2[p1:]
            elif judge == 1:
                tmp_gene = g1[:p1] + g2[p1:p2] + g1[p2:]
            else:
                tmp_gene = g1[:p2] + g2[p2:]
            return tmp_gene
        else:
            return GACrossover.spc(g1, g2 ,size)

    @staticmethod
    def dc(genes, size):
        gene = []
        l = len(genes)
        for i in range(size):
            g = genes[np.random.randint(l)]
            gene.append(g[i])
        return gene

    @staticmethod
    def ac(g1, g2, size):
        gene = []
        for i in range(size):
            gene.append( (g1[i] + g2[i]) // 2)
        return gene

    @staticmethod
    def hc(g1, g2, size, dims, space):
        radio = 1.2
        v1 = knob2point(g1,dims)
        v2 = knob2point(g2,dims)
        v1, v2 = max(v1, v2) , min(v1,v2)
        v = int(v2 + radio * (v1 - v2))
        v = v % space
        gene = point2knob(v,dims)
        return gene


class EnumCrossover(Enum):
    spc  = 0 # single point cross
    tpc  = 1 # two point cross
    mpc  = 2 # multipoint cross
    ec   = 3 # even cross
    etpc = 4 # Even two-point cross
    dc   = 5 # Discrete cross
    ac   = 6 # Arithmetic cross
    hc   = 7 # Heuristic cross

    @staticmethod
    def randomCrossover():
        l = len(EnumCrossover)
        idx = np.random.randint(l)
        crossover = EnumCrossover(idx)
        return crossover