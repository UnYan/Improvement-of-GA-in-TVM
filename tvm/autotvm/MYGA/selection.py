import numpy as np
import geatpy
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from enum import Enum

class GASelection:
    @staticmethod
    def dup(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1 ,p2 = geatpy.dup(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def ecs(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1 ,p2 = geatpy.ecs(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def etour(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1 ,p2 = geatpy.etour(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def otos(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        tmp_scores = tmp_scores[: (len(tmp_scores) // 2) * 2]
        p1 ,p2 = geatpy.otos(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def rcs(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        tmp = geatpy.rcs(tmp_scores)
        p1, p2 = tmp[0], tmp[1]
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def rps(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1 ,p2 = geatpy.rps(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def rws(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        # print(tmp_scores)
        p1 ,p2 = geatpy.rws(tmp_scores, num)
        # print(p1, p2)
        p1 %= len(genes)
        p2 %= len(genes)
        # print("rws")
        return  p1, p2

    @staticmethod
    def sus(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1 ,p2 = geatpy.sus(tmp_scores, num, None)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def tour(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1 ,p2 = geatpy.tour(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def urs(genes, scores, num=2):
        tmp_scores = np.copy(scores)[:, np.newaxis]
        p1, p2 = geatpy.urs(tmp_scores, num)
        p1 %= len(genes)
        p2 %= len(genes)
        return  p1, p2

    @staticmethod
    def crws(genes, scores, num=2):
        scores = np.array(scores)
        scores += 1e-8
        scores /= np.max(scores)
        probs = scores / np.sum(scores)
        p = np.copy(probs)
        for i in range(1, len(scores)):
            p[i] = p[i-1] + probs[i]
        g = []
        while len(g) < 2:
            for i in range(len(scores)):
                if np.random.random() < p[i]:
                    g.append(i)
                    if len(g) == 2:
                        break

        return  g[0], g[1]


class EnumSelection(Enum):
    dup   = 0
    ecs   = 1
    etour = 2
    otos  = 3
    rcs   = 4
    rps   = 5
    rws   = 6
    sus   = 7
    tour  = 8
    urs   = 9
    crws  = 10
    @staticmethod
    def randomSelection():
        l = len(EnumSelection)
        idx = np.random.randint(l)
        selection = EnumSelection(idx)
        return selection



