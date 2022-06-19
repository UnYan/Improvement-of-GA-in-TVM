## 自适应遗传算法
import numpy as np
from enum import Enum
import math

# 自适应遗传算法类，包括AGA，LAGA，CAGA，IAGA四类
class AGA:
    # target: 0 -> Pc , 1 -> Pm
    # @staticmethod
    # def GA(k1, k2, f_max, f_avg, f, target):
    #     if f < f_avg:
    #         return k2
    #     else:
    #         return k1 * (f_max - f) / (f_max - f_avg)

    @staticmethod
    def AGA(k1, k2, f_max, f_avg, f, target):
        if f < f_avg:
            return k2
        else:
            return k1 * (f_max - f) / (f_max - f_avg)

    @staticmethod
    def LAGA(k1, k2, f_max, f_avg, f, target):
        if target == 0:
            if f < f_avg:
                return k2
            else:
                return k1 - (k2 - k1)  * (f - f_avg) / (f_max - f_avg)
        else:
            if f < f_avg:
                return k2
            else:
                return k1 - (k2 - k1)  * (f_max - f) / (f_max - f_avg)

    @staticmethod
    def CAGA(k1, k2, f_max, f_avg, f, target):
        if f < f_avg:
            return k2
        else:
            return (k1 + k2) / 2 + (k2 - k1) * math.cos( (f - f_avg) / (f_max - f_avg) - math.pi) / 2

    @staticmethod
    def IAGA(k1, k2, f_max, f_avg, f, target):
        A = 9.903438
        if f < f_avg:
            return k2
        else:
            return (k2 - k1) / (1 + math.exp( A * 2 * (f - f_avg) / (f_max - f_avg))) + k1


class EnumAGA(Enum):
    GA   = 0 # normal Genetic Algorithm
    AGA  = 1 # Adaptive Genetic Algorithm
    LAGA = 2 # Linear Adaptive Genetic Algorithm
    CAGA = 3 # Cosine Adaptive Genetic Algorithm
    IAGA = 4 # Improved Adaptive Genetic Algorithm

    @staticmethod
    def randomAGA():
        l = len(EnumAGA)
        idx = np.random.randint(l)
        aga = EnumAGA(idx)
        return aga
