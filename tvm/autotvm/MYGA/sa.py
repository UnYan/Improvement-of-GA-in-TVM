# #模拟退火
import numpy as np
from enum import Enum
import math

class EnumSA(Enum):
    GA     = 0
    GA_SA1 = 1
    GA_SA2 = 2

    @staticmethod
    def randomSA():
        l = len(EnumSA)
        idx = np.random.randint(l)
        sa = EnumSA(idx)
        return sa