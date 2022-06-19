# from tvm.autotvm.my_utils.utils_config import CrossoverConfig
from tvm.autotvm.MYGA.config import PopulationConfig, SelectionConfig, MultipopulationConfig
import numpy as np
from tvm.autotvm.MYGA import GACrossover
# pop = [[1, 2, 3], [2, 3, 1], [1, 4, 2]]
# scores = [2, 3, 1]
#
# config = CrossoverConfig(5, 3)
#
# next_pop = UtilsGa.crossover_pop(pop, scores, config)
# print(next_pop)
# a = [3, 2, 3, 4, 5]
# dims = [4, 2, 4, 6, 6]
# a = GAMutation.im(a, dims, 0)
# print(a)

def multipoptest():
    a = 1
    config = {
        "config1" : {
        "pop_size": 100,
        "elite_num": 3,
        "selection": {
            "op": "dup"
        },
        "crossover": {
            "op": "tpc",
            "rate": 0.6
        },
        "mutation": {
            "op": "im2",
            "rate": 0.02
        }
    },
        "config2": {
            "pop_size": 200,
            "elite_num": 4,
            "selection": {
                "op": "rws"
            },
            "crossover": {
                "op": "spc",
                "rate": 0.6
            },
            "mutation": {
                "op": "rm",
                "rate": 0.001
            }
        },
        "config3": {
            "pop_size": 300,
            "elite_num": 5,
            "selection": {
                "op": "dup"
            },
            "crossover": {
                "op": "tpc",
                "rate": 0.6
            },
            "mutation": {
                "op": "em",
                "rate": 0.06
            }
        }
    }
    pops = MultipopulationConfig(config)
    b = 1


def popConfigTest():
    config = {
        "pop_size" : 200,
        "elite_num" : 3 ,
        "selection" : {
            "op" : "dup"
        },
        "crossover":{
            "op" : "tpc",
            "rate" : 0.6
        },
        "mutation": {
            "op": "im2",
            "rate": 0.001
        }
    }
    cfg = PopulationConfig(config)
    o = SelectionConfig()
    # print(o.getOP())
    # print(hasattr(o, "_SelectionConfig__op" ))
    print(cfg.getMutationConfig().getRate())

# multipoptest()
# a = [0, 1, 2, 3]
# a[0:2] = a[1::-1]
# print(a)
# print(np.random.randint(2))
# for i in range(1, 3):
#     print(i)
g1 = [1,2,3,4,5]
g2 = [6,7,8,9,10]
tmp_scores = [1,3,4,5,6]
for i in range(1,2):
    print(i)
tmp_scores = tmp_scores[: (len(tmp_scores) // 2) * 2 ]
# print(tmp_scores)
# print(tmp_scores[-1])
# for i in range(100):
#     g = GACrossover.mpc(g1,g2,5)
#     print(g)
