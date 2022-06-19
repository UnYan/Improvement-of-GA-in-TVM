from autotvm.MYGA.selection import EnumSelection
from autotvm.MYGA.mutation import EnumMutation
from autotvm.MYGA.crossover import EnumCrossover
from .utils import Utils


# 配置类，用于设置选择策略，交叉算子等参数
class SelectionConfig:
    def __init__(self, selectionOP = EnumSelection.rws):
        self.__op = selectionOP

    def setOP(self, op):
        if type(op) is EnumSelection:
            self.__op = op
        elif type(op) is str:
            try:
                self.__op = EnumSelection.__getitem__(op)
            except:
                pass

    def getOP(self):
        return self.__op


class CrossoverConfig:
    def __init__(self, rate = 0.5, crossOP = EnumCrossover.spc):
        self.__max_rate = 0.9
        self.__min_rate = 0.4
        self.__rate = rate
        self.__op = crossOP

    def setOP(self, op):
        if type(op) is EnumCrossover:
            self.__op = op
        elif type(op) is str:
            try:
                self.__op = EnumCrossover.__getitem__(op)
            except:
                pass

    def setRate(self, rate):
        if rate > self.__min_rate and rate < self.__max_rate:
            self.__rate = rate

    def getOP(self):
        return self.__op

    def getRate(self):
        return self.__rate

class MutationConfig:
    def __init__(self, rate = 0.05, mutationOP = EnumMutation.rm):
        self.__max_rate = 0.1
        self.__min_rate = 0.01
        self.__rate = rate
        self.__op = mutationOP

    def setOP(self, op):
        if type(op) is EnumMutation:
            self.__op = op
        elif type(op) is str:
            try:
                self.__op = EnumMutation.__getitem__(op)
            except:
                pass


    def setRate(self, rate):
        if rate > self.__min_rate and rate < self.__max_rate:
            self.__rate = rate

    def getOP(self):
        return self.__op

    def getRate(self):
        return self.__rate

class PopulationConfig:
    # config example
    # config = {
    #     "pop_size" : 200,
    #     "elite_num" : 3
    # }

    def __init__(self, config:dict):
        self.__pop_size = 100
        self.__elite_num = 3

        self.__selectionConfig = SelectionConfig()
        self.__crossoverConfig = CrossoverConfig()
        self.__mutationConfig = MutationConfig()


        self.numberParams = ["pop_size", "elite_num"]
        self.configParams = ["selection", "crossover", "mutation"]
        self.configMap = {
            "selection" : "setSelectionConfig",
            "crossover" : "setCrossoverConfig",
            "mutation" : "setMutationConfig"
        }
        # self.prams = ["pop_size", "elite_num", "selection", "crossover", "mutation"]

        if config:
            for key, value in config.items():
                if key in self.numberParams:
                    self.__setattr__("__" + key, value)
                elif key in self.configMap.keys() and type(value) is dict:
                    method = getattr(self, self.configMap[key])
                    method(value)

    def getSize(self):
        return self.__pop_size

    def setSize(self, size):
        if type(size) is int and size > 10:
            self.__pop_size = size
            self.__elite_num = math.min(self.__pop_size, self.__elite_num)

    def setEliteNum(self, num):
        if type(num) is int and num > 1:
            self.__elite_num = math.min(num, self.__pop_size)

    def getEliteNum(self):
        return self.__elite_num

    def getSelectionConfig(self):
        return self.__selectionConfig

    def getCrossoverConfig(self):
        return self.__crossoverConfig

    def getMutationConfig(self):
        return self.__mutationConfig

    def setSelectionConfig(self, selectionConfig):
        if type(selectionConfig) is SelectionConfig:
            self.__selectionConfig = selectionConfig
        elif type(selectionConfig) is dict:
            Utils.parseDict(self.__selectionConfig, selectionConfig,{"op" : "setOP"})

    def setCrossoverConfig(self, crossoverConfig):
        if type(crossoverConfig) is CrossoverConfig:
            self.__crossoverConfig = crossoverConfig
        elif type(crossoverConfig) is dict:
            Utils.parseDict(self.__crossoverConfig, crossoverConfig,{"op" : "setOP", "rate" : "setRate"})

    def setMutationConfig(self, mutationConfig):
        if type(mutationConfig) is MutationConfig:
            self.__mutationConfig = mutationConfig
        elif type(mutationConfig) is dict:
            Utils.parseDict(self.__mutationConfig, mutationConfig,{"op" : "setOP", "rate" : "setRate"})













class MultipopulationConfig:
    def __init__(self):
        pass