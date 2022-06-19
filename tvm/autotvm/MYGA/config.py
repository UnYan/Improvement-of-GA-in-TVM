from tvm.autotvm.MYGA.selection import EnumSelection, GASelection
from tvm.autotvm.MYGA.mutation import EnumMutation,GAMutation
from tvm.autotvm.MYGA.crossover import EnumCrossover, GACrossover
from tvm.autotvm.my_utils.utils import Utils
from tvm.autotvm.MYGA.aga import AGA, EnumAGA
from tvm.autotvm.MYGA.sa import EnumSA

class SelectionConfig:
    def __init__(self, selectionOP = EnumSelection.rws):
        self.__op = EnumSelection.rws
        self.setOP(selectionOP)

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

    def toString(self):
        return self.getOP().name

class CrossoverConfig:
    def __init__(self, rate = 0.7, crossOP = EnumCrossover.spc):
        self.max_rate = 0.9
        self.min_rate = 0.6

        self.__rate = 0.7
        self.setRate(rate)

        self.__op = EnumCrossover.spc
        self.setOP(crossOP)

    def toString(self):
        return "{}_{}".format(self.__op.name, self.__rate)

    def setOP(self, op):
        if type(op) is EnumCrossover:
            self.__op = op
        elif type(op) is str:
            try:
                self.__op = EnumCrossover.__getitem__(op)
            except:
                pass

    def setRate(self, rate):
        if rate > self.min_rate and rate < self.max_rate:
            self.__rate = rate

    def getOP(self):
        return self.__op

    def getRate(self):
        return self.__rate

class MutationConfig:
    def __init__(self, rate = 0.1, mutationOP = EnumMutation.rm):
        self.max_rate = 0.2
        self.min_rate = 0.01

        self.__rate = 0.1
        self.setRate(rate)

        self.__op = EnumMutation.rm
        self.setOP(mutationOP)

    def toString(self):
        return "{}_{}".format(self.__op.name, self.__rate)

    def setOP(self, op):
        if type(op) is EnumMutation:
            self.__op = op
        elif type(op) is str:
            try:
                self.__op = EnumMutation.__getitem__(op)
            except:
                pass


    def setRate(self, rate):
        if rate > self.min_rate and rate < self.max_rate:
            self.__rate = rate

    def getOP(self):
        return self.__op

    def getRate(self):
        return self.__rate

class ExtraConfig:
    def __init__(self, aga = EnumAGA.GA, sa = EnumSA.GA):
        self.__AGA = EnumAGA.GA
        self.setAGA(aga)
        self.__SA = EnumSA.GA
        self.setSA(sa)
    def toString(self):
        return "{}_{}".format(self.__AGA.name, self.__SA.name)

    def setAGA(self, aga):
        if type(aga) is EnumAGA:
            self.__AGA = aga
        elif type(aga) is str:
            try:
                self.__AGA = EnumAGA.__getitem__(aga)
            except:
                pass

    def getAGA(self):
        return self.__AGA

    def getSA(self):
        return self.__SA

    def setSA(self, sa):
        if type(sa) is EnumSA:
            self.__SA = sa
        elif type(sa) is str:
            try:
                self.__SA = EnumSA.__getitem__(sa)
            except:
                pass

class PopulationConfig:
    # config example
    # config = {
    #     "pop_size": 200,
    #     "elite_num": 3,
    #     "selection": {
    #         "op": "dup"
    #     },
    #     "crossover": {
    #         "op": "tpc",
    #         "rate": 0.6
    #     },
    #     "mutation": {
    #         "op": "im2",
    #         "rate": 0.001
    #     }
    # }
    def __init__(self, config:dict = {}):
        self.__pop_size = 100
        self.__elite_num = 3

        self.__selectionConfig = SelectionConfig()
        self.__crossoverConfig = CrossoverConfig()
        self.__mutationConfig = MutationConfig()
        self.__extraConfig = ExtraConfig()


        self.normalParams = ["pop_size", "elite_num"]
        self.configMap = {
            "selection" : "setSelectionConfig",
            "crossover" : "setCrossoverConfig",
            "mutation" : "setMutationConfig",
            "extra"    : "setExtraConfig"
        }
        # self.prams = ["pop_size", "elite_num", "selection", "crossover", "mutation"]

        if config:
            for key, value in config.items():
                if key in self.normalParams:
                    self.__setattr__("_" + self.__class__.__name__ + "__" + key, value)
                elif key in self.configMap.keys() and type(value) is dict:
                    method = getattr(self, self.configMap[key])
                    method(value)

    def toString(self):
        s = "{}_{}_{}_{}_{}".format(self.getSelectionConfig().toString(),
                                 self.getCrossoverConfig().toString(),
                                 self.getMutationConfig().toString(),
                                 self.getExtraConfig().toString(),
                                 self.__pop_size)
        return s

    def getSize(self):
        return self.__pop_size

    def setSize(self, size):
        if type(size) is int and size > 10:
            self.__pop_size = size
            self.__elite_num = min(self.__pop_size, self.__elite_num)

    def setEliteNum(self, num):
        if type(num) is int and num > 1 and num < self.__pop_size:
            self.__elite_num = min(num, self.__elite_num)

    def getEliteNum(self):
        return self.__elite_num

    def setExtraConfig(self, extraConfig):
        if type(extraConfig) is ExtraConfig:
            self.__extraConfig = extraConfig
        elif type(extraConfig) is dict:
            Utils.parseDict(self.__extraConfig, extraConfig,{"AGA" : "setAGA", "SA":"setSA"})

    def getExtraConfig(self):
        return self.__extraConfig

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
    # config = {
    #     "config1" : {
    #     "pop_size": 100,
    #     "elite_num": 3,
    #     "selection": {
    #         "op": "dup"
    #     },
    #     "crossover": {
    #         "op": "tpc",
    #         "rate": 0.6
    #     },
    #     "mutation": {
    #         "op": "im2",
    #         "rate": 0.02
    #     }
    # },
    #     "config2": {
    #         "pop_size": 200,
    #         "elite_num": 4,
    #         "selection": {
    #             "op": "rws"
    #         },
    #         "crossover": {
    #             "op": "spc",
    #             "rate": 0.6
    #         },
    #         "mutation": {
    #             "op": "rm",
    #             "rate": 0.001
    #         }
    #     },
    #     "config3": {
    #         "pop_size": 300,
    #         "elite_num": 5,
    #         "selection": {
    #             "op": "dup"
    #         },
    #         "crossover": {
    #             "op": "tpc",
    #             "rate": 0.6
    #         },
    #         "mutation": {
    #             "op": "em",
    #             "rate": 0.06
    #         }
    #     }
    # }
    def __init__(self, config:dict = {}):
        self.pop_configs = []
        for key,value in config.items():
            if type(value) is dict:
                self.pop_configs.append(PopulationConfig(value))
            elif type(value) is PopulationConfig:
                self.pop_configs.append(value)
        if len(self.pop_configs) == 0 :
            self.pop_configs.append(PopulationConfig())

    def toString(self):
        s = ""
        for i in range(len(self.pop_configs)):
            config = self.pop_configs[i]
            s += "pop_{}_".format(i) + config.toString() + ","
        return s