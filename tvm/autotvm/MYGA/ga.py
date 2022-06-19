import numpy as np
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from tvm.autotvm.MYGA.selection import EnumSelection,GASelection
from tvm.autotvm.MYGA.config import SelectionConfig, CrossoverConfig, MutationConfig, PopulationConfig, MultipopulationConfig
from tvm.autotvm.MYGA.mutation import EnumMutation, GAMutation
from tvm.autotvm.MYGA.crossover import EnumCrossover, GACrossover
from tvm.autotvm.MYGA.aga import AGA,EnumAGA
from tvm.autotvm.MYGA.sa import EnumSA
from tvm.autotvm.measure import MeasureInput, create_measure_batch
import math

class Solution:
    def __init__(self, gene, score=0.0):
        self.gene = gene
        self.score = score

class Population:
    def __init__(self, tunerConfig, popConfig : PopulationConfig):
        self.tunerConfig = tunerConfig # tuner config,don't need to care
        self.popConfig = popConfig # pop config,set selection op and other op
        self.space = len(self.tunerConfig.space) # get search space size
        self.popConfig.setSize(min(self.popConfig.getSize(), self.space)) # pop size can't exceed the search space size

        self.solutions = [] # pop individuals
        self.elites = [] # pop elites

        self.genes = [] # individual's gene
        self.scores = [] # individual's score

        self.tmp_solution = []

        self.next_solution = []
        self.next_batch = []
        self.next_id = -1 # index of next individual

    def init(self):
        # random initialization
        for _ in range(self.popConfig.getSize()):
            tmp_gene = point2knob(np.random.randint(self.space), self.tunerConfig.dims)
            tmp_solution = Solution(tmp_gene)
            self.solutions.append(tmp_solution)

    def setEvaluate(self, evaluate):
        self.evaluate = evaluate

    def update(self):
        # check whether to use SA or not
        if self.popConfig.getExtraConfig().getSA() == EnumSA.GA: # don't use SA
            self.genes = []
            self.scores = []
            for solution in self.solutions:
                if solution.score > 0:
                    self.genes.append(solution.gene)
                    self.scores.append(solution.score)
            self.crossover()
            self.mutation()
            self.next_id = -1
        else: # use SA
            self.T = 4 # The initial temperature of SA
            self.a = 0.5 # the decrease rate of SA
            self.Tmin = 1 # the lowest temperature of SA
            while self.T > self.Tmin: # SA begin
                self.genes = []
                self.scores = []
                for solution in self.solutions:
                    if solution.score > 0:
                        self.genes.append(solution.gene)
                        self.scores.append(solution.score)
                self.crossover()
                self.mutation()
                self.T *= self.a
                self.next_id = -1

    # 返回种群适应度的最大值，平均值和最小值
    def get_max_and_avg_and_min(self, solutions):
        f_max = 0
        f_min = solutions[0].score
        f_avg = 0
        sum = 0
        num = 0
        for solution in solutions:
            if solution.score > f_max:
                f_max = solution.score
            if solution.score < f_min:
                f_min = solution.score
            if solution.score != 0.0:
                sum += solution.score
                num += 1

        if num != 0:
            f_avg = sum / num

        return f_max , f_avg, f_min

    # 交叉操作
    def crossover(self):
        self.tmp_solution = []

        dims = self.tunerConfig.dims
        size = self.popConfig.getSize()

        # get selection and crossover op
        selectionOP = self.popConfig.getSelectionConfig().getOP()
        crossoverOP = self.popConfig.getCrossoverConfig().getOP()

        # get the function corresponding to the selection and crossover op
        selectMethod = getattr(GASelection, str(selectionOP.name))
        crossoverMethod = getattr(GACrossover, str(crossoverOP.name))

        # get AGA
        aga = self.popConfig.getExtraConfig().getAGA()
        AGAMode = False
        toEvaluate = []
        if aga != EnumAGA.GA:
            f_max , f_avg, f_min= self.get_max_and_avg_and_min(self.solutions)
            if f_max != 0 :
                AGAMode = True

            AGAMethod = getattr(AGA, str(aga.name))

        # begin to crossover
        while len(self.tmp_solution) < size:
            if len(self.genes) >= 2:
                p1, p2 = selectMethod(self.genes, self.scores)
                s1, s2 = self.solutions[p1], self.solutions[p2]
                if s1.score < s1.score:
                    s1, s2 = s2, s1
            else:
                s1 = s2 = self.solutions[0]
            if AGAMode:
                rate = AGAMethod(self.popConfig.getCrossoverConfig().min_rate,
                                 self.popConfig.getCrossoverConfig().max_rate,
                                 f_max, f_avg, s1.score, target = 0)
            else:
                rate = self.popConfig.getCrossoverConfig().getRate()
            if np.random.random() < rate:
                if crossoverOP == EnumCrossover.dc:
                    if len(self.genes) >= 2:
                        tmp_gene = GACrossover.dc(self.genes, len(self.tunerConfig.dims))
                    else:
                        tmp_gene = GACrossover.dc([s1.gene, s2.gene], len(self.tunerConfig.dims))
                elif crossoverOP == EnumCrossover.hc:
                    tmp_gene = GACrossover.hc(s1.gene, s2.gene, len(self.tunerConfig.dims), self.tunerConfig.dims, self.space)
                else:
                    tmp_gene = crossoverMethod(s1.gene, s2.gene, len(self.tunerConfig.dims))
                if self.popConfig.getExtraConfig().getAGA() != EnumAGA.GA or self.popConfig.getExtraConfig().getSA() != EnumSA.GA:
                    score = (s1.score + s2.score) / 2
                else:
                    score = 0.0
                self.tmp_solution.append(Solution(tmp_gene, score))

    # def crossover_single(self, selection, crossover):
    #     if len(self.scores) >= 2:
    #         g1, g2 = selectMethod(self.genes, self.scores)
    #     else:
    #         g1 = g2 = self.genes[0]
    #
    #     if hasattr(GASelection, str(crossoverOP.name)):
    #         crossoverMethod = getattr(GACrossover, str(crossoverOP.name))
    #         tmp_gene = crossoverMethod(g1, g2, len(self.tunerConfig.dims))
    #         return tmp_gene
    #     else:
    #         return GACrossover.spc(g1, g2, len(self.tunerConfig.dims))

    def mutation(self):
        next_solution = []

        aga = self.popConfig.getExtraConfig().getAGA()
        f_max, f_avg, f_min = self.get_max_and_avg_and_min(self.tmp_solution)
        AGAMode = False
        if aga != EnumAGA.GA:
            if f_max != 0:
                AGAMode = True

            aga = self.popConfig.getExtraConfig().getAGA()
            AGAMethod = getattr(AGA, str(aga.name))

        dims = self.tunerConfig.dims
        size = self.popConfig.getSize()

        mutationOP = self.popConfig.getMutationConfig().getOP()

        # begin to matute
        for solution in self.tmp_solution:
            if AGAMode:
                rate = AGAMethod(self.popConfig.getMutationConfig().min_rate,
                                 self.popConfig.getMutationConfig().max_rate,
                                 f_max, f_avg, solution.score, target = 1)
            else:
                rate = self.popConfig.getMutationConfig().getRate()
            if np.random.random() < rate:
                if hasattr(GAMutation, str(mutationOP.name)):
                    mutationMethod = getattr(GAMutation, str(mutationOP.name))
                    gene = mutationMethod(solution.gene, self.tunerConfig.dims, rate)
                else:
                    gene = GAMutation.rm(solution.gene, self.tunerConfig.dims, rate)
                if self.popConfig.getExtraConfig().getSA() == EnumSA.GA:
                    solution.gene = gene
                    solution.score = 0.0
                elif self.popConfig.getExtraConfig().getSA() == EnumSA.GA_SA1:
                    value = solution.score
                    p = self.checkGene(self.T, value, solution.score)
                    if np.random.random() < p :
                        solution.gene = gene
                        solution.score = 0.0
                elif self.popConfig.getExtraConfig().getSA() == EnumSA.GA_SA2:
                    value = solution.score
                    p = self.checkGene(self.T, value, (f_avg - f_min)/2)
                    if np.random.random() < p :
                        solution.gene = gene
                        solution.score = 0.0
            next_solution.append(solution)

        self.solutions = next_solution

    # def mutation_single(self, gene):
    #     rate = self.popConfig.getMutationConfig().getRate()
    #     mutationOP = self.popConfig.getMutationConfig().getOP()
    #
    #     if hasattr(GAMutation, str(mutationOP.name)):
    #         mutationMethod = getattr(GAMutation, str(mutationOP.name))
    #         tmp_gene = mutationMethod(gene, self.tunerConfig.dims, rate)
    #         return tmp_gene
    #     else:
    #         return GAMutation.rm(gene, self.tunerConfig.dims, rate)

    def checkGene(self, T, f1, f2):
        f = (f1 - f2) / f2
        if f > 0:
            return math.exp(- f / T)
        else:
            return 1

    def has_next(self):
        return self.next_id < (self.popConfig.getSize() - 1)

    def get_next(self):
        self.next_id += 1
        return self.solutions[self.next_id % len(self.solutions)]

class Evaluate:
    def __init__(self, task, measure_option, dims, space):
        self.task = task
        self.measure_option = measure_option
        self.dims = dims
        self.space = space

    def evaluate(self, solutions):
        measure_batch = create_measure_batch(self.task, self.measure_option)
        values = [self.space.get(knob2point(solution.gene, self.dims)) for solution in solutions]
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        for i in range(1, max((len(values)+1)//n_parallel, 2)):
            inputs = [MeasureInput(self.task.target, self.task, value)
                      for value in
                      values[ (i-1)*n_parallel : min(i*n_parallel,len(values)) ]
                      ]
            results = measure_batch(inputs)
            i = 0
            for inp, res in zip(inputs, results):
                if i < len(solutions):
                    solution = solutions[i]
                    if res.error_no == 0:
                        y = inp.task.flop / np.mean(res.costs)
                        solution.score = y
                    else:
                        solution.score = 0.0
                    i += 1
                else:
                    break

class Multipopulation:
    def __init__(self, tunerConfig, multipopConfig : MultipopulationConfig = MultipopulationConfig()):
        self.pops = []
        self.multipopConfig = multipopConfig
        for popConfig in self.multipopConfig.pop_configs:
            pop = Population(tunerConfig, popConfig)
            pop.init()
            self.pops.append(pop)
        self.pop_num = len(self.pops)
        self.maxGenes = []
        self.maxScores = []
        self.maxI = []

        self.next = []

    def setEvaluate(self, evaluate):
        for pop in self.pops:
            pop.setEvaluate(evaluate)

    def eliteInduvidual(self):
        MP = len(self.pops)
        maxGenes = []
        maxScores = []
        maxI = []
        if MP > 1:
            for i in range(MP):
                if len(self.pops[i].solutions) > 0:
                    solutions = np.array(self.pops[i].solutions)
                    index = 0
                    for j in range(len(solutions)):
                        if solutions[j].score > solutions[index].score:
                            index = j
                    maxI.append(index)
                    maxScores.append(self.pops[i].solutions[index].score)
                    maxGenes.append(self.pops[i].solutions[index].gene)
                else:
                    maxI.append(-1)
                    maxScores.append(-1)
                    maxGenes.append(None)
        self.maxGenes = maxGenes
        self.maxScores = maxScores
        self.maxI = maxI

    def immigrant(self):
        MP = len(self.pops)
        if MP > 1:
            for i in range(MP):
                maxGene = self.maxGenes[i]
                maxScore = self.maxScores[i]
                if maxScore != -1:
                    next_i = (i + 1) % MP
                    solutions = np.array(self.pops[next_i].solutions)
                    if len(solutions) > 0:
                        index = 0
                        for j in range(len(solutions)):
                            if solutions[j].score < solutions[index].score:
                                index = j
                        self.pops[next_i].solutions[index].gene = maxGene
                        self.pops[next_i].solutions[index].score = maxScore

    def update(self,inputs, results):
        i = 0
        for inp, res in zip(inputs, results):
            if i < len(self.next):
                solution = self.next[i]
                if res.error_no == 0:
                    y = inp.task.flop / np.mean(res.costs)
                    solution.score = y
                else:
                    solution.score = 0.0
                i += 1

    def has_next(self):
        for pop in self.pops:
            if pop.has_next():
                return True

        return False

    def next_batch(self,batch_size):
        self.next = []
        next_genes = []
        num = 0
        check = 0
        index = 0
        while num < batch_size and check < self.pop_num:
            pop = self.pops[index % self.pop_num]
            index += 1
            if pop.has_next():
                solution = pop.get_next()
                self.next.append(solution)
                next_genes.append(solution.gene)
                num += 1
                check = 0
            else:
                check += 1

        return next_genes

class UtilsGa:
    @staticmethod
    def init_pop(config):
        genes = []
        visited = set([])

        # random initialization
        config.pop_size = min(config.pop_size, len(config.space))
        config.elite_num = min(config.pop_size, config.elite_num)
        for _ in range(config.pop_size):
            tmp_gene = point2knob(np.random.randint(len(config.space)), config.dims)
            while knob2point(tmp_gene, config.dims) in config.visited:
                tmp_gene = point2knob(np.random.randint(len(config.space)), config.dims)

            genes.append(tmp_gene)
            visited.add(knob2point(tmp_gene, config.dims))

        return genes, visited

    @staticmethod
    def crossover_single(pop : Population):
        if type(pop.selection) is not EnumSelection :
            pop.selection = EnumSelection.rws
        if type(pop.crossover) is not EnumCrossover :
            pop.crossover = EnumCrossover.spc
        if hasattr(GASelection, str(selection.name)):
            selectMethod = getattr(GASelection, str(pop.selection.name))
            g1 , g2 = selectMethod(pop.genes, pop.scores)
            crossoverMethod = getattr(GACrossover, str(pop.crossover.name))
            tmp_gene = crossoverMethod(g1, g2, len(pop.config.dims))
            return tmp_gene
        else:
            return None

    @staticmethod
    def crossover_pop(pop : Population):
        tmp_genes = []
        if setSelection:
            # print("setSelection : " + setSelection.name)
            for _ in range(len(pop.config.space)):
                gene = UtilsGa.crossover_single(pop)
                tmp_genes.append(gene)
        else:
            rates = []
            for value in pop.config.rates.values():
                rates.append(value)
            indices = np.arange(len(pop.rates))
            rates = np.array(pop.rates)
            rates += 1e-8
            rates /= np.max(rates)
            probs = rates / np.sum(rates)
            for _ in range(len(pop.config.space)):
                selection_index = np.random.choice(indices, size=1, replace=False, p=probs)
                selection = EnumSelection(selection_index)
                gene = UtilsGa.crossover_single(pop)
                tmp_genes.append(gene)
        return tmp_genes



