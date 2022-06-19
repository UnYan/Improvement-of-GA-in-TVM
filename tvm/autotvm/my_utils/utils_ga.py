import numpy as np
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from autotvm.MYGA.selection import EnumSelection,GASelection
from .config import SelectionConfig, CrossoverConfig, MutationConfig
from autotvm.MYGA.mutation import EnumMutation, GAMutation
from autotvm.MYGA.crossover import EnumCrossover, GACrossover


class Population:
    def __init__(self, config, pop_size = 100, elite_num=3,
                 selectionConfig : SelectionConfig = SelectionConfig(),
                 crossoverConfig : CrossoverConfig = CrossoverConfig(),
                 mutationConfig  : MutationConfig  = MutationConfig() ):

        self.config = config
        self.pop_size = pop_size
        self.elite_num = elite_num

        self.selectionConfig = selectionConfig
        self.crossoverConfig = crossoverConfig
        self.mutationConfig = mutationConfig

        if type(self.selectionConfig.selectionOP) is not EnumSelection:
            self.selectionConfig.selectionOP = EnumSelection.rws

        if type(self.crossoverConfig.crossOP) is not EnumCrossover:
            self.crossoverConfig.crossOP = EnumCrossover.spc

        if type(self.mutationConfig.mutationOP) is not EnumMutation:
            self.mutationConfig.mutationOP = EnumMutation.rm


        self.genes = []
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.visited = set([])

    def init(self):
        # random initialization
        self.pop_size = min(self.pop_size, len(self.config.space))
        self.elite_num = min(self.pop_size, self.elite_num)
        for _ in range(self.pop_size):
            tmp_gene = point2knob(np.random.randint(len(self.config.space)), self.config.dims)
            while knob2point(tmp_gene, self.config.dims) in self.visited:
                tmp_gene = point2knob(np.random.randint(len(self.config.space)), self.config.dims)

            self.genes.append(tmp_gene)
            self.visited.add(knob2point(tmp_gene, self.config.dims))

    def update(self):
        pass

    def crossover(self):
        tmp_genes = []
        while len(tmp_genes) < self.pop_size:
            if np.random.random() < self.crossoverConfig.rate:
                tmp_gene = self.crossover_single()
                if tmp_gene and tmp_gene not in tmp_genes:
                    tmp_genes.append(tmp_gene)
        self.genes = tmp_genes

    def crossover_single(self):
        selectMethod = getattr(GASelection, str(self.selectionConfig.selectionOP.name))
        g1, g2 = selectMethod(self.genes, self.scores)

        if hasattr(GASelection, str(self.selectionConfig.selectionOP.name)):
            crossoverMethod = getattr(GACrossover, str(self.crossoverConfig.crossOP.name))
            tmp_gene = crossoverMethod(g1, g2, len(self.config.dims))
            return tmp_gene
        else:
            return GACrossover.spc(g1, g2, len(self.config.dims))


    def mutation(self):
        for gene in self.genes:
            if np.random.random() < self.mutationConfig.rate:
                gene = self.mutation_single(gene)

    def mutation_single(self, gene):
        if hasattr(GAMutation, str(self.mutationConfig.mutationOP.name)):
            mutationMethod = getattr(GAMutation, str(self.mutationConfig.mutationOP.name))
            tmp_gene = mutationMethod(gene, self.config.dims, self.mutationConfig.rate)
            return tmp_gene
        else:
            return GAMutation.rm(gene, self.config.dims, self.mutationConfig.rate)

class Multipopulation:
    def __init__(self,pops):
        self.pops = pops
        self.maxGenes = []
        self.maxScores = []
        self.maxI = []

    def eliteInduvidual(self):
        MP = len(self.pops)
        maxGenes = []
        maxScores = []
        maxI = []
        if MP > 1:
            for i in range(MP):
                if len(self.pops[i].scores) > 0:
                    scores = np.array(self.pops[i].scores)
                    index = np.argmax(scores) % len(self.pops[i].genes)
                    maxI.append(index)
                    maxScores.append(self.pops[i].scores[index])
                    maxGenes.append(self.pops[i].genes[index])
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
                    scores = np.array(self.pops[next_i].scores)
                    if len(scores) > 0:
                        minIndex = np.argmin(scores) % len(self.pops[i].genes)
                        self.pops[next_i].genes[minIndex] = maxGene
                        self.pops[next_i].scores[minIndex] = maxScore

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



