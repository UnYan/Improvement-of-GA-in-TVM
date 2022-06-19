# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=consider-using-enumerate,invalid-name,abstract-method

"""Tuner with genetic algorithm"""

import numpy as np
from ..measure import MeasureInput, create_measure_batch
from ..utils import format_si_prefix
from .tuner import Tuner
from .model_based_tuner import knob2point
from tvm.autotvm.MYGA.selection import EnumSelection
from tvm.autotvm.MYGA.crossover import EnumCrossover
from tvm.autotvm.MYGA.mutation import EnumMutation
from tvm.autotvm.MYGA.ga import Multipopulation, Population,AGA,EnumAGA,Evaluate
from tvm.autotvm.MYGA.config import MultipopulationConfig, PopulationConfig
import logging
from ..measure import MeasureInput, create_measure_batch
from ..utils import format_si_prefix

from ..env import GLOBAL_SCOPE
logger = logging.getLogger("autotvm")

class MyGATuner(Tuner):
    """Tuner with my genetic algorithm.
    This tuner does not have a cost model so it always run measurement on real machines.
    This tuner expands the :code:`ConfigEntity` as gene.

    Parameters
    ----------
    pop_size: int
        number of genes in one generation
    elite_num: int
        number of elite to keep
    mutation_prob: float
        probability of mutation of a knob in a gene
    """

    def __init__(self, task, pop_size=100, pop_num = 1, elite_num=3, iterate_num = 3,
                 selection = EnumSelection.rws, crossover = EnumCrossover.spc, mutation = EnumMutation.rm,
                 crossover_prob = 0.7, mutation_prob=0.1, update_rate = 0.3,
                 aga = EnumAGA.GA,
                 popsConfig = None):
        super(MyGATuner, self).__init__(task)

        # print("selection: " + self.selection.name)
        assert elite_num <= pop_size, "The number of elites must be less than population size"
        self.update_rate = update_rate

        # space info
        self.space = task.config_space
        self.dim_keys = []
        self.dims = []
        for k, v in self.space.space_map.items():
            self.dim_keys.append(k)
            self.dims.append(len(v))

        class Config:
            def __init__(self, space, dims):
                self.space = space
                self.dims = dims

        # current generation
        self.orgin_trial_pt = 0
        self.trial_pt = 0
        self.config = Config(self.space, self.dims)
        # random initialization
        if popsConfig is None:
            if pop_num <= 1:
                popsConfig = {
                    "pop1" : {
                        "pop_size": pop_size,
                        "elite_num": elite_num,
                        "selection": {
                            "op": selection.name
                        },
                        "crossover": {
                            "op":  crossover.name ,
                            "rate": crossover_prob
                        },
                        "mutation": {
                            "op": mutation.name,
                            "rate": mutation_prob
                        },
                        "extra":{
                            "AGA": aga.name
                        }
                    }
                }
            else:
                popsConfig = {}
                for i in range(pop_num):
                    dic = {
                        "pop" + str(i) : {
                            "pop_size": pop_size,
                            "elite_num": elite_num,
                            "selection": {
                                "op": EnumSelection.randomSelection().name
                            },
                            "crossover": {
                                "op": EnumCrossover.randomCrossover().name,
                                "rate": crossover_prob
                            },
                            "mutation": {
                                "op": EnumMutation.randomMutation().name,
                                "rate": mutation_prob
                            },
                            "extra": {
                                "AGA": EnumAGA.GA
                            }
                        }
                    }
                    popsConfig.update(dic)
            multiPopConfig = MultipopulationConfig(popsConfig)
        elif type(popsConfig) is dict:
            multiPopConfig = MultipopulationConfig(popsConfig)
        elif type(popsConfig) is MultipopulationConfig:
            multiPopConfig = MultipopulationConfig(popsConfig)
        elif type(popsConfig) is PopulationConfig:
            multiPopConfig = MultipopulationConfig({"pop":popsConfig})
        else:
            multiPopConfig = MultipopulationConfig()
        self.multipop = Multipopulation(self.config, multiPopConfig)
        print(self.multipop.multipopConfig.toString())
        self.pop_num = len(self.multipop.pops)
        self.setSA = False

    def next_batch(self, batch_size):
        # ret = []
        # size = 0
        # for pop in self.multipop.pops:
        #     size += pop.popConfig.getSize()
        # self.orgin_trial_pt = self.trial_pt
        # for _ in range(batch_size):
        #     idx_pop = self.trial_pt % self.pop_num
        #     idx = (self.trial_pt // self.pop_num) % self.multipop.pops[idx_pop].popConfig.getSize()
        #     gene = self.multipop.pops[idx_pop].genes[idx]
        #     self.trial_pt += 1
        #     ret.append(self.space.get(knob2point(gene, self.dims)))
        #
        # self.trial_pt = self.trial_pt % size
        ret = []
        genes = self.multipop.next_batch(batch_size)
        for gene in genes:
            value = knob2point(gene, self.dims)
            if value > len(self.space):
                # print(self.ims)
                # print(gene)
                value %= len(self.space)
            ret.append(self.space.get(value))
        return ret

    def evaluate(self, gene):
        measure_batch = create_measure_batch(self.task, self.measure_option)
        value = knob2point(gene, self.dims)
        value = self.space.get(value)
        inputs = [MeasureInput(self.task.target, self.task, value)]
        results = measure_batch(inputs)
        if results[0].error_no == 0:
            y = inputs[0].task.flop / np.mean(results[0].costs)
            return y
        else:
            return 0.0

    def evaluate(self, pop):
        # measure_batch = create_measure_batch(self.task, self.measure_option)
        # inputs = [MeasureInput(self.task.target, self.task, solution.gene) for solution in pop.solutions]
        # results = measure_batch(inputs)
        # i = 0
        # for inp, res in zip(inputs, results):
        #     if i < len(self.next):
        #         solution = self.next[i]
        #         if res.error_no == 0:
        #             y = inp.task.flop / np.mean(res.costs)
        #             solution.score = y
        #         else:
        #             solution.score = 0.0
        #         i += 1
        pass


    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        si_prefix: str
            One of tvm.autotvm.utils.SI_PREFIXES. The SI prefix to use when reporting FLOPS.
        """
        self.measure_option = measure_option
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        if not self.setSA:
            evaluate = Evaluate(self.task, self.measure_option, self.dims, self.space)
            self.multipop.setEvaluate(evaluate)
            self.setSA =True
        
        # Validate si_prefix arg
        format_si_prefix(0, si_prefix)

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0
        errors = []
        while i < n_trial:
            if not self.has_next():
                break

            configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                    result_msg = res
                else:
                    flops = 0
                    error_ct += 1
                    tb, error = res.costs
                    if isinstance(error, str):
                        errors.append(tb + "\n" + error)
                    else:
                        errors.append(tb + "\n" + str(error))
                    result_msg = errors[-1]

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug(
                    "No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
                    i + k + 1,
                    si_prefix,
                    format_si_prefix(flops, si_prefix),
                    format_si_prefix(self.best_flops, si_prefix),
                    result_msg,
                    config,
                )

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > self.error_ct_threshold:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Switching to debug mode.")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        if error_ct == i:
            _, f = tempfile.mkstemp(prefix="tvm_tuning_errors_", suffix=".log", text=True)
            with open(f, "w") as file:
                file.write("\n".join(errors))
            logging.warning(
                "Could not find any valid schedule for task %s. "
                "A file containing the errors has been written to %s.",
                self.task,
                f,
            )
        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def update(self, inputs, results):
        # 多种群移民
        self.multipop.update(inputs, results)
        self.multipop.eliteInduvidual()
        self.multipop.immigrant()

        judge = False
        # 根据迭代概率及所有种群是否已遍历完毕判断是否进行种群迭代
        for pop in self.multipop.pops:
            if np.random.random() < self.update_rate or not pop.has_next():
            # if not pop.has_next():
                judge = True
                break

        # 进行种群迭代
        if judge:
            for pop in self.multipop.pops:
                pop.update()

            self.trial_pt = 0


    def has_next(self):
        for pop in self.multipop.pops:
            if not pop.has_next():
                return False
        return True
        # visited_size = 0
        # genes_size = 0
        # for pop in self.multipop.pops:
        #     visited_size += len(pop.visited)
        #     genes_size += len(pop.genes)
        # return visited_size - (genes_size - self.trial_pt) < (len(self.space) * self.pop_num)

    def load_history(self, data_set, min_seed_records=500):
        pass
