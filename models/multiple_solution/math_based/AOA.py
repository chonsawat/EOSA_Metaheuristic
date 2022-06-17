#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 09:56, 07/07/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

import numpy as np
from copy import deepcopy
from models.multiple_solution.optimizer import Optimizer


class OriginalAOA(Optimizer):
    """
    The original version of: Arithmetic Optimization Algorithm (AOA)
    Link:
        https://doi.org/10.1016/j.cma.2020.113609
    """

    def __init__(self, problem, root_paras, epoch=10000, pop_size=100, alpha=5, miu=0.5, moa_min=0.2, moa_max=0.9, **kwargs):
        """
        Args:
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            alpha (int): fixed parameter, sensitive exploitation parameter, Default: 5,
            miu (float): fixed parameter , control parameter to adjust the search process, Default: 0.5,
            moa_min (float): range min of Math Optimizer Accelerated, Default: 0.2,
            moa_max (float): range max of Math Optimizer Accelerated, Default: 0.9,
        """
        super().__init__(problem, root_paras, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = False
        self.verbose=True
        self.epoch = epoch
        self.pop_size = pop_size
        self.alpha = alpha
        self.miu = miu
        self.moa_min = moa_min
        self.moa_max = moa_max
        self.objective_func=problem
        self.a_scatter=None
        self.g_best=None
        self.pop= [self._create_solution__(minmax=1) for _ in range(self.pop_size)]
        
    def _set_scatter__(self, a_scatter):
        self.a_scatter=a_scatter

    def _get_initial_solutions__(self):
        return self.pop

    def _create_solution__(self, minmax=0):
        x = np.random.uniform(self.domain_range[0], self.domain_range[1], (self.problem_size, 1))
        fit = self._fitness_model__(solution=x, minmax=minmax)
        fragrance = 0
        return [x, fit, fragrance]

    def _fitness_model__(self, solution=None, minmax=0):
        """ Assumption that objective function always return the original value """
        return 0#self.objective_func(solution, self.pop_size) if minmax == 0 \
            #else 1.0 / self.objective_func(solution, self.pop_size)
    def train(self):
         sorted_pop = sorted(self.pop, key=lambda temp: temp[self.ID_FIT])
         self.g_best = deepcopy(sorted_pop[0])

         for epoch in range(self.epoch):
             self.evolve(epoch)
             self.loss_train.append(self.g_best[self.ID_FIT])
             if self.verbose:
                 print("> Epoch: {}, Best fit: {}".format(epoch + 1, self.g_best[self.ID_FIT]))
         self.solution = self.g_best
         return self.g_best[self.ID_POS], self.g_best[self.ID_FIT], self.loss_train
     
    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        moa = self.moa_min + epoch * ((self.moa_max - self.moa_min) / self.epoch)  # Eq. 2
        mop = 1 - (epoch ** (1.0 / self.alpha)) / (self.epoch ** (1.0 / self.alpha))  # Eq. 4
        
        ub=self.domain_range[1]
        lb=self.domain_range[0]
        
        pop_new = []
        for idx in range(0, self.pop_size):
            pos_new = deepcopy(self.pop[idx][self.ID_POS])
            for j in range(0, self.problem.n_dims):
                r1, r2, r3 = np.random.rand(3)
                if r1 > moa:  # Exploration phase
                    if r2 < 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] / (mop + self.EPSILON) * \
                                     ((ub - lb) * self.miu + lb)
                    else:
                        pos_new[j] = self.g_best[self.ID_POS][j] * mop * ((ub - lb) * self.miu + lb)
                else:  # Exploitation phase
                    if r3 < 0.5:
                        pos_new[j] = self.g_best[self.ID_POS][j] - mop * ((ub - lb) * self.miu + lb)
                    else:
                        pos_new[j] = self.g_best[self.ID_POS][j] + mop * ((ub - lb) * self.miu + lb)
            pos_new = self.amend_position_faster(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


