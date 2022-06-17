import numpy as np
from copy import deepcopy
from models.multiple_solution.root_multiple import RootAlgo
import random
from utils.differential_equations import DiffEquation
from utils.paper_equations import *
import math
from scipy.stats import expon


class BaseEBOA(RootAlgo):
    """
    Standard version of Ebola Virus Optimization Algorithm (belongs to Biology-based Algorithms)
    - In this algorithms: 
    """
    ID_POS = 0
    ID_FIT = 1
    ID_INDIVIDUAL=1
    ID_INDIVIDUAL_INDEX=0
    NEIGHBOURHOOD_THRESHHOLD=0.5
    MIN_MAX_INFECTED_SOL=1
    

    def __init__(self, root_algo_paras=None, eboa_paras=None, model_rates=None):
        RootAlgo.__init__(self, root_algo_paras)
        self.epoch = eboa_paras["epoch"]
        self.pop_size = eboa_paras["pop_size"]
        self.model_rates=model_rates
        self.max_incubation_period=5
        self.epxilon = 0.0001
        self.SEARCHABLE_LIMIT=(self.pop_size*10)/100 # to swtich between random and linear infection in a population
        self.S, self.E, self.I, self.H, self.R, self.V, self.D, self.Q=[], [], [], [], [], [], [], []
        self.s_epoch, self.i_epoch, self.h_epoch, self.r_epoch, self.v_epoch, self.d_epoch, self.q_epoch=[], [], [], [], [], [], []
        self.solutions=[]
        self.S = [ (i, self._create_solution__(minmax=1)) for i in range(self.pop_size)]
        #print(self.S)
        #self._show_all_fits()
        self.a_scatter=None
        self.unpack_pop=[]
        
    def _set_scatter__(self, a_scatter):
        self.a_scatter=a_scatter
        
    def _get_initial_solutions__(self):
        return self._unslice_solutions__(self.S)
    
    def _unslice_solutions__(self, pop):
        unsliced_pop=[]
        for p in pop:
            idx, i=p
            unsliced_pop.append(i)
        return unsliced_pop
    
    def _unpack_actual_solutions_for_sort__(self, pop):
        self.unpack_pop=[]
        for p in pop:
            idx, i=p            
            individual=deepcopy(i[0])
            fit=deepcopy(i[1])
            self.unpack_pop.append([individual, fit, deepcopy(idx)])
        return self.unpack_pop
    
    def _pack_actual_solutions__(self):
        packed_pop=[]
        for p in self.unpack_pop:
            individual=deepcopy(i[0])
            fit=deepcopy(i[1])
            idx=deepcopy(i[2])
            packed_pop.append( (idx, [individual, fit]) )
        self.S=packed_pop
        packed_pop=[]
        self.unpack_pop=[]
        
    def _unslice_actual_solutions__(self, pop):
        unsliced_pop=[]
        for p in pop:
            i=p[0]
            unsliced_pop.append(i)
        return unsliced_pop
    
    def _train__(self):      
        timelines =self._create_initial_timelines__()
        PE=[(i,self._create_solution__()) for i in range(5)] # population of infected pathogens in environment PE
        # Find prey which is the best solution
        gbest = self._get_global_best__(pop=self._unpack_actual_solutions_for_sort__(self.S), id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
        self.E.append(self.S[0])
        self.I.append(self.S[0]) # add the first individual on the Susceptible list to the Infected list
        icase=self.I[0] # make that first individual on the Susceptible list as indexcase
        gbest=current_best=icase[1] # make the fitness value of indexcase the current & global best
        
        _, ind=self.S[0]
        self.S[0]= self._symbol_of_infected_individual__(), ind #Since the first susceptible individual is now infected, we mark him out of S
        
        pos=[] # initialize the  container storing positions of infected 
        pos.append(timelines[0]) #add the position of the selected indexcase
        
        incub=[]
        incub.append(generate_incubation_period_for_an_individual(self.max_incubation_period))
        
        #stores the probability values of each infected individual exceeding neighbourhood 
        prob=[]
        
        #we chose the exponential distribution because it models movement of infected person which rises
        #and then falls due to isolation upon detection.
        prob.append(expon.rvs(scale=1,loc=0,size=self.epoch)) 
        
        for i in range(self.epoch):
            #time = 2 - 2 * i / (self.epoch - 1)            # linearly decreased from 2 to 0
            
            diff_params = {"epoch": self.epoch, "S": self.S, "I": self.I, "H": self.H, "V": self.V, "R": self.R, "D": self.D, "PE": PE, "Q": self.Q}
            dif=DiffEquation(diffparams=diff_params, model_rates=self.model_rates)
            
            newI=[]
            inc_newI=[]
            prob_newI=[]
            pos_newI=[]
            
            #select a random number of infected individuals and isolate them
            if len(self.I) > 1:
                qrate=(equation12(dif, len(self.Q)))
                qrate=(np.abs(qrate))[i]
                qsize=math.ceil((qrate * len(self.I))+self.epxilon)  #
                qsize=random.randint(1, qsize)
                self._quarantine_infectecd_individuals__(qsize)            
            
            #limit the number of infected who can propagate the disease by eliminating quarantined indi from I
            sizeInfectable=len(self.I) - len(self.Q) #remove some for quarantine
            actualSizeInfectable=math.ceil((sizeInfectable*25)/100) #only a percentage can actually infect
            #print('A '+str(actualSizeInfectable)+' S='+str(sizeInfectable)+' I='+str(len(self.I))+'   Q='+str(len(self.Q)))
            #print('infectable='+str(sizeInfectable)+'  I='+str(len(self.I))+'  Q='+str(len(self.Q)))
            for j in range(actualSizeInfectable): #len(self.I)
                pos[j], drate=equation1(pos[j], max(pos)) # computes the new postion of infected individual at [j]
                d=incub[j] 
                
                if (d >= self.max_incubation_period) :                    
                    neighbourhood = prob[j][i] #probability that pos[j] exceeds NEIGHBOURHOOD_THRESHHOLD at time (i)
                    rate=(equation7(dif, len(self.I)))
                    rate=(np.abs(rate))[i] 
                    
                    newS=self._size_uninfected_susceptible__(self.S)
                    fracNewS=math.ceil((newS*0.5)/100) #two percent of newS
                    if neighbourhood < self.NEIGHBOURHOOD_THRESHHOLD:
                        size=math.ceil((0.1 * rate)+self.epxilon+ (fracNewS))     # add a fractiion of newS
                        indvd_change_factor=0.1 * rate
                    else :
                        size=math.ceil((0.7 * rate)+self.epxilon+ (fracNewS))              # add a fractiion of newS
                        indvd_change_factor=0.7 * rate
                    
                    s=newS
                    proposed_of_infected=random.randint(1, size)
                                        
                    #randomly pick the size_of_infected from Susceptible and make them now infected
                    tmp, size_of_infected=self._infect_susceptible_population__(proposed_of_infected, newS, indvd_change_factor, gbest)    
                    #print('genSize='+str(size)+' availS='+str(s)+' propIn='+str(proposed_of_infected)+' actualn='+str(size_of_infected))                        
                    for ni in range(size_of_infected):
                        #generate the incubation time for this newly infected individual
                        inc_newI.append(generate_incubation_period_for_an_individual(self.max_incubation_period))
                        #generate the probabilities value of neighbourhood for all epoch for this individual
                        prob_newI.append(expon.rvs(scale=1,loc=0,size=self.epoch))
                        #copy its initial position and store it
                        pos_newI.append(pos[j])
                        #Add the newly infected individual
                        newI.append(tmp[ni]) 
            
            self.I.extend(newI)
            incub.extend(inc_newI)
            prob.extend(prob_newI)
            pos.extend(pos_newI)
            
            infected_size=self._new_infected_change__(newi=self.I, eqtn=equation8(dif, len(newI)), e=i, fl='h')
            #print('H >>newI '+str(len(newI))+' size_of_infected='+str(infected_size))                        
            h =self._hospitalize_infected_population__(self.I, infected_size)    
            self.H =[self.H.append(h[i]) for i in range(len(h))]
            
            infected_size=self._new_infected_change__(newi=h, eqtn=equation10(dif, len(h)), e=i, fl='v')
            #print('V >>h '+str(len(h))+' size_of_infected='+str(infected_size))                        
            v =self._vaccinate_hospitalized_population__(h, infected_size)    
            self.V =[self.V.append(v[i]) for i in range(len(v))]
            
            infected_size=self._new_infected_change__(newi=self.I, eqtn=equation9(dif, len(newI)), e=i, fl='r')
            #print('R >>newI '+str(len(newI))+' size_of_infected='+str(infected_size))                        
            r =self._recover_infected_population__(self.I, infected_size)    
            self.R =[self.R.append(r[i]) for i in range(len(r))]
            if r:
                self.I, incub, prob, pos=self._remove_dead_or_recovered_from_infected__(infected=self.I, recovdead=r, incub=incub, prob=prob, pos=pos)
            self._addback_recovered_2_susceptible__(deepcopy(r))
                
            infected_size=self._new_infected_change__(newi=self.I, eqtn=equation11(dif, len(newI)), e=i, fl='d')
            #print('D >>newI '+str(len(newI))+' size_of_infected='+str(infected_size))                        
            d =self._die_infected_population__(self.I, infected_size)    
            self.D =[self.D.append(d[i]) for i in range(len(d))]
            if d:
                self.I, incub, prob, pos=self._remove_dead_or_recovered_from_infected__(infected=self.I, recovdead=d, incub=incub, prob=prob, pos=pos)
            self._rebirth_2replace_dead_in_susceptible__(deepcopy(d)) 
            
            #self._show_all_fits()
            current_best = self._get_global_best__(pop=self._unpack_actual_solutions_for_sort__(self.S), id_fitness=self.ID_FIT, id_best=self.ID_MIN_PROBLEM)
            if current_best[self.ID_FIT] < gbest[self.ID_FIT]:
                gbest = deepcopy(current_best)
            self.solutions.append(gbest[self.ID_FIT])
            
            self.s_epoch.append(self._size_uninfected_susceptible__(self.S))
            self.i_epoch.append(len(self.I))
            self.h_epoch.append(len(self.H))
            self.r_epoch.append(len(self.R))
            self.v_epoch.append(len(self.V))
            self.d_epoch.append(len(self.D))
            self.q_epoch.append(len(self.Q))
            self.Q=[]
            self.a_scatter.update(self._unslice_actual_solutions__(self._unslice_solutions__(self.S)))
            print("Epoch = {}, Infected = {}, Best fit so far = {}".format(i + 1, len(self.I), gbest[self.ID_FIT]))
        
        #self._show_all_fits()
        #print(self.solutions)
        return gbest[self.ID_POS], self.solutions
    
    def _show_all_fits(self, sols=None):
        #print('------------------------------------------------------------')
        for s in self.S:
            idx, idv=s
            print(idv[1])
            
    def _quarantine_infectecd_individuals__(self, qsize=0):
        for i in range(qsize):
            self.Q.append(self.I[i])
    
    def _remove_dead_or_recovered_from_infected__(self, infected=None, recovdead=None, incub=None, prob=None, pos=None):        
        tmp_infected=[]
        tmp_incub=[]
        tmp_prob=[]
        tmp_pos=[]
        already_selected=[]
        
        indexs=[]
        for i in range(len(recovdead)):            
            idx_r, dr_individ=recovdead[i]
            indexs.append(idx_r)
        #print(str(indexs)+' toRecoverORdead  <<<<>>>> '+str(len(indexs)))
            
        for i in range(len(recovdead)):            
            idx_r, dr_individ=recovdead[i]
            #print('idx_r  <**** '+str(idx_r))
            for j in range(len(infected)):
                idx_i, i_individ=infected[j]  #(i_individ[0] != dr_individ[0]).all()
                if  idx_i != idx_r and idx_i not in already_selected and idx_i not in indexs:
                    tmp_infected.append(infected[j])
                    tmp_incub.append(incub[j])
                    tmp_prob.append(prob[j])
                    tmp_pos.append(pos[j])
                    #print('idx_i  ==>'+str(idx_i)+'  idx_r  ==> '+str(idx_r))
                    already_selected.append(idx_i)
                #else:
                   #print(str(already_selected)+' idx_i  <**** '+str(idx_i)+'  idx_r  <**** '+str(idx_r)) 
        
        #print(str(len(infected))+'  '+str(len(tmp_infected)))                    
        return tmp_infected, tmp_incub, tmp_prob, tmp_pos
    
    def _symbol_of_infected_individual__(self):
        solution = None #[np.random.uniform(0, 0, self.problem_size)]
        fit=0
        return solution#, fit
    
    def _new_infected_change__(self, newi=None, eqtn=None,  e=None, fl=None):
        equat_value=(eqtn)
        rate=(np.abs(equat_value))[e]
        rate=0 if math.isnan(rate) else rate
        maxi=math.ceil((0.1 * rate * len(newi))) #+self.epxilon
        infected_size=random.randint(0, maxi)
        return infected_size
        
    def _size_uninfected_susceptible__(self, pop=None):
        suscp=[]
        for i in range(len(pop)):
            x, individ=pop[i]
            if x is not None:
                suscp.append(pop[i])
        return len(suscp)
    
    def _remove_infected_individuals_from_S__(self, pop=None):
        suscp=[]
        for i in range(len(pop)):
            x, individ=pop[i]
            if x is not None:#(individ == self._symbol_of_infected_individual__()).all():
                suscp.append(pop[i])
        return suscp 
    
    def _addback_recovered_2_susceptible__(self, recovered=None):
         #if an individual recovers, change its status from None to original INDEX before infection
         #but don't change the fit/genetic of the individual
        for r in recovered:
            r_indx, r_individual=r
            for s in self.S:
                s_indx, s_individual=s 
                if  (s_individual[0] == r_individual[0]).all() and s_indx is None:
                    #print(str(r_indx)+' index recovered '+str(len(recovered)))
                    self.S[r_indx]=(r_indx, s_individual)        
    
    def _rebirth_2replace_dead_in_susceptible__(self, dead=None): 
        #if an individual dies, birth a new individual entirly to replace the dead
        #Then change its status from None to original INDEX before infection and death
        for d in dead:
            d_indx, d_individual=d
            for s in self.S:
                s_indx, s_individual=s
                if (s_individual[0] == d_individual[0]).all() and s_indx is None:
                    new_solution=self._create_solution__(minmax=self.MIN_MAX_INFECTED_SOL)
                    #print(str(d_indx)+' index rebirth '+str(len(dead)))
                    self.S[d_indx]=(d_indx, new_solution)
    
    def _die_infected_population__(self, population=None, size_of_infected=None):
        f = lambda x: random.randint(0, (x))  
        tmp=[]
        pop_size=len(population)-1
        for _ in range(size_of_infected):
            x=f(pop_size)
            tmp.append(deepcopy(population[x]))
        return tmp
    
    def _hospitalize_infected_population__(self, population=None, size_of_infected=None):
        f = lambda x: random.randint(0, (x))  
        tmp=[]
        pop_size=len(population)-1
        for _ in range(size_of_infected):
            x=f(pop_size)
            tmp.append(deepcopy(population[x]))
        return tmp
    
    def _vaccinate_hospitalized_population__(self, population=None, size_of_infected=None):
        f = lambda x: random.randint(0, (x))  
        tmp=[]
        pop_size=len(population)-1
        for _ in range(size_of_infected):
            x=f(pop_size)
            tmp.append(deepcopy(population[x]))
        return tmp
    
    def _recover_infected_population__(self, population=None, size_of_infected=None):
        f = lambda x: random.randint(0, (x))  
        tmp=[]
        pop_size=len(population)-1
        for _ in range(size_of_infected):
            x=f(pop_size)
            tmp.append(deepcopy(population[x]))
        return tmp
    
    def _infect_susceptible_population__(self, size_to_infect=None, uninfectedS=None, indvd_change_factor=None, gbest=None):
        f = lambda x: random.randint(0, (x))  
        tmp=[]
                
        diff=uninfectedS-size_to_infect
        if diff <= 0:
            size_to_infect=uninfectedS
            
        pop_size=len(self.S)-1
            
        for _ in range(size_to_infect):            
            if uninfectedS <= self.SEARCHABLE_LIMIT: #linearly search for candidate to infect since pop is small
                for j in range(pop_size+1):                    
                    idx, individual=self.S[j]                
                    if idx is not None:
                        x=j
                    #else:
                        #self._boost_imunity_for_infection_escpades__(j, indvd_change_factor, gbest)
            else: #randomly infect since uninfected population is still large
                while True: #to ensure that we do not select an index which will return already infected
                    x=f(pop_size)
                    idx, individual=self.S[x]                
                    if idx is not None:
                        break
                    #else:
                        #self._boost_imunity_for_infection_escpades__(x, indvd_change_factor, gbest)
                         
            original_index, individual=self._weaken_imunity_of_infected__(x, indvd_change_factor, gbest)
            tmp.append( (original_index, deepcopy(individual)) )
            
            
        return tmp, size_to_infect
    
    def _boost_imunity_for_infection_escpades__(self, x, indvd_change_factor, gbest):
        #boost the imunity and self-protectionism of those individual who escapes infection
        escape_index, escape_ix=self.S[x]
        v = np.abs((np.random.rand()* indvd_change_factor) * (gbest[self.ID_POS]- deepcopy(escape_ix[self.ID_POS])))
        v = self.domain_range[0] if (v < self.domain_range[0]).all() else v
        v = self.domain_range[1] if (v > self.domain_range[1]).all() else v           
        escape_infected_ind= v
        escape_fit_infected = self._fitness_model__(solution=escape_infected_ind, minmax=self.MIN_MAX_INFECTED_SOL)
        self.S[x]=escape_index, [escape_infected_ind, escape_fit_infected]
    
    def _weaken_imunity_of_infected__(self, x, indvd_change_factor, gbest):
        #weakens the imunity and self-protectionism of those individual who are infection
        original_index, ix=self.S[x] 
        l = np.random.uniform(-1, 1)
        v=indvd_change_factor * np.exp(1 * l) * np.cos(2 * np.pi * l) + gbest[self.ID_POS]
        infected_ind= v
        fit_infected = self._fitness_model__(solution=infected_ind, minmax=self.MIN_MAX_INFECTED_SOL)
        individual=[infected_ind, fit_infected]
        self.S[x]=self._symbol_of_infected_individual__(), individual #since it has been selected, mark a None i.e, infected individual                
        return original_index, individual