from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Item(ABC):
    
    def __init__(self, name, init = None) -> None:
        self.name = name
        self.prior = init
    
    @abstractmethod
    def transition(self):
        pass

    @abstractmethod
    def simulate(self, n = 10):
        pass

class ConstantItem(Item):
    
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value
    
    def transition(self):
        return (self.value, self.value)

    def simulate(self, n = 10):
        return ([self.value]*n, [self.value]*n)

class DiscreteItem(Item):
    def __init__(self, name, states, transition_matrix, init = None) -> None:
        
        super().__init__(name)
        
        # record keeping
        self.states = states
        self._state_names = list(self.states.keys())
        self._state_values = list(self.states.items())
        
        # establish initial state
        if init is not None:
            if callable(init):
                self.prior = init()
            else:
                self.prior = init
        else:
            self.prior = np.random.choice(self._state_names)
        
        # cast to np.matrix    
        try:
            self.transition_matrix = np.asmatrix(transition_matrix)
        except:
            raise
        
        # convert to a square matrix if a single column or row is provided
        if (self.transition_matrix.shape[0] == 1):
            row = self.transition_matrix
            self.transition_matrix = np.vstack([row]*row.shape[1])
        elif (self.transition_matrix.shape[1] == 1):
            column = self.transition_matrix
            row = column.T
            self.transition_matrix = np.vstack([row]*row.shape[1])
        
        # check that this is proper transition matrix
        # each row should sum to 1.0
        assert np.all(self.transition_matrix.sum(axis = 1) == 1.0), \
            f'This is not a proper transition matrix:\n {self.transition_matrix}'
        
        # check that the dimensions of states and transition matrix align
        assert len(self.states) == len(self.transition_matrix), \
            f'The dimensions of the states and the transition matrix do not align:\n' + \
            f'States: \n' + \
            f'\t{self.states}' + \
            f'Transition Matrix:\n' + \
            f'\t{self.transition_matrix}'
        
    def transition(self):
        prior_pos = self._state_names.index(self.prior)
        transition_probs = np.ravel(self.transition_matrix[prior_pos])
        next_state_name = np.random.choice(self._state_names, p = transition_probs)
        next_state_value = self.states[next_state_name]
        self.prior = next_state_name
        
        return (next_state_name, next_state_value)

    def simulate(self, n = 10):
        results = []
        for i in range(n):
            event = self.transition()
            results.append(event)
        state_names = [x[0] for x in results]
        state_values = [x[1] for x in results]
        return (state_names, state_values)


class ContinuousItem(Item):
    def __init__(self, name, transition_fx, init = None) -> None:
        
        super().__init__(name)
        self.transition_fx = transition_fx
        
        # establish initial state
        if init is not None:
            if callable(init):
                self.prior = init()
            else:
                self.prior = init
        else:
            assert not ('prior' in self.transition_fx.__code__.co_varnames), \
                f'Must provide initial value for transition function.'
        
    def transition(self):
        if ('prior' in self.transition_fx.__code__.co_varnames):
            res = self.transition_fx(prior = self.prior)
        else:
            res = self.transition_fx()
        self.prior = res
        
        # to match the format of DiscreteItem
        return (res, res)

    def simulate(self, n = 10):
        
        results = []
        for i in range(n):
            event = self.transition()
            results.append(event)
        state_values = [x[0] for x in results]
        return (state_values, state_values)

class Budget(Item):
    
    def __init__(self, *items) -> None:
        self.items = items
    
    def transition(self):
        res = []
        for i in self.items:
            res.append(i.transition())
        state_names = [x[0] for x in res]
        state_values = [x[1] for x in res]
        return (state_names, state_values)

    def simulate(self, n = 10):
        res = []
        for i in self.items:
            res.append(i.simulate(n))
        
        # make state names dataframe
        state_names = {item.name: item_res[0] for item, item_res in zip(self.items, res)}
        df_state_names = pd.DataFrame(state_names, )
        df_state_names.columns = [i.name for i in self.items]
        df_state_names.index.name = 'time'
        
        # make state values dataframe
        state_values = {item.name: item_res[1] for item, item_res in zip(self.items, res)}
        df_state_values = pd.DataFrame(state_values)
        df_state_values.columns = [i.name for i in self.items]
        df_state_values.index.name = 'time'
        
        return (df_state_names, df_state_values)

