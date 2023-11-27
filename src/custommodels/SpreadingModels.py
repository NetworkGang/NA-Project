import random

class SIRModelBase:
    def __init__(self, G) -> None:
        self.G = G
        self.t = 0
        self.infected = set()
        self.susceptible = set(self.G.nodes)
        self.recovered = set()
    
    def iterate(self, n: int):
        n = 1 if n is None else n
        results = []
        for _ in range(n):
            new_state, newly_infected, newly_recovered = self._iterate()
            results.append((new_state, newly_infected, newly_recovered))
        if n == 1:
            return results[0]
        else:
            return results
    
    def try_recover(self, node):
        return random.random() < self.gamma

class CascadeModel(SIRModelBase):
    def __init__(self, G) -> None:
        super().__init__(G)
    
    def set_initial_infected(self, initial_infected: list):
        """Set the initial infected nodes. initial_infected is a list of node ids as STRINGS."""
        self.infected = set(initial_infected)
        self.susceptible = set(self.G.nodes) - self.infected
        self.recovered = set()
    
    def set_parameters(self, beta: float, gamma: float):
        """Set the parameters of the model. Beta is the infection rate, gamma is the recovery rate."""
        self.beta = beta
        self.gamma = gamma
    
    def _iterate(self):
        self.t += 1
        _newly_infected = set()
        _newly_recovered = set()
        for node in self.G.nodes:
            if node in self.susceptible and self.try_infect(node):
                _newly_infected.add(node)
                continue

            if node in self.infected and self.try_recover(node):
                _newly_recovered.add(node)
                continue

            if node in self.recovered:
                continue
        new_suceptible = self.susceptible - _newly_infected
        new_infected = self.infected.union(_newly_infected) - _newly_recovered
        new_recovered = self.recovered.union(_newly_recovered)
        state = {i: 0 for i in new_suceptible}
        state.update({i: 1 for i in new_infected})
        state.update({i: 2 for i in new_recovered})
        self.susceptible = new_suceptible
        self.infected = new_infected
        self.recovered = new_recovered
        return state, _newly_infected, _newly_recovered

    def try_infect(self, node):
        neighbors = set(self.G.neighbors(node))
        infected_neighbors = neighbors.intersection(self.infected)
        fraction = len(infected_neighbors) / len(neighbors)
        
        return fraction >= self.beta

class ThresholdModel(SIRModelBase):
    def __init__(self, G) -> None:
        super().__init__(G)
    
    def set_initial_infected(self, initial_infected: list):
        """Set the initial infected nodes. initial_infected is a list of node ids as STRINGS."""
        self.infected = set(initial_infected)
        self.susceptible = set(self.G.nodes) - self.infected
        self.recovered = set()
    
    def set_parameters(self, theta: float, gamma: float):
        """Set the parameters of the model. Theta is the infection threshold, gamma is the recovery rate."""
        self.theta = theta
        self.gamma = gamma
    
    def _iterate(self):
        self.t += 1
        _newly_infected = set()
        _newly_recovered = set()
        for node in self.G.nodes:
            if node in self.susceptible and self.try_infect(node):
                _newly_infected.add(node)
                continue

            if node in self.infected and self.try_recover(node):
                _newly_recovered.add(node)
                continue

            if node in self.recovered:
                continue
        new_suceptible = self.susceptible - _newly_infected
        new_infected = self.infected.union(_newly_infected) - _newly_recovered
        new_recovered = self.recovered.union(_newly_recovered)
        state = {i: 0 for i in new_suceptible}
        state.update({i: 1 for i in new_infected})
        state.update({i: 2 for i in new_recovered})
        self.susceptible = new_suceptible
        self.infected = new_infected
        self.recovered = new_recovered
        return state, _newly_infected, _newly_recovered

    def try_infect(self, node):
        neighbors = set(self.G.neighbors(node))
        infected_neighbors = neighbors.intersection(self.infected)
        
        return len(infected_neighbors) >= self.theta