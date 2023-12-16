import random

class SIRModelBase:
    def __init__(self, G) -> None:
        self.G = G
        self.t = 0
        self.infected = set()
        self.susceptible = set(self.G.nodes)
        self.recovered = set()

        self._node_datatype = list(G.nodes)[0].__class__

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
    
    def set_initial_infected(self, initial_infected: list):
        """Set the initial infected nodes. initial_infected is a list of node ids as STRINGS."""
        self.infected = self.infected.union(set(initial_infected))
        self.susceptible = self.susceptible - self.infected
        self.recovered = self.recovered - self.infected

        if self._node_datatype != initial_infected[0].__class__:
            print(f"WARNING: This model was created using a graph with nodes of type {self._node_datatype}, but the initial infected nodes are of type {initial_infected[0].__class__}. This may cause unexpected behaviour.")

    def set_initial_recovered(self, initial_recovered: list):
        """Set the initial recovered nodes. initial_recovered is a list of node ids as STRINGS."""
        self.recovered = self.recovered.union(set(initial_recovered))
        self.susceptible = self.susceptible - self.recovered
        self.infected = self.infected - self.recovered

        if self._node_datatype != initial_recovered[0].__class__:
            print(f"WARNING: This model was created using a graph with nodes of type {self._node_datatype}, but the initial recovered nodes are of type {initial_recovered[0].__class__}. This may cause unexpected behaviour.")

class CascadeModel(SIRModelBase):
    def __init__(self, G) -> None:
        super().__init__(G)

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
    
    def set_parameters(self, theta: float, beta: float, gamma: float):
        """Set the parameters of the model. Theta is the infection threshold, gamma is the recovery rate. Beta is the infection rate"""
        self.theta = theta
        self.gamma = gamma
        self.beta = beta
    
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
        
        if len(infected_neighbors) >= self.theta:
            return random.random() < self.beta


class ClassicalModel(SIRModelBase):
    def __init__(self, G) -> None:
        super().__init__(G)
    
    def set_parameters(self, beta: float, gamma: float):
        """Set the parameters of the model. gamma is the recovery rate, beta is the infection rate"""

        self.gamma = gamma
        self.beta = beta
    
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
        return random.random() < (1 - ((1 - self.beta)**len(infected_neighbors)) ) # sna p 253

    