import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from custommodels.SpreadingModels import SIRModelBase, ClassicalModel
from custommodels.Pickle import pickle, load
from copy import deepcopy

#region load graph
path = "./data/loc-brightkite_edges.txt.gz"
G = nx.read_edgelist(path, create_using=nx.Graph())
largest_cc = max(nx.connected_components(G), key=len)
len(largest_cc)
G = G.subgraph(largest_cc)
#endregion

#region load data
sorted_degree = load('sorted_degree', relative_path=False)
betweenness_centrality = load('betweenness_centrality', relative_path=False)
sorted_betweenness = sorted(list(betweenness_centrality.items()), key=lambda x: x[1], reverse=True)
#endregion

#region Helper functions
def results_to_noderatios(results, num_initial_infected, G):
    # results is a list of tuples (new_state: dict, newly_infected: list, newly_recovered: list)
    # this turns it into a list of lists, [susceptible: int[], infected: int[], recovered: int[]]
    # the 0th element in each list is the first step, the 1st element is the second step, etc.
    n = len(G.nodes)
    num_susceptible = [n - num_initial_infected]
    num_infected = [num_initial_infected]
    num_recovered = [0]
    for state, i, r in results:
        num_susceptible.append(list(state.values()).count(0))
        num_infected.append(list(state.values()).count(1))
        num_recovered.append(list(state.values()).count(2))
    return np.array(num_susceptible)/n, np.array(num_infected)/n, np.array(num_recovered)/n

_n = 10
def disjoint_sets(n = None):
    n = n or _n

    sorted_betweenness_copy = deepcopy(sorted_betweenness)
    sorted_degree_copy = deepcopy(sorted_degree)
    top_n_deg = []
    top_n_bet = []
    while True:
        if len(top_n_deg) == n and len(top_n_bet) == n:
            break

        deg_candidate = sorted_degree_copy.pop(0)
        bet_candidate = sorted_betweenness_copy.pop(0)

        if str(deg_candidate[0]) == str(bet_candidate[0]):
            continue
        if str(deg_candidate[0]) not in top_n_bet and len(top_n_deg) < n:
            top_n_deg.append(str(deg_candidate[0]))
        if str(bet_candidate[0]) not in top_n_deg and len(top_n_bet) < n:
            top_n_bet.append(str(bet_candidate[0]))
    
    sorted_betweenness_copy = deepcopy(sorted_betweenness)
    sorted_degree_copy = deepcopy(sorted_degree)
    bot_n_deg = []
    bot_n_bet = []
    while True:
        if len(bot_n_deg) == n and len(bot_n_bet) == n:
            break

        deg_candidate = sorted_degree_copy.pop(-1)
        bet_candidate = sorted_betweenness_copy.pop(-1)
        if str(deg_candidate[0]) == str(bet_candidate[0]):
            continue
        if str(deg_candidate[0]) not in bot_n_bet and len(bot_n_deg) < n:
            bot_n_deg.append(str(deg_candidate[0]))
        if str(bet_candidate[0]) not in bot_n_deg and len(bot_n_bet) < n:
            bot_n_bet.append(str(bet_candidate[0]))
    
    return top_n_deg, top_n_bet, bot_n_deg, bot_n_bet

def betweenness_from_nodelist(nl):
    # Returns tuples (node, betweenness)
    return [(node, betweenness_centrality[int(node)]) for node in nl]
def degree_from_nodelist(nl):
    # Returns tuples (node, degree)
    return [(node, G.degree[node]) for node in nl]

def get_extreme_deg(n = None):
    n = n or _n

    top_n_degree = [str(node) for node, _ in sorted_degree[:n]]
    bottom_n_degree = [str(node) for node, _ in sorted_degree[-n-1:-1]]
    return top_n_degree, bottom_n_degree

def get_extreme_bet(n = None):
    n = n or _n

    top_n_betweenness = [str(node) for node, _ in sorted_betweenness[:n]]
    bottom_n_betweenness = [str(node) for node, _ in sorted_betweenness[-n-1:-1]]
    return top_n_betweenness, bottom_n_betweenness

# un is for union
def undisjoint(n = None):
    n = n or _n

    # Get top n*2 degree nodes
    top_deg, bot_deg = get_extreme_deg(n*2)
    # Sort them by betweenness
    top_deg_bet = betweenness_from_nodelist(top_deg)
    bot_deg_bet = betweenness_from_nodelist(bot_deg)
    top_deg_bet.sort(key=lambda x: x[1], reverse=True)
    bot_deg_bet.sort(key=lambda x: x[1], reverse=False)
    # Get top n degree nodes
    top_deg_bet = [node for node, _ in top_deg_bet[:n]]
    bot_deg_bet = [node for node, _ in bot_deg_bet[:n]]
    
    return top_deg_bet, bot_deg_bet


random_infect = deepcopy(list(G.nodes()))
random.shuffle(random_infect)

random_recovered = deepcopy(list(G.nodes()))
random.shuffle(random_recovered)

def get_random_for_infection(n=None):
    n = n or _n

    return random_infect[:n]
    

def get_random_for_vaccine(n=None):
    n = n or _n

    return random_recovered[:n]

def results_to_noderatios(results, num_initial_infected, G):
    # results is a list of tuples (new_state: dict, newly_infected: list, newly_recovered: list)
    # this turns it into a list of lists, [susceptible: int[], infected: int[], recovered: int[]]
    # the 0th element in each list is the first step, the 1st element is the second step, etc.
    n = len(G.nodes)
    num_susceptible = [n - num_initial_infected]
    num_infected = [num_initial_infected]
    num_recovered = [0]
    for state, i, r in results:
        num_susceptible.append(list(state.values()).count(0))
        num_infected.append(list(state.values()).count(1))
        num_recovered.append(list(state.values()).count(2))
    return np.array(num_susceptible)/n, np.array(num_infected)/n, np.array(num_recovered)/n
#endregion

its = 50
def plot_models(initial_infected: list, initial_vaccinated: list, lineargs: list, title: str = None, show = True):
    """The mother of all plot functions"""
    if not len(initial_infected) == len(initial_vaccinated) == len(lineargs):
        raise ValueError("initial_infected, initial_vaccinated, and linestyles must be the same length")
    
    models = [getmodel() for _ in range(len(initial_infected))]
    for i, model in enumerate(models):
        print(f"Plotting model {i+1}/{len(models)}", end='\r')
        model.set_initial_infected(initial_infected[i])
        if initial_vaccinated[i]:
            model.set_initial_recovered(initial_vaccinated[i])

        results = model.iterate(its)
        sus, inf, rec = results_to_noderatios(results, len(initial_infected[i]), G)
        
        sus_args, inf_args, rec_args = lineargs[i]
        if sus_args:
            plt.plot(sus, **sus_args)
        if inf_args:
            plt.plot(inf, **inf_args)
        if rec_args:
            plt.plot(rec, **rec_args)
        print()
    plt.legend(loc = 'upper right')
    plt.xlabel("Time")
    plt.ylabel("Fraction of nodes")
    if title:
        plt.title(title)
    # save to file
    plt.savefig(f"./data/plots/{title}.svg", format="svg")
    if show:
        plt.show()
    plt.clf()

#region model stuff
beta, gamma = .4, .05
def getmodel():
    mod = ClassicalModel(G)
    mod.set_parameters(beta, gamma)
    return mod
#endregion

skip = True
#region Infected, top 10, random, bottom 10
if not skip:
    top_10_deg, bot_10_deg = get_extreme_deg()
    top_10_bet, bot_10_bet = get_extreme_bet()
    rnd_10 = get_random_for_infection()

    deg_infs = [top_10_deg, bot_10_deg, rnd_10]
    deg_vacs = [None, None, None]
    deg_lineargs = [
        # susceptible line, infected line, recovered line
        [None, {'label': 'Top 10 degree infected', 'color': '#3b4b0e'}, None], # top_10_deg
        [None, {'label': 'Bottom 10 degree infected', 'color': '#688419'}, None], # bot_10_deg
        [None, {'label': 'Random 10 infected', 'color': '#a9c950'}, None] # rnd_10
    ]

    bet_infs = [top_10_bet, bot_10_bet, rnd_10]
    bet_vacs = [None, None, None]
    bet_lineargs = [
        # susceptible line, infected line, recovered line
        [None, {'label': 'Top 10 betweenness infected', 'color': '#3b4b0e'}, None], # top_10_bet
        [None, {'label': 'Bottom 10 betweenness infected', 'color': '#688419'}, None], # bot_10_bet
        [None, {'label': 'Random 10 infected', 'color': '#a9c950'}, None] # rnd_10
    ]

    print("Plotting degree models")
    plot_models(
        deg_infs,
        deg_vacs,
        deg_lineargs,
        title='Infection of top 10, bottom 10, and random 10 nodes by degree'
    )
    print("Plotting betweenness models")
    plot_models(
        bet_infs,
        bet_vacs,
        bet_lineargs,
        title='Infection of top 10, bottom 10, and random 10 nodes by betweenness'
    )
#endregion

skip = True 
#region Infected, disjoint top 10, random, disjoint bottom 10
if not skip:
    top_n_deg, top_n_bet, bot_n_deg, bot_n_bet = disjoint_sets()
    rnd_10 = get_random_for_infection()

    deg_infs = [top_n_deg, bot_n_deg, rnd_10]
    deg_vacs = [None, None, None]
    deg_lineargs = [
        # susceptible line, infected line, recovered line
        [None, {'label': 'Disjoint top 10 degree infected', 'color': '#3b4b0e'}, None], # top_n_deg
        [None, {'label': 'Disjoint bottom 10 degree infected', 'color': '#688419'}, None], # bot_n_deg
        [None, {'label': 'Random 10 infected', 'color': '#a9c950'}, None] # rnd_10
    ]

    bet_infs = [top_n_bet, bot_n_bet, rnd_10]
    bet_vacs = [None, None, None]
    bet_lineargs = [
        # susceptible line, infected line, recovered line
        [None, {'label': 'Disjoint top 10 betweenness infected', 'color': '#3b4b0e'}, None], # top_n_bet
        [None, {'label': 'Disjoint bottom 10 betweenness infected', 'color': '#688419'}, None], # bot_n_bet
        [None, {'label': 'Random 10 infected', 'color': '#a9c950'}, None] # rnd_10
    ]

    print("Plotting degree models disjoint")
    plot_models(
        deg_infs,
        deg_vacs,
        deg_lineargs,
        title = 'Infection by degree - disjoint'
    )
    print("Plotting betweenness models disjoint")
    plot_models(
        bet_infs,
        bet_vacs,
        bet_lineargs,
        title = 'Infection by betweenness - disjoint'
    )
#endregion

skip = True
#region Immune normal
if not skip:
    n_vax = 5000
    n_inf = 10
    rnd_n = get_random_for_infection(n_inf)
    rnd_n_vac = get_random_for_vaccine(n_vax)
    top_n_bet, bot_n_bet = get_extreme_bet(n_vax)
    top_n_deg, bot_n_deg = get_extreme_deg(n_vax)

    infs = [rnd_n, rnd_n, rnd_n, rnd_n]
    vacs = [top_n_deg, top_n_bet, rnd_n_vac, None]
    lineargs = [
        [None, {'label': f'Top {n_vax} degree vaxxd'}, None],
        [None, {'label': f'Top {n_vax} betweenness vaxxd'}, None],
        [None, {'label': f'Random {n_vax} vaxxd'}, None],
        [None, {'label': f'No vaccination'}, None]
    ]
    print("Plotting immune normal")
    plot_models(
        infs,
        vacs,
        lineargs,
        title=f'{n_inf} infected, top {n_vax} vaxx\'d by degree, betweenness, and random'
    )
#endregion

skip = True  
#region Immune disjoint
if not skip:
    n_vax = 5000
    n_inf = 10
    rnd_n = get_random_for_infection(n_inf)
    rnd_n_vac = get_random_for_vaccine(n_vax)
    top_n_deg, top_n_bet, bot_n_deg, bot_n_bet = disjoint_sets(n_vax)

    infs = [rnd_n, rnd_n, rnd_n, rnd_n]
    vacs = [top_n_deg, top_n_bet, rnd_n_vac, None]
    lineargs = [
        [None, {'label': f'Top {n_vax} degree','color': '#94bc24'}, None],
        [None, {'label': f'Top {n_vax} betweenness','color': '#688419'}, None],
        [None, {'label': f'Random {n_vax}','color': '#cade92'}, None],
        [None, {'label': f'No vaccination','color': '#243444'}, None]
    ]
    print("Plotting immune disjoint")
    plot_models(
        infs,
        vacs,
        lineargs,
        title=f'Immunity based on degree, betweenness, and random - disjoint'
    )
#endregion

skip = True   
#region Immune joint
if not skip:
    n_vax = 5000
    n_inf = 10
    rnd_n = get_random_for_infection(n_inf)
    rnd_n_vac = get_random_for_vaccine(n_vax)
    top, bot = undisjoint(n_vax)
    top_n_deg, top_n_bet, bot_n_deg, bot_n_bet = disjoint_sets(n_vax)

    infs = [rnd_n, rnd_n, rnd_n, rnd_n, rnd_n]
    vacs = [top, rnd_n_vac, top_n_deg, top_n_bet, None]
    lineargs = [
        [None, {'label': f'Top {n_vax} joint', 'color': '#2494bc'}, None],
        [None, {'label': f'Random {n_vax}','color': '#cade92'}, None],
        [None, {'label': f'Top {n_vax} degree','color': '#94bc24'}, None],
        [None, {'label': f'Top {n_vax} betweenness','color': '#688419'}, None],
        [None, {'label': f'No vaccination','color': '#243444'}, None]
    ]
    print("Plotting immune disjoint")
    plot_models(
        infs,
        vacs,
        lineargs,
        title=f'Immunity based on degree, betweenness, joint and random - disjoint'
    )
#endregion

skip = True
#region Immune test from 5000
if not skip:
    n_inf = 10
    infs = []
    vacs = []
    lineargs = []

    vax_values = (5000, 1000, 100)
    colors = ('orangered', 'mediumturquoise', 'violet')
    for n_vax, color in zip(vax_values, colors):
        rnd_n = get_random_for_infection(n_inf)
        rnd_n_vac = get_random_for_vaccine(n_vax)
        top_n_bet, bot_n_bet = get_extreme_bet(n_vax)
        top_n_deg, bot_n_deg = get_extreme_deg(n_vax)

        infs = infs + [rnd_n, rnd_n, rnd_n]
        vacs = vacs + [top_n_deg, top_n_bet, rnd_n_vac]

        lineargs = lineargs + [
            [None, {'label': f'Top {n_vax} degree vaxxd', 'color': color}, None],
            [None, {'label': f'Top {n_vax} betweenness vaxxd', 'color': color, 'linestyle': 'dashed'}, None],
            [None, {'label': f'Random {n_vax} vaxxd', 'color': color, 'linestyle': 'dotted'}, None]
        ]
    print("Plotting immune not disjoint with vax")
    plot_models(
        infs,
        vacs,
        lineargs,
        title=f'Vaccinating top nodes with n = {vax_values}, both between and degree'
    )
#endregion

skip = True
#region Immune test from 5000
if not skip:
    n_inf = 10
    infs = []
    vacs = []
    lineargs = []

    vax_values = (5000, 1000, 100)
    colors = ('#2494bc', '#94bc24', '#688419')
    for n_vax, color in zip(vax_values, colors):
        rnd_n = get_random_for_infection(n_inf)
        rnd_n_vac = get_random_for_vaccine(n_vax)
        top, bot = undisjoint(n_vax)

        infs = infs + [rnd_n, rnd_n]
        vacs = vacs + [top, rnd_n_vac]

        lineargs = lineargs + [
            [None, {'label': f'Top {n_vax} combined vaxxd', 'color': color}, None],
            [None, {'label': f'Random {n_vax} vaxxd', 'color': color, 'linestyle': 'dotted'}, None]
        ]
    print("Plotting immune undisjoint with vax")
    plot_models(
        infs,
        vacs,
        lineargs,
        title=f'Vaccinating top nodes with n = {vax_values}, combined between and degree'
    )
#endregion

skip = False
#region degree and betweenness, all lines, 5k vax, 10 infected
if not skip:
    n_vax = 5000
    n_inf = 10
    rnd_n = get_random_for_infection(n_inf)
    top_n_deg, top_n_bet, bot_n_deg, bot_n_bet = disjoint_sets()

    infs_deg = [rnd_n]
    infs_bet = [rnd_n]
    vacs_deg = [top_n_deg]
    vacs_bet = [top_n_bet]
    lineargs = [
        [
            { 'label': f'Susceptible' },
            { 'label': f'Infected' },
            { 'label': f'Recovered' }
        ]
    ]

    print("Plotting degree")
    plot_models(
        infs_deg,
        vacs_deg,
        lineargs,
        title=f'Infection by degree - disjoint'
    )
    print("Plotting betweenness")
    plot_models(
        infs_bet,
        vacs_bet,
        lineargs,
        title=f'Infection by betweenness - disjoint'
    )
#endregion