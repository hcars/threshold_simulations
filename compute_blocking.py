import copy
import csv
from os.path import exists

import networkx as nx
import numpy as np

import coverage_heuristic as cbh
import utils


def choose_seed(core, seed_size):
    component = np.random.choice(core, seed_size, replace=False)
    seed_set_1 = []
    seed_set_2 = []
    seed_set_3 = []
    for index in range(len(component)):
        roll = np.random.randint(1, 4)
        if roll == 3:
            seed_set_3.append(component[index])
        elif roll == 2:
            seed_set_2.append(component[index])
        elif roll == 1:
            seed_set_1.append(component[index])
    return seed_set_1, seed_set_2, seed_set_3


def main():
    field_names = ['network_name', 'threshold', 'seed_size', 'budget_total'] + [
        str(i) + "_no_block" for i in
        range(4)] + [str(i) + "_cbh"
                     for i in
                     range(4)] + [
                      str(i) + "_degree" for i in range(4)]
    with open('complex_net_proposal/experiment_results/results.csv', 'w', newline='') as csv_fp:
        csv_writer = csv.writer(csv_fp, delimiter=',')
        csv_writer.writerow(field_names)
    # Load in networks
    network_folder = "complex_net_proposal/experiment_networks/"
    # Constants
    seeds = (6893, 20591, 20653)
    net_names = ["fb-pages-politician", "astroph", "wiki"]
    thresholds = (2, 3, 4)
    budgets = [.005] + [.01 + i * .01 for i in range(12)]
    sample_number = 10

    for i in range(len(net_names)):
        np.random.seed(seeds[i])
        net_name = net_names[i]
        G = nx.read_edgelist(network_folder + net_name + '.edges', nodetype=int, create_using=nx.Graph)
        # If there is a node file add it.
        if exists(net_name + ".nodes"):
            with open(network_folder + net_name + ".nodes", 'r') as node_fp:
                node_str = node_fp.readline().strip('\n')
                node_id = int(node_str)
                G.add_node(node_id)
        for node in G.nodes:
            G.nodes[node]['affected_1'] = 0
            G.nodes[node]['affected_2'] = 0
        # Select seed set
        k_core = nx.k_core(G, 20)
        for seed_size in [10, 20]:
            # Initialize accumulators
            # Mult-level dict threshold -> (budget -> (results_avg, results_blocked_avg, results_degree_avg))
            avgs = {
                threshold: {int(budget * G.number_of_nodes()): tuple({state: 0 for state in range(4)} for i in range(3))
                            for budget in budgets} for threshold in
                thresholds}
            for sample in range(sample_number):
                # Choose seed set
                seed_set_1, seed_set_2, seed_set_3 = choose_seed(list(k_core.nodes()), seed_size)
                seed_set = set(seed_set_1 + seed_set_2 + seed_set_3)
                for k in range(len(thresholds)):
                    # Pull out threshold
                    threshold = thresholds[k]
                    for j in range(len(budgets)):
                        # Get the budget
                        budget = int(budgets[j] * G.number_of_nodes())
                        network = copy.deepcopy(G)
                        # Configure model
                        model = utils.config_model(network, threshold, seed_set_1, seed_set_2, seed_set_3)
                        node_infections_1, node_infections_2, results = model.simulation_run()
                        # Analyze node counts
                        infected_1 = results['node_count'][1] + results['node_count'][3]
                        infected_2 = results['node_count'][2] + results['node_count'][3]
                        total_infected = sum(results['node_count'][i] for i in range(1, 4))
                        # Select nodes appropriately
                        # Select nodes appropriately
                        if infected_1 > infected_2:
                            ratio_total = infected_1 / total_infected
                            budget_1 = np.ceil(ratio_total * budget)
                            budget_2 = budget - budget_1
                        elif infected_1 < infected_2:
                            ratio_total = infected_2 / total_infected
                            budget_2 = np.ceil(ratio_total * budget)
                            budget_1 = budget - budget_2
                        else:
                            budget_1 = budget // 2
                            budget_2 = budget - budget_1

                        # Run through the CBH from DMKD for both contagions.
                        choices_1 = cbh.try_all_sets(node_infections_1, budget_1, model, set(seed_set_1 + seed_set_3),
                                                     1)
                        choices_2 = cbh.try_all_sets(node_infections_2, budget_2, model, set(seed_set_2 + seed_set_3),
                                                     2)
                        # Run again
                        network = copy.deepcopy(G)

                        # Configure model
                        model = utils.config_model(network, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1,
                                                   choices_2)
                        node_infections_1_blocked, node_infections_2_blocked, results_blocked = model.simulation_run()
                        # Find high degree nodes
                        network = copy.deepcopy(G)
                        choices_1 = []
                        choices_2 = []
                        nodes_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)
                        index = 0
                        while len(choices_1) < budget_1:
                            if nodes_by_degree[index][0] not in seed_set:
                                choices_1.append(nodes_by_degree[index][0])
                            index += 1
                        index = 0
                        while len(choices_2) < budget_2:
                            if nodes_by_degree[index][0] not in seed_set:
                                choices_2.append(nodes_by_degree[index][0])
                            index += 1

                        # Run forward
                        model = utils.config_model(network, threshold, seed_set_1, seed_set_2, seed_set_3, choices_1,
                                                   choices_2)
                        node_infections_1_blocked_degree, node_infections_2_blocked_degree, results_blocked_degree = model.simulation_run()
                        for state in range(4):
                            avgs[threshold][budget][0][state] += results['node_count'][state]
                            avgs[threshold][budget][1][state] += results_blocked['node_count'][state]
                            avgs[threshold][budget][2][state] += results_blocked_degree['node_count'][state]
            for threshold in thresholds:
                for budget in budgets:
                    budget = int(budget * G.number_of_nodes())
                    for accumulator in range(3):
                        for state in range(4):
                            avgs[threshold][budget][accumulator][state] /= sample_number
                    with open('complex_net_proposal/experiment_results/results.csv', 'a', newline='') as results_fp:
                        csv_writer = csv.writer(results_fp, delimiter=',')
                        # Write problem data
                        result_data = [net_name, str(threshold), str(seed_size),
                                       str(budget)] + list(
                            map(lambda x: str(x), avgs[threshold][budget][0].values())) + list(
                            map(lambda x: str(x), avgs[threshold][budget][1].values())) + list(
                            map(lambda x: str(x), avgs[threshold][budget][2].values()))
                        csv_writer.writerow(result_data)


if __name__ == '__main__':
    main()
