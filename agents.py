'''
FILE (minus some comments) FROM CEDEGAO ET AL 2021
I (J.D. Top) DID NOT MAKE THIS!
Novel comments and new functions should be labelled as such
'''

import numpy as np
import networkx as nx
import itertools as it
from scipy.special import logsumexp
from epistemic_structures import *
import utils as f
import timeit

'''
All CompMods written as python class object
Specific CompMod inherits a more general model category
All models must inherit task_parameters
'''
answerglobal = 0
class task_parameters:
    '''
    Design models tailored for the task
    '''
    def __init__(self, data):
        # task parameters
        self.cards = 'AAAA8888' #Top: all cards in the entire game
        self.possible_hand = ['AA', 'A8', '88'] #Top: you always have exactly two cards, and A8=8A
        self.possible_order = ['Amy,Ben,You', 'Ben,You,Amy', 'You,Amy,Ben'] #Top: Amy,Ben is the same as Ben,Amy
        self.num_cards = len(self.cards) #Top: number of cards in the game (8 in this case)
        self.num_agents = 3 #Top: You, Amy, and Ben
        self.num_cards_each = 2 #Top: AA, A8, or 88
        self.num_phase = 10 #Top: 10 games were played
        self.colnamesIn = ['phase', 'cards', 'order', 'outcomeArray', 'response', 'answer'] #Top: 
        self.colnamesOut = self.colnamesIn + ['AmyResponse', 'BenResponse', 'corAnswer', 'round',
                                              'subj', 'outcome', 'cards_iso'] # add parameter names when simulating
        self.Sub = np.unique(data['subj']) # list of subject numbers
        self.num_subj = len(self.Sub) #Top: number of subjects
        self.data = [f.pd2np(data[data.subj == n][self.colnamesOut]) for n in self.Sub] # turn data into list of subject data

    def initialize(self):
        '''
        Initialize the parameters according to appropriate distributions
        output (1d np array): initial parameters to put into the optimizer
        '''
        count, param0 = 0, np.empty(len(self.parameter_space), dtype=np.float64)
        for k, v in self.continuous_parameter.items():
            if self.initialize_method[k] == 'uniform':
                interval = self.continuous_parameter[k]
                param0[count] = np.random.uniform(low=interval[0], high=interval[1])
            count += 1
        return param0

# stochastic intake models (NoisyDEL can be considered as intake_prob == 1)
class NoisyDEL(task_parameters, modal_model):
    '''
    Model that assumes subjects holds the full model in mind and eliminate nodes as the game progresses
    Parameter:
        noise (float): chance of random guessing IF the model says she shouldn't know
    Note: parameter_names, colnamesIn, colnamesOut are crucial in making sure the right value are retrieved for the right variable
    Use them to guide indexing in the functions
    '''
    def __init__(self, data):
        task_parameters.__init__(self, data)
        modal_model.__init__(self)
        self.name = 'NoisyDEL'
        # parameters and their bounds
        self.discrete_parameter = {}
        self.continuous_parameter = {'noise':[0,1]}
        self.parameter_combination = list(it.product(*[self.discrete_parameter.values()]))
        # all possible combinations of the discrete parameters
        self.parameter_space = list(self.continuous_parameter.values())
        # all ranges of continuous parameters
        self.parameter_names = list(self.discrete_parameter.keys()) + list(
        self.continuous_parameter.keys())  # note the convention that discrete precedes continuous params
        # how to initialize continuous parameters
        self.initialize_method = {'noise': 'uniform'}
    def agent_by_round(self, state, players, announcements, graph):
        '''
        state (string): 'AA88A8' (your cards, Amy's, Ben's)
        players: (list of strings) ['Amy', 'Ben', 'You'] (note: not the game order. must match announcement order)
        announcements: (1D np array) [] if subject goes first, [False, True] if subject goes third etc
        graph: (nx object) epistemic model
        return: response of the current turn (bool), updated model (graph)
        '''
        G = graph
        for i in range(len(announcements)):
            player, announcement = players[i], announcements[i]
            g = self.update_model(G, announcement, self.player_number[player])
            state_in = len([i for i, d in g.nodes(data=True) if d['state'] == state]) != 0
            if state_in:
                G = g
        response = self.compute_correct_response(G, state, self.player_number['You'])
        return response, G
    def agent_by_game(self, param, state, order, outcomeArray):
        '''
        param (list): [noise]
        state (str), order (list of str): cards and order in the game
        outcomeArray: (list of bool) [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: list of response (bool) for the entire game, the selected hand (str), correct response or not (bool)
        '''
        noise = param[self.parameter_names.index('noise')]
        G, player_idx = self.generate_full_model(), order.index('You')
        responses, answer = [], ''
        players_after, announcement_after_player = [], []
        outcome = True
        for rnd in outcomeArray:
            announcement_before_player, announcement_after_player = np.append(announcement_after_player, rnd[:player_idx]), rnd[player_idx:]
            players_before, players_after = np.append(players_after, order[:player_idx]), order[player_idx:]
            response, G = self.agent_by_round(state, players_before, announcement_before_player, G)
            possible_answers = self.compute_possible_states(G, state, self.player_number['You'])

            if np.random.random_sample() < noise: # guess
                if np.random.random_sample() < 0.5: # guess don't know
                    responses.append(False)
                    outcome = outcome and not rnd[order.index('You')]
                else: # guess know and randomly choose one from candidates
                    responses.append(True)
                    answer = np.random.choice(self.possible_hand) # randomly choose one of three
                    outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                    break
            else: # not guess
                if response:
                    assert len(possible_answers) == 1
                    responses.append(response)
                    answer = possible_answers[0][:2]
                    outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                    break # if know, the game ends
                else:
                    responses.append(response)
                    outcome = outcome and not rnd[order.index('You')]
        return responses, answer, outcome
    def agent(self, subNum, param):
        '''
        param: same as above
        subNum (int): which subject data we are simulating
        return: numpy data matrix
        '''
        colnamesOut = self.colnamesOut + self.parameter_names
        data = np.ones(len(colnamesOut)) # initialize output data. note for small dataset (<50000 rows) numpy is more efficient
        # get coloumn index for different task variables
        phase_idx = self.colnamesIn.index('phase')
        cards_idx = self.colnamesIn.index('cards')
        order_idx = self.colnamesIn.index('order')
        outcomeArray_idx = self.colnamesIn.index('outcomeArray')
        # get coloumn index for output variables
        phase_out = colnamesOut.index('phase')
        cards_out = colnamesOut.index('cards')
        order_out = colnamesOut.index('order')
        outcomeArray_out = colnamesOut.index('outcomeArray')
        AmyResponse_out = colnamesOut.index('AmyResponse')
        BenResponse_out = colnamesOut.index('BenResponse')
        corAnswer_out = colnamesOut.index('corAnswer')
        round_out = colnamesOut.index('round')
        response_out = colnamesOut.index('response')
        answer_out = colnamesOut.index('answer')
        subj_out = colnamesOut.index('subj')
        outcome_out = colnamesOut.index('outcome')
        cards_iso_out = colnamesOut.index('cards_iso')
        noise_out = colnamesOut.index('noise')
        # remove redundant rows due to rnd info
        real_data = f.unique([tuple(row) for row in self.data[subNum - 1]])
        # get task parameters
        phases, all_cards, all_order, all_outcomeArray = real_data[:, phase_idx], real_data[:, cards_idx], real_data[:, order_idx], real_data[:, outcomeArray_idx]
        assert len(all_cards) == len(all_order) == len(phases)
        for i in range(len(phases)): # for each game
            phase, order, cards, outcomeArray = phases[i], all_order[i], all_cards[i], all_outcomeArray[i]
            outcomeArray_evaled, cards_iso = eval(outcomeArray), self.iso_map[cards]
            responses, answer, outcome = self.agent_by_game(param, cards, order.split(","), outcomeArray_evaled)
            for rnd in range(len(responses)):
                outcomeSubarray, response, log = outcomeArray_evaled[rnd], responses[rnd], [None]*3
                noise = param[self.parameter_names.index('noise')]
                for o in range(len(outcomeSubarray)):
                    log[o] = outcomeSubarray[o]
                AmyResponse, BenResponse, corAnswer = log[order.split(",").index('Amy')], log[order.split(",").index('Ben')], log[order.split(",").index('You')]
                # prepare output row to append to data
                outputs = np.empty(len(colnamesOut), dtype=object)
                outputs[phase_out], outputs[cards_out], outputs[order_out], outputs[outcomeArray_out] = phase, cards, order, outcomeArray
                outputs[AmyResponse_out], outputs[BenResponse_out], outputs[corAnswer_out], outputs[
                    round_out] = AmyResponse, BenResponse, corAnswer, rnd+1
                outputs[response_out], outputs[answer_out], outputs[subj_out], outputs[outcome_out] = response, answer, subNum, outcome
                outputs[cards_iso_out], outputs[noise_out] = cards_iso, noise
                data = np.vstack((data, outputs))
        return data[1:]
    
    def LLH_by_round(self, state, players, announcements, graph, update=1):
        '''
        state (string): 'AA88A8' (your cards, Amy's, Ben's)
        players: (list of strings) ['Amy', 'Ben', 'You'] (note: not the game order. must match announcement order)
        announcements: (1D np Array) [] if subject goes first, [False, True] if subject goes third etc
        graph: (nx object) epistemic model
        return: list of all possible model (graph) generated, list of respective log likehood (float)
        '''
        num_announcements = len(announcements)
        models, LLHs = [], []
        if num_announcements == 0:
            models.append(graph)
            LLHs.append(np.log(1))
        else:
            for branch in it.product(*[[0,1]]*num_announcements):
                G, LLH = graph, 0
                for i in range(len(branch)):
                    player, announcement, need_update = players[i], announcements[i], branch[i]
                    if need_update:
                        LLH += np.log(update)
                    else:
                        LLH += np.log(1 - update)
                    if need_update: # if 1 means update accordingly
                        g = self.update_model(G, announcement, self.player_number[player])
                        state_in = len([i for i, d in g.nodes(data=True) if d['state'] == state]) != 0
                        if state_in:
                            G = g
                models.append(G)
                LLHs.append(LLH)
        return models, LLHs
    def LLH_bayes_net(self, param, state, order, outcomeArray):
        '''
        param (list): [noise]
        state (str), order (list): cards and order in the game
        outcomeArray (list of bool): [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: a bayes_net (nx object) encoding the log likelihood of each possible situation generated by this model
        '''
        noise = param[self.parameter_names.index('noise')]
        G, player_idx,bayes_net, max_round = self.generate_full_model(), order.index('You'), nx.DiGraph(), len(outcomeArray)
        bayes_net.add_node((0,), model=G) # initialize start node.
        players_after, announcement_after_player, respond_false_prob = [], [], 1
        for idx in range(max_round): # for every round
            rnd, child_id = outcomeArray[idx], 0 # prepare for the input to the LLH_round
            announcement_before_player, announcement_after_player = \
                np.append(announcement_after_player, rnd[:player_idx]), rnd[player_idx:]
            players_before, players_after = np.append(players_after, order[:player_idx]), order[player_idx:]
            current_layer = [node_pair for node_pair in bayes_net.nodes(data=True) if node_pair[0][0] == idx and node_pair[1]['model']]
            for node_pair in current_layer: # for all possible output model from the previous rnd except the special hand node
                node, prev_model = node_pair[0], node_pair[1]['model']
                if node[0] != 0:
                    parent_response = self.compute_correct_response(prev_model, state, self.player_number['You'])
                    respond_false_prob = (1-parent_response)*(1-noise) + noise / 2
                models, LLHs = self.LLH_by_round(state, players_before, announcement_before_player, prev_model)
                assert len(models) == len(LLHs)
                for hand in self.possible_hand:
                    bayes_net.add_node((idx + 1, hand), model=None) # add the answer node
                for i in range(len(models)): # for all updated models
                    child_id += 1
                    model = models[i]
                    model_answer = self.compute_correct_response(model, state, self.player_number['You'])
                    respond_true_prob = model_answer*(1-noise) + (noise/2) # chance of answer true
                    bayes_net.add_node((idx+1, child_id), model=model)
                    bayes_net.add_edge(node, (idx+1, child_id), weight=np.log(respond_false_prob) + LLHs[i])
                    for hand in self.possible_hand: # all hand nodes
                        bayes_net.add_edge((idx + 1, child_id), (idx + 1, hand),
                                           weight=np.log(respond_true_prob/len(self.possible_hand)))
        return bayes_net
    def LLH_by_game(self, param, state, order, outcomeArray):
        '''
        return: a dictionary of dictionary and float
        If the key is int, it represents the rnd (1,2,or 3) where agent answers I know
        Then its value is a dictionary
            key (str): possible hand ('AA', 'A8', '88')
            value (float): their log likelihood
        If the key is '', it represents the agent answered I don't know till the end
            value (float): log likelihood
        '''
        bayes_net = self.LLH_bayes_net(param, state, order, outcomeArray)
        response_llh = {}
        for rnd in range(len(outcomeArray)):
            answer_chance = {}
            for hand in self.possible_hand:
                hand_llh = []
                for path in nx.all_simple_paths(bayes_net, source=(0,), target=(rnd+1, hand)):
                    path_llh = 0 # initialize branch conditional probability
                    for parent in range(len(path)-1):
                        path_llh += bayes_net.get_edge_data(path[parent], path[parent+1])['weight']
                    hand_llh.append(path_llh)
                answer_chance[hand] = logsumexp(hand_llh) # marginalize
            response_llh[rnd+1] = answer_chance
        LLH = []
        for dic in response_llh.values():
            for llh in dic.values():
                LLH.append(llh)
        response_llh[''] = np.log(1-np.exp(logsumexp(LLH)))
        return response_llh
    def nLLH(self, param, Data):
        '''
        param: (list) Model parameters
        Data: (numpy matrix) The actual subject data
        return: (float) The negative log likelihood
        '''
        # get coloumn index for different task variables
        phase_out = self.colnamesOut.index('phase')
        round_out = self.colnamesOut.index('round')
        answer_out = self.colnamesOut.index('answer')
        cards_out = self.colnamesOut.index('cards')
        order_out = self.colnamesOut.index('order')
        # prep work
        llh, games =0, np.unique(Data[:, phase_out])  # list of block numbers
        for g in games:  # loop through games
            current_game = Data[Data[:, phase_out] == g]
            cardss, orders = np.unique(current_game[:, cards_out]), np.unique(current_game[:, order_out])
            rounds, answers = current_game[:, round_out], np.unique(current_game[:, answer_out])
            assert len(cardss) == 1 and len(orders) == 1
            num_round, answer, cards, order = len(rounds), answers[0], cardss[0], orders[0].split(",")
            if np.isnan(sum(param)): # a wierd bug that scipy minimize sometimes sample [nan nan] as parameter
                continue
            LLH_look_up = self.LLH_by_game(param, cards, order, self.compute_game_response(list(cards), order))
            if answer in self.possible_hand:
                llh += LLH_look_up[num_round][answer]
            else:
                llh += LLH_look_up['']
        if np.isnan(llh):
            llh = -np.inf # fit sometimes (not always) turn inf in a funtion into nan somehow. I think it's an internal bug
        return -llh

class SIWEB(task_parameters, bounded_modal_model):
    '''
    Model that assumes subjects hold a bounded model in mind and stochastically intake announcements to update it
    Parameter:
        level (int): bounds the initial model
        intake_prob (float): the chance of intaking an annoucement to update the model every turn
        noise (float): the chance of random guessing IF the model says she shouldn't know
    Note: parameter_names, colnamesIn, colnamesOut are crucial in making sure the right value are retrieved for the right variable
    Use them to guide indexing in the functions
    '''
    def __init__(self, data):
        task_parameters.__init__(self, data)
        bounded_modal_model.__init__(self)
        self.name = 'SIWEB'
        # parameters and their bounds
        self.discrete_parameter = {'level':range(5)}
        self.continuous_parameter = {'intake_prob':[0,1], 'noise':[0,1]}
        self.parameter_combination = list(it.product(*self.discrete_parameter.values()))
        # all possible combinations of the discrete parameters
        self.parameter_space = list(self.continuous_parameter.values())
        # all ranges of continuous parameters
        self.parameter_names = list(self.discrete_parameter.keys()) + list(
        self.continuous_parameter.keys())  # note the convention that discrete precedes continuous params
        # how to initialize continuous parameters
        self.initialize_method = {'intake_prob':'uniform', 'noise': 'uniform'}

    def agent_by_round(self, state, players, announcements, graph, intake_prob):
        '''
        state (string): 'AA88A8' (your cards, Amy's, Ben's)
        players: (list of strings) ['Amy', 'Ben', 'You'] (note: not the game order. must match announcement order)
        announcements: (1D np array) [] if subject goes first, [False, True] if subject goes third etc
        graph: (nx object) epistemic model
        intake_prob (float): one of the model parameters
        return: response of the current turn (bool), updated model (graph)
        '''
        G = graph
        for i in range(len(announcements)):
            response = self.compute_my_response(G, state[2:4], state[4:6])
            if response:
                return response, G
            player, announcement = players[i], announcements[i]
            if np.random.random_sample() < intake_prob:
                G = self.update_model(G, announcement, self.player_number[player])
        response = self.compute_my_response(G, state[2:4], state[4:6])
        return response, G
    def agent_by_game(self, param, state, Amy_cards, Ben_cards, order, outcomeArray):
        '''
        param (list): [level, intake_prob, noise]
        state (str), order (list of str): cards and order in the game
        outcomeArray: (list of bool) [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: list of response (bool) for the entire game, the selected hand (str), correct response or not (bool)
        '''
        assert Amy_cards == state[2:4] and Ben_cards == state[4:6]

        level = param[self.parameter_names.index('level')]
        noise, intake_prob = param[self.parameter_names.index('noise')], param[self.parameter_names.index('intake_prob')]
        G, player_idx = self.generate_partial_model(Amy_cards, Ben_cards, level), order.index('You')
        responses, answer = [], ''
        players_after, announcement_after_player = [], []
        outcome = True
        for rnd in outcomeArray:
            announcement_before_player, announcement_after_player = np.append(announcement_after_player, rnd[:player_idx]), rnd[player_idx:]
            players_before, players_after = np.append(players_after, order[:player_idx]), order[player_idx:]
            response, G = self.agent_by_round(state, players_before, announcement_before_player, G, intake_prob)
            if response and response != -1:
                responses.append(response)
                possibilities = []
                for i in list(G.nodes):
                    if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][4:6] == Ben_cards:
                        possibilities.append(i)
                assert len(possibilities) == 1
                answer = G.nodes[possibilities[0]]['state'][:2]
                outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                break # if know, the game ends
            else:
                if response == -1: # inconsistent and have to guess
                    if np.random.random_sample() < 0.5: # guess don't know
                        responses.append(False)
                        outcome = outcome and not rnd[order.index('You')]
                    else: # guess know and randomly choose one from candidates
                        responses.append(True)
                        answer = np.random.choice(self.possible_hand) # randomly choose one of three
                        outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                        break
                else: # don't know
                    if np.random.random_sample() < noise:
                        responses.append(True)
                        answer = np.random.choice(self.possible_hand) # random choose one of three
                        outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                        break
                    else:
                        responses.append(response) # honestly say don't know
                        outcome = outcome and not rnd[order.index('You')]
        return responses, answer, outcome
    def agent(self, subNum, param):
        '''
        param: same as above
        subNum (int): which subject data we are simulating
        return: numpy data matrix
        '''
        colnamesOut = self.colnamesOut + self.parameter_names
        data = np.ones(len(colnamesOut)) # initialize output data. Note for small dataset (<50000 rows) numpy is more efficient
        # get coloumn index for different task variables
        phase_idx = self.colnamesIn.index('phase')
        cards_idx = self.colnamesIn.index('cards')
        order_idx = self.colnamesIn.index('order')
        outcomeArray_idx = self.colnamesIn.index('outcomeArray')
        # get coloumn index for output variables
        phase_out = colnamesOut.index('phase')
        cards_out = colnamesOut.index('cards')
        order_out = colnamesOut.index('order')
        outcomeArray_out = colnamesOut.index('outcomeArray')
        AmyResponse_out = colnamesOut.index('AmyResponse')
        BenResponse_out = colnamesOut.index('BenResponse')
        corAnswer_out = colnamesOut.index('corAnswer')
        round_out = colnamesOut.index('round')
        response_out = colnamesOut.index('response')
        answer_out = colnamesOut.index('answer')
        subj_out = colnamesOut.index('subj')
        outcome_out = colnamesOut.index('outcome')
        cards_iso_out = colnamesOut.index('cards_iso')
        level_out = colnamesOut.index('level')
        intake_prob_out = colnamesOut.index('intake_prob')
        noise_out = colnamesOut.index('noise')
        # remove redundant rows due to rnd info
        real_data = f.unique([tuple(row) for row in self.data[subNum - 1]])
        # get task parameters
        phases, all_cards, all_order, all_outcomeArray = real_data[:, phase_idx], real_data[:, cards_idx], real_data[:, order_idx], real_data[:, outcomeArray_idx]
        assert len(all_cards) == len(all_order) == len(phases)

        for i in range(len(phases)): #for each game
            phase, order, cards, outcomeArray = phases[i], all_order[i], all_cards[i], all_outcomeArray[i]
            outcomeArray_evaled, cards_iso, Amy_cards, Ben_cards = eval(outcomeArray), self.iso_map[cards], cards[2:2+self.num_cards_each], cards[4:4+self.num_cards_each]
            responses, answer, outcome = self.agent_by_game(param, cards, Amy_cards, Ben_cards, order.split(","), outcomeArray_evaled)
            for rnd in range(len(responses)):
                outcomeSubarray, response, log = outcomeArray_evaled[rnd], responses[rnd], [None]*3
                level, intake_prob, noise = param
                for o in range(len(outcomeSubarray)):
                    log[o] = outcomeSubarray[o]
                AmyResponse, BenResponse, corAnswer = log[order.split(",").index('Amy')], log[order.split(",").index('Ben')], log[order.split(",").index('You')]
                # prepare output row to append to data
                outputs = np.empty(len(colnamesOut), dtype=object)
                outputs[phase_out], outputs[cards_out], outputs[order_out], outputs[outcomeArray_out] = phase, cards, order, outcomeArray
                outputs[AmyResponse_out], outputs[BenResponse_out], outputs[corAnswer_out], outputs[
                    round_out] = AmyResponse, BenResponse, corAnswer, rnd+1
                outputs[response_out], outputs[answer_out], outputs[subj_out], outputs[outcome_out] = response, answer, subNum, outcome
                outputs[cards_iso_out], outputs[level_out], outputs[noise_out] = cards_iso, level, noise
                outputs[intake_prob_out] = intake_prob
                data = np.vstack((data, outputs))
        return data[1:]

    def LLH_by_round(self, intake_prob, state, players, announcements, graph):
        '''
        intake_prob: (float) between 0 and 1 capture chance of updating the model
        state: 'AA88A8' (your cards, Amy's, Ben's)
        players: (list of strings) ['Amy', 'Ben', 'You'] (note: not the game order. must match announcement order)
        announcements: (1D np Array) [] if subject goes first, [False, True] if subject goes third etc
        graph: (nx object) epistemic model
        return: list of all possible model (graph) generated, list of respective log likehood (float)
        '''
        num_announcements = len(announcements)
        models, LLHs = [], []
        if num_announcements == 0:
            models.append(graph)
            LLHs.append(np.log(1))
        else:
            for branch in it.product(*[[0,1]]*num_announcements):
                G, LLH = graph, 0
                for i in range(len(branch)):
                    player, announcement, need_update = players[i], announcements[i], branch[i]
                    if need_update:
                        LLH += np.log(intake_prob)
                    else:
                        LLH += np.log(1 - intake_prob)
                    response = self.compute_my_response(G, state[2:4], state[4:6])        
                    if response:
                        continue
                    if need_update: # if 1 means update accordingly
                        player, announcement = players[i], announcements[i]
                        G = self.update_model(G, announcement, self.player_number[player])
                models.append(G)
                LLHs.append(LLH)
        return models, LLHs
    def LLH_bayes_net(self, param, state, order, outcomeArray):
        '''
        param (list): [noise]
        state (str), order (list): cards and order in the game
        outcomeArray (list of bool): [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: a bayes_net (nx object) encoding the log likelihood of each possible situation generated by this model
        '''
        level, intake_prob, noise = param[self.parameter_names.index('level')], min(param[self.parameter_names.index('intake_prob')], 1), min(param[self.parameter_names.index('noise')],1)
        G, player_idx,bayes_net, max_round = self.generate_partial_model(state[2:4], state[4:6], level), order.index('You'), nx.DiGraph(), len(outcomeArray)
        bayes_net.add_node((0,), model=G) # initialize start node
        players_after, announcement_after_player, respond_false_prob = [], [], 1
        for idx in range(max_round): # for every round
            rnd, child_id = outcomeArray[idx], 0 # prepare for the input to the LLH_round
            announcement_before_player, announcement_after_player = \
                np.append(announcement_after_player, rnd[:player_idx]), rnd[player_idx:]
            players_before, players_after = np.append(players_after, order[:player_idx]), order[player_idx:]
            current_layer = [node_pair for node_pair in bayes_net.nodes(data=True) if node_pair[0][0] == idx and node_pair[1]['model']]
            for node_pair in current_layer: # for all possible output model from the previous rnd except the special hand node
                node, prev_model = node_pair[0], node_pair[1]['model']
                prev_response = self.compute_my_response(prev_model, state[2:4], state[4:6])
                if prev_response != 1 or node[0] == 0:
                    # if the parent is a model still with uncertainty unless it's the start node
                    # generate all possible updated the models from it
                    models, LLHs = self.LLH_by_round(intake_prob, state, players_before, announcement_before_player, prev_model)
                    assert len(models) == len(LLHs)
                    for hand in self.possible_hand:
                        bayes_net.add_node((idx + 1, hand), model=None) # add the answer node
                    for i in range(len(models)): # for all updated models
                        child_id += 1
                        model = models[i]
                        response = self.compute_my_response(model, state[2:4], state[4:6])
                        if response != -1:
                            respond_true_prob = response*1 + (1-response)*noise # chance of answer true
                        else:
                            respond_true_prob = 0.5
                        bayes_net.add_node((idx+1, child_id), model=model)
                        if prev_response != -1:
                            bayes_net.add_edge(node, (idx+1, child_id), weight=np.log(respond_false_prob) + LLHs[i])
                        else:
                            bayes_net.add_edge(node, (idx+1, child_id), weight=np.log(0.5) + LLHs[i])
                        for hand in self.possible_hand: # all hand nodes
                            bayes_net.add_edge((idx + 1, child_id), (idx + 1, hand),
                                                   weight=np.log(respond_true_prob/len(self.possible_hand)))
                    respond_false_prob = 1 - noise
        return bayes_net
    def LLH_by_game(self, param, state, order, outcomeArray):
        '''
        return: a dictionary of dictionary or float
        If the key is int, it represents the rnd (1,2,or 3) where agent answers I know
        Then its value is a dictionary
            key (str): possible hand ('AA', 'A8', '88')
            value (float): their log likelihood
        If the key is '', it represents the agent answered I don't know till the end
            value (float): log likelihood
        '''
        bayes_net = self.LLH_bayes_net(param, state, order, outcomeArray)
        response_llh = {}
        for rnd in range(len(outcomeArray)):
            answer_chance = {}
            for hand in self.possible_hand:
                hand_llh = []
                for path in nx.all_simple_paths(bayes_net, source=(0,), target=(rnd+1, hand)):
                    path_llh = 0 # initialize branch conditional probability
                    for parent in range(len(path)-1):
                        path_llh += bayes_net.get_edge_data(path[parent], path[parent+1])['weight']
                    hand_llh.append(path_llh)
                answer_chance[hand] = logsumexp(hand_llh) # marginalize
            response_llh[rnd+1] = answer_chance
        LLH = []
        for dic in response_llh.values():
            for llh in dic.values():
                LLH.append(llh)
        response_llh[''] = np.log(1-min(1,np.exp(logsumexp(LLH)))) # sometimes get 1.000000002 overflow
        return response_llh
    def nLLH(self, continuous_param, Data, discrete_param):
        '''
        param: (list) Model parameters
        Data: (numpy matrix) The actual subject data
        return: (float) The negative log likelihood
        '''
        # get coloumn index for different task variables
        phase_out = self.colnamesOut.index('phase')
        round_out = self.colnamesOut.index('round')
        answer_out = self.colnamesOut.index('answer')
        cards_out = self.colnamesOut.index('cards')
        order_out = self.colnamesOut.index('order')
        outcomeArray_out = self.colnamesOut.index('outcomeArray')
        # prep work
        if np.isnan(sum(continuous_param)) or max(continuous_param) > 1: # a wierd bug that scipy minimize sometimes sample [nan nan] as parameter or have 1.00001 which exceeds bound
            return np.inf
        llh, games =0, np.unique(Data[:, phase_out])  # list of block numbers
        for g in games:  # loop through games
            current_game = Data[Data[:, phase_out] == g]
            cardss, orders = np.unique(current_game[:, cards_out]), np.unique(current_game[:, order_out])
            rounds, answers = current_game[:, round_out], np.unique(current_game[:, answer_out])
            outcomeArray = eval(current_game[:, outcomeArray_out][0])
            assert len(cardss) == 1 and len(orders) == 1
            num_round, answer, cards, order = len(rounds), answers[0], cardss[0], orders[0].split(",")
            # continue
            Amy_cards, Ben_cards = cards[2:4], cards[4:6]
            LLH_look_up = self.LLH_by_game(list(discrete_param)+list(continuous_param), cards, order, outcomeArray)
            if answer in self.possible_hand:
                llh += LLH_look_up[num_round][answer]
            else:
                llh += LLH_look_up['']
        return -llh

# stochastic update models
nSample = 200 # number of samples used to estimate likelihood

#Top: SUWEB
'''
Top: SUWEB has parameters
epistemic_level in {0,1,2,3,4} - edges from initial possible worlds to peripheral states
update_prob in [0,1] - probability a node is actually removed when it should be
noise - If subject doesn't know between states, guess 'know' with probability noise
'''
class SUWEB(task_parameters, imperfect_update_model):
    '''
    Model that assumes subjects hold a bounded model in mind and stochastically eliminate nodes when updating it
    Parameter:
        level (int): bounds the initial model
        update_prob (float): the chance of successfully eliminating a node when the update requires an elimination 
        noise (float): chance of random guessing IF the model says she shouldn't know
    Note: parameter_names, colnamesIn, colnamesOut are crucial in making sure the right value are retrieved for the right variable
    Use them to guide indexing in the functions
    '''
    def __init__(self, data, nSample = nSample): #Top: initialization
        task_parameters.__init__(self, data) #Top: initialize task parameters (data frame)
        imperfect_update_model.__init__(self) #Top: initialize a SUWEB
        self.name = 'SUWEB' #Top: this is SUWEB
        self.nSample = nSample
        # parameters and their bounds
        self.discrete_parameter = {'level':range(5)} #Top: epistemic level l = 0,1,2,3,4
        self.continuous_parameter = {'update_prob':[0,1], 'noise':[0,1]}
        self.parameter_combination = list(it.product(*self.discrete_parameter.values())) #Top: [(0,), (1,), (2,), (3,), (4,)]
        # all possible combinations of the discrete parameters
        self.parameter_space = list(self.continuous_parameter.values()) #Top: [[0, 1], [0, 1]]
        # all ranges of continuous parameters
        self.parameter_names = list(self.discrete_parameter.keys()) + list(self.continuous_parameter.keys())  # note the convention that discrete precedes continuous params
        # how to initialize continuous parameters
        self.initialize_method = {'update_prob':'uniform', 'noise': 'uniform'} #Top: Uniform parameter sweep over continuous parameters
    
    def agent_by_round(self, state, players, announcements, graph, update_prob):
        '''
        state (string): 'AA88A8' (your cards, Amy's, Ben's)
        players: (list of strings) ['Amy', 'Ben', 'You'] (note: not the game order. must match announcement order)
        announcements: (1D np array) [] if subject goes first, [False, True] if subject goes third etc #Top: previous announcements by agents
        graph: (nx object) epistemic model
        intake_prob (float): one of the model parameters #Top: update_prob not intake_prob, probability of deleting node
        return: response of the current turn (bool), updated model (graph)
        '''
        G = graph #Top: copy input graph
        for i in range(len(announcements)):
            response = self.compute_my_response(G, state[2:4], state[4:6]) #Top: True for 'know', False for 'don't know'
            if response: #Top: if you know while listening to announcements
                return response, G #Top: return response and (un)updated model
            player, announcement = players[i], announcements[i]
            G = self.update_model(G, update_prob, announcement, self.player_number[player])
        response = self.compute_my_response(G, state[2:4], state[4:6]) #Top: if you know after listening to announcements
        return response, G

    '''
    Modification of agent_by_game by J.D. Top
    Returns the answer given a single model and visible cards
    
    Uses compute_my_response from epistemic_structures.py
    '''
    @staticmethod
    def agent_by_game_single_game(level, update_prob, noise, state, Amy_cards, Ben_cards, G):
        #start1 = timeit.default_timer()
        verbose = False
        '''
        param (list): [level, update_prob, noise] #Top: level is epistemic_level l, maximum edges from initial interior nodes, update_prob is probability of removing node, noise is probability of guessing if you don't know
        state (str), order (list of str): cards and order in the game #Top: state is e.g. 'AA88A8', order is e.g. ['Amy','You','Ben']
        outcomeArray: (list of bool) [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: list of response (bool) for the entire game, the selected hand (str), correct response or not (bool)
        '''
        if verbose:
            print("agent_by_game_single_game")
            print("state: " + state)
        assert Amy_cards == state[2:4] and Ben_cards == state[4:6], "State: " + state + ", Amy: " + Amy_cards + ", Ben: " + Ben_cards
        #
        # Top: throws error if false
        possib_hand = ['AA', 'A8', '88']
        reponse = -1
        possibilities = []  # possible states for the participant
        if verbose:
            print("Loop over nodes:")
        for i in list(G.nodes):  # Top: Loop over nodes
            if verbose:
                print("Node: "+ G.nodes[i]['state'])
            if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][
                                                         4:6] == Ben_cards:  # Top: if input corresponds to this
                # possible world
                possibilities.append(i)  # Top: Add it to the list of possibilities
        n = len(possibilities)  # Top: Calculate the number of possibilities
        imposs = False
        if n == 0:  # Top: There are no possibilities. Someone lied or made a mistake.
            response = -1
            imposs = True
        elif n == 1:  # Top: There is exactly one possibility, the player knows his/her cards
            response = True
        else:  # Top: there are two or more possibilities. The player's cards are not known by the player.
            response = False

        responses = []
        answer = ''
        if response and response != -1:  # Top: True indicates 'I know', -1 indicates empty possible world model
            responses.append(response)
            possibilities = []  # Top: all nodes consistent with observation of Amy and Ben's cards
            for i in list(G.nodes):
                if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][4:6] == Ben_cards:
                    possibilities.append(i)
            assert len(
                possibilities) == 1  # Top: throws error if false (there must be exactly one possibility given your observations)
            answer = G.nodes[possibilities[0]]['state'][:2]  # Top: get AA/88/A8
        else:
            if response == -1:  # inconsistent and have to guess Top: model empty
                if np.random.random_sample() < 0.5:  # guess don't know
                    responses.append(False)
                else:  # guess know and randomly choose one from s
                    responses.append(True)
                    answer = np.random.choice(possib_hand)  # randomly choose one of three
            else:
                if np.random.random_sample() < noise:  # Top: noise is the probability of saying 'know' when you don't
                    responses.append(True)
                    answer = np.random.choice(possib_hand)  # random choose one of three Top: AA/A8/88
                else:
                    responses.append(response)  # honestly say don't know
        #stop1 = timeit.default_timer()
        #global answerglobal
        #answerglobal += stop1 - start1
        return responses[0], imposs, answer

    @staticmethod  # Top: NEW
    def printglobals():
        global answerglobal
        print("Total answer time c: " + str(answerglobal))

    def agent_by_game(self, param, state, Amy_cards, Ben_cards, order, outcomeArray):
        '''
        param (list): [level, update_prob, noise] #Top: level is epistemic_level l, maximum edges from initial interior nodes, update_prob is probability of removing node, noise is probability of guessing if you don't know
        state (str), order (list of str): cards and order in the game #Top: state is e.g. 'AA88A8', order is e.g. ['Amy','You','Ben']
        outcomeArray: (list of bool) [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: list of response (bool) for the entire game, the selected hand (str), correct response or not (bool)
        '''
        assert Amy_cards == state[2:4] and Ben_cards == state[4:6] #Top: throws error if false

	#Top: copy parameters to their own variables
        level = param[self.parameter_names.index('level')]
        noise, update_prob = param[self.parameter_names.index('noise')], param[self.parameter_names.index('update_prob')]
        G, player_idx = self.generate_partial_model(Amy_cards, Ben_cards, level), order.index('You') #Top: possible worlds model with epistemic bound and player number in player order
        responses, answer = [], ''
        players_after, announcement_after_player = [], []
        outcome = True
        for rnd in outcomeArray: #Top: for each round
            announcement_before_player, announcement_after_player = np.append(announcement_after_player, rnd[:player_idx]), rnd[player_idx:] #Top: True for 'I know', false for 'I don't know'
            players_before, players_after = np.append(players_after, order[:player_idx]), order[player_idx:]
            response, G = self.agent_by_round(state, players_before, announcement_before_player, G, update_prob) #Top response and updated model for the current round
            if response and response != -1: #Top: True indicates 'I know', -1 indicates empty possible world model
                responses.append(response)
                possibilities = [] #Top: all nodes consistent with observation of Amy and Ben's cards
                for i in list(G.nodes):
                    if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][4:6] == Ben_cards:
                        possibilities.append(i)
                assert len(possibilities) == 1 #Top: throws error if false (there must be exactly one possibility given your observations)
                answer = G.nodes[possibilities[0]]['state'][:2] #Top: get AA/88/A8
                outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                break  # if know, the game ends
            else:
                if response == -1: # inconsistent and have to guess Top: model empty
                    if np.random.random_sample() < 0.5: # guess don't know
                        responses.append(False)
                        outcome = outcome and not rnd[order.index('You')]
                    else: # guess know and randomly choose one from s
                        responses.append(True)
                        answer = np.random.choice(self.possible_hand) # randomly choose one of three
                        outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                        break
                else:
                    if np.random.random_sample() < noise: #Top: noise is the probability of saying 'know' when you don't
                        responses.append(True)
                        answer = np.random.choice(self.possible_hand) # random choose one of three Top: AA/A8/88
                        outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                        break
                    else:
                        responses.append(response) # honestly say don't know
                        outcome = outcome and not rnd[order.index('You')]
        return responses, answer, outcome #Top: outcome is whether response is correct

    def agent(self, subNum, param):
        '''
        param: same as above
        subNum (int): which subject data we are simulating
        return: numpy data matrix
        '''
        colnamesOut = self.colnamesOut + self.parameter_names
        data = np.ones(len(colnamesOut)) # initialize output data. note for small dataset (<50000 rows) numpy is more efficient
        # get coloumn index for different task variables
        phase_idx = self.colnamesIn.index('phase')
        cards_idx = self.colnamesIn.index('cards')
        order_idx = self.colnamesIn.index('order')
        outcomeArray_idx = self.colnamesIn.index('outcomeArray')
        # get coloumn index for output variables
        phase_out = colnamesOut.index('phase')
        cards_out = colnamesOut.index('cards')
        order_out = colnamesOut.index('order')
        outcomeArray_out = colnamesOut.index('outcomeArray')
        AmyResponse_out = colnamesOut.index('AmyResponse')
        BenResponse_out = colnamesOut.index('BenResponse')
        corAnswer_out = colnamesOut.index('corAnswer')
        round_out = colnamesOut.index('round')
        response_out = colnamesOut.index('response')
        answer_out = colnamesOut.index('answer')
        subj_out = colnamesOut.index('subj')
        outcome_out = colnamesOut.index('outcome')
        cards_iso_out = colnamesOut.index('cards_iso')
        level_out = colnamesOut.index('level')
        update_prob_out = colnamesOut.index('update_prob')
        noise_out = colnamesOut.index('noise')
        # remove redundant rows due to rnd info
        real_data = f.unique([tuple(row) for row in self.data[subNum - 1]])
        # get task parameters
        phases, all_cards, all_order, all_outcomeArray = real_data[:, phase_idx], real_data[:, cards_idx], real_data[:, order_idx], real_data[:, outcomeArray_idx]
        assert len(all_cards) == len(all_order) == len(phases)

        for i in range(len(phases)): # for each game
            phase, order, cards, outcomeArray = phases[i], all_order[i], all_cards[i], all_outcomeArray[i]
            outcomeArray_evaled, cards_iso, Amy_cards, Ben_cards = eval(outcomeArray), self.iso_map[cards], cards[2:2+self.num_cards_each], cards[4:4+self.num_cards_each]
            responses, answer, outcome = self.agent_by_game(param, cards, Amy_cards, Ben_cards, order.split(","), outcomeArray_evaled)
            for rnd in range(len(responses)):
                outcomeSubarray, response, log = outcomeArray_evaled[rnd], responses[rnd], [None]*3
                level, update_prob, noise = param
                for o in range(len(outcomeSubarray)):
                    log[o] = outcomeSubarray[o]
                AmyResponse, BenResponse, corAnswer = log[order.split(",").index('Amy')], log[order.split(",").index('Ben')], log[order.split(",").index('You')]
                # prepare output row to append to data
                outputs = np.empty(len(colnamesOut), dtype=object)
                outputs[phase_out], outputs[cards_out], outputs[order_out], outputs[outcomeArray_out] = phase, cards, order, outcomeArray
                outputs[AmyResponse_out], outputs[BenResponse_out], outputs[corAnswer_out], outputs[
                    round_out] = AmyResponse, BenResponse, corAnswer, rnd+1
                outputs[response_out], outputs[answer_out], outputs[subj_out], outputs[outcome_out] = response, answer, subNum, outcome
                outputs[cards_iso_out], outputs[level_out], outputs[noise_out] = cards_iso, level, noise
                outputs[update_prob_out] = update_prob
                data = np.vstack((data, outputs))
        return data[1:]

    def approx_game_likelihood(self, param, state, Amy_cards, Ben_cards, order, outcomeArray):
        start1 = timeit.default_timer()
        '''
        return: a dictionary of dictionary or float
        If the key is int, it represents the rnd (1,2,or 3) where agent answers I know
        Then its value is a dictionary
            key (str): possible hand ('AA', 'A8', '88')
            value (float): their log likelihood
        If the key is '', it represents the agent answered I don't know till the end
            value (float): log likelihood
        '''
        results = []
        for i in range(self.nSample):
            res, ans, _ = self.agent_by_game(param, state, Amy_cards, Ben_cards, order, outcomeArray)
            results.append((tuple(res), ans))

        probs = {}
        for r in results:
            probs[r] = 0
        for r in results:
            probs[r] += 1
        for k in probs.keys():
            probs[k] = np.log(probs[k] / self.nSample)
        s = logsumexp(list(probs.values()))
        assert s < 0.01 and s > -0.01, s

        max_round = len(outcomeArray) # if the game ends in two rounds, then impossible to get to the third round
        look_up = {}
        cards = self.possible_hand
        convert = {1: (True,), 2: (False, True), 3: (False, False, True)}
        for i in range(1, max_round+1):
            look_up[i] = {}
        for i in range(1, max_round+1):
            for c in cards:
                if (convert[i], c) in probs.keys():
                    look_up[i][c] = probs[(convert[i], c)]
                else:
                    look_up[i][c] = np.log(0)
        if ((False,) * max_round, '') in probs.keys():
            look_up[''] = probs[((False,) * max_round, '')]
        else:
            look_up[''] = np.log(0)
        stop1 = timeit.default_timer()
        print("Cedegao time needed to get Log Likelihood: " + str(stop1 - start1))
        return look_up 

    def nLLH(self, continuous_param, Data, discrete_param):
        '''
        param: (list) Model parameters
        Data: (numpy matrix) The actual subject data
        return: (float) The negative log likelihood
        '''
        # get coloumn index for different task variables
        phase_out = self.colnamesOut.index('phase')
        round_out = self.colnamesOut.index('round')
        answer_out = self.colnamesOut.index('answer')
        cards_out = self.colnamesOut.index('cards')
        order_out = self.colnamesOut.index('order')
        outcomeArray_out = self.colnamesOut.index('outcomeArray')
        llh, games =0, np.unique(Data[:, phase_out]) # list of block numbers
        for g in games: # loop through games
            current_game = Data[Data[:, phase_out] == g]
            cardss, orders = np.unique(current_game[:, cards_out]), np.unique(current_game[:, order_out])
            rounds, answers = current_game[:, round_out], np.unique(current_game[:, answer_out])
            outcomeArray = eval(current_game[:, outcomeArray_out][0])
            assert len(cardss) == 1 and len(orders) == 1
            num_round, answer, cards, order = len(rounds), answers[0], cardss[0], orders[0].split(",")
            if np.isnan(sum(continuous_param)): # a wierd bug that scipy minimize sometimes sample [nan nan] as parameter
                continue
            Amy_cards, Ben_cards = cards[2:4], cards[4:6]
            LLH_look_up = self.approx_game_likelihood(list(discrete_param)+list(continuous_param), cards, Amy_cards,
                                                      Ben_cards, order, outcomeArray)  # TOP: get likelihood of
            # specific game here
            if answer in self.possible_hand:
                llh += LLH_look_up[num_round][answer]
            else:
                llh += LLH_look_up['']
        return -llh

class SUWNB(task_parameters, imperfect_update_model):
    '''
    Model that assumes subjects hold the full model in mind and stochastically eliminate nodes when updating it
    Parameter:
        update_prob (float): the chance of successfully eliminating a node when the update requires an elimination 
        noise (float): chance of random guessing IF the model says she shouldn't know
    Note: parameter_names, colnamesIn, colnamesOut are crucial in making sure the right value are retrieved for the right variable
    Use them to guide indexing in the functions
    '''
    def __init__(self, data, nSample = nSample):
        task_parameters.__init__(self, data)
        imperfect_update_model.__init__(self)
        self.name = 'SUWNB'
        self.nSample = nSample
        # parameters and their bounds
        self.discrete_parameter = {}
        self.continuous_parameter = {'update_prob':[0,1], 'noise':[0,1]}
        self.parameter_combination = list(it.product(*[self.discrete_parameter.values()]))
        # all possible combinations of the discrete parameters
        self.parameter_space = list(self.continuous_parameter.values())
        # all ranges of continuous parameters
        self.parameter_names = list(self.discrete_parameter.keys()) + list(
        self.continuous_parameter.keys())  # note the convention that discrete precedes continuous params
        # how to initialize continuous parameters
        self.initialize_method = {'update_prob':'uniform', 'noise': 'uniform'}
        
    def agent_by_round(self, state, players, announcements, graph, update_prob):
        '''
        state (string): 'AA88A8' (your cards, Amy's, Ben's)
        players: (list of strings) ['Amy', 'Ben', 'You'] (note: not the game order. must match announcement order)
        announcements: (1D np array) [] if subject goes first, [False, True] if subject goes third etc
        graph: (nx object) epistemic model
        intake_prob (float): one of the model parameters
        return: response of the current turn (bool), updated model (graph)
        '''
        G = graph
        for i in range(len(announcements)):
            response = self.compute_my_response(G, state[2:4], state[4:6])
            if response:
                return response, G
            player, announcement = players[i], announcements[i]
            G = self.update_model(G, update_prob, announcement, self.player_number[player])
        response = self.compute_my_response(G, state[2:4], state[4:6])
        return response, G
    def agent_by_game(self, param, state, Amy_cards, Ben_cards, order, outcomeArray):
        '''
        param (list): [update_prob, noise]
        state (str), order (list of str): cards and order in the game
        outcomeArray: (list of bool) [[FALSE, FALSE, FALSE], [FALSE, TRUE]] represents the correct announcements players should produce
        return: list of response (bool) for the entire game, the selected hand (str), correct response or not (bool)
        '''
        assert Amy_cards == state[2:4] and Ben_cards == state[4:6]

        noise, update_prob = param[self.parameter_names.index('noise')], param[self.parameter_names.index('update_prob')]
        G, player_idx = self.generate_partial_model(Amy_cards, Ben_cards), order.index('You')
        responses, answer = [], ''
        players_after, announcement_after_player = [], []
        outcome = True
        for rnd in outcomeArray:
            announcement_before_player, announcement_after_player = np.append(announcement_after_player, rnd[:player_idx]), rnd[player_idx:]
            players_before, players_after = np.append(players_after, order[:player_idx]), order[player_idx:]
            response, G = self.agent_by_round(state, players_before, announcement_before_player, G, update_prob)
            if response and response != -1:
                responses.append(response)
                possibilities = []
                for i in list(G.nodes):
                    if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][4:6] == Ben_cards:
                        possibilities.append(i)
                assert len(possibilities) == 1
                answer = G.nodes[possibilities[0]]['state'][:2]
                outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                break # if know, the game ends
            else:
                if response == -1: # inconsistent and have to guess
                    if np.random.random_sample() < 0.5: # guess don't know
                        responses.append(False)
                        outcome = outcome and not rnd[order.index('You')]
                    else: # guess know and randomly choose one from candidates
                        responses.append(True)
                        answer = np.random.choice(self.possible_hand) # randomly choose one of three
                        outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                        break
                else: # don't know
                    if np.random.random_sample() < noise:
                        responses.append(True)
                        answer = np.random.choice(self.possible_hand) # random choose one of three
                        outcome = outcome and rnd[order.index('You')] and answer == state[:2]
                        break
                    else:
                        responses.append(response) # honestly say don't know
                        outcome = outcome and not rnd[order.index('You')]
        return responses, answer, outcome
    def agent(self, subNum, param):
        '''
        param: same as above
        subNum (int): which subject data we are simulating
        return: numpy data matrix
        '''
        colnamesOut = self.colnamesOut + self.parameter_names
        data = np.ones(len(colnamesOut))  # initialize output data. note for small dataset (<50000 rows) numpy is more efficient
        # get coloumn index for different task variables
        phase_idx = self.colnamesIn.index('phase')
        cards_idx = self.colnamesIn.index('cards')
        order_idx = self.colnamesIn.index('order')
        outcomeArray_idx = self.colnamesIn.index('outcomeArray')
        # get coloumn index for output variables
        phase_out = colnamesOut.index('phase')
        cards_out = colnamesOut.index('cards')
        order_out = colnamesOut.index('order')
        outcomeArray_out = colnamesOut.index('outcomeArray')
        AmyResponse_out = colnamesOut.index('AmyResponse')
        BenResponse_out = colnamesOut.index('BenResponse')
        corAnswer_out = colnamesOut.index('corAnswer')
        round_out = colnamesOut.index('round')
        response_out = colnamesOut.index('response')
        answer_out = colnamesOut.index('answer')
        subj_out = colnamesOut.index('subj')
        outcome_out = colnamesOut.index('outcome')
        cards_iso_out = colnamesOut.index('cards_iso')
        update_prob_out = colnamesOut.index('update_prob')
        noise_out = colnamesOut.index('noise')
        # remove redundant rows due to rnd info
        real_data = f.unique([tuple(row) for row in self.data[subNum - 1]])
        # get task parameters
        phases, all_cards, all_order, all_outcomeArray = real_data[:, phase_idx], real_data[:, cards_idx], real_data[:, order_idx], real_data[:, outcomeArray_idx]
        assert len(all_cards) == len(all_order) == len(phases)

        for i in range(len(phases)): #for each game
            phase, order, cards, outcomeArray = phases[i], all_order[i], all_cards[i], all_outcomeArray[i]
            outcomeArray_evaled, cards_iso, Amy_cards, Ben_cards = eval(outcomeArray), self.iso_map[cards], cards[2:2+self.num_cards_each], cards[4:4+self.num_cards_each]
            responses, answer, outcome = self.agent_by_game(param, cards, Amy_cards, Ben_cards, order.split(","), outcomeArray_evaled)
            for rnd in range(len(responses)):
                outcomeSubarray, response, log = outcomeArray_evaled[rnd], responses[rnd], [None]*3
                update_prob, noise = param
                for o in range(len(outcomeSubarray)):
                    log[o] = outcomeSubarray[o]
                AmyResponse, BenResponse, corAnswer = log[order.split(",").index('Amy')], log[order.split(",").index('Ben')], log[order.split(",").index('You')]
                # prepare output row to append to data
                outputs = np.empty(len(colnamesOut), dtype=object)
                outputs[phase_out], outputs[cards_out], outputs[order_out], outputs[outcomeArray_out] = phase, cards, order, outcomeArray
                outputs[AmyResponse_out], outputs[BenResponse_out], outputs[corAnswer_out], outputs[
                    round_out] = AmyResponse, BenResponse, corAnswer, rnd+1
                outputs[response_out], outputs[answer_out], outputs[subj_out], outputs[outcome_out] = response, answer, subNum, outcome
                outputs[cards_iso_out], outputs[noise_out] = cards_iso, noise
                outputs[update_prob_out] = update_prob
                data = np.vstack((data, outputs))
        return data[1:]

    def approx_game_likelihood(self, param, state, Amy_cards, Ben_cards, order, outcomeArray):
        '''
        return: a dictionary of dictionary or float
        If the key is int, it represents the rnd (1,2,or 3) where agent answers I know
        Then its value is a dictionary
            key (str): possible hand ('AA', 'A8', '88')
            value (float): their log likelihood
        If the key is '', it represents the agent answered I don't know till the end
            value (float): log likelihood
        '''
        results = []
        for i in range(self.nSample):
            res, ans, _ = self.agent_by_game(param, state, Amy_cards, Ben_cards, order, outcomeArray)
            results.append((tuple(res), ans))

        probs = {}
        for r in results:
            probs[r] = 0
        for r in results:
            probs[r] += 1
        for k in probs.keys():
            probs[k] = np.log(probs[k] / self.nSample)
        s = logsumexp(list(probs.values()))
        assert s < 0.01 and s > -0.01, s

        max_round = len(outcomeArray) # if the game ends in two rounds, then impossible to get to the third round
        look_up = {}
        cards = self.possible_hand
        convert = {1: (True,), 2: (False, True), 3: (False, False, True)}
        for i in range(1, max_round+1):
            look_up[i] = {}
        for i in range(1, max_round+1):
            for c in cards:
                if (convert[i], c) in probs.keys():
                    look_up[i][c] = probs[(convert[i], c)]
                else:
                    look_up[i][c] = np.log(0)
        if ((False,) * max_round, '') in probs.keys():
            look_up[''] = probs[((False,) * max_round, '')]
        else:
            look_up[''] = np.log(0)
        return look_up
    def nLLH(self, continuous_param, Data):
        '''
        param: (list) Model parameters
        Data: (numpy matrix) The actual subject data
        return: (float) The negative log likelihood
        '''
        # get coloumn index for different task variables
        phase_out = self.colnamesOut.index('phase')
        round_out = self.colnamesOut.index('round')
        answer_out = self.colnamesOut.index('answer')
        cards_out = self.colnamesOut.index('cards')
        order_out = self.colnamesOut.index('order')
        outcomeArray_out = self.colnamesOut.index('outcomeArray')
        llh, games =0, np.unique(Data[:, phase_out])  # list of block numbers
        for g in games:  # loop through games
            current_game = Data[Data[:, phase_out] == g]
            cardss, orders = np.unique(current_game[:, cards_out]), np.unique(current_game[:, order_out])
            rounds, answers = current_game[:, round_out], np.unique(current_game[:, answer_out])
            outcomeArray = eval(current_game[:, outcomeArray_out][0])
            assert len(cardss) == 1 and len(orders) == 1
            num_round, answer, cards, order = len(rounds), answers[0], cardss[0], orders[0].split(",")
            if np.isnan(sum(continuous_param)): # a wierd bug that scipy minimize sometimes sample [nan nan] as parameter
                continue
            Amy_cards, Ben_cards = cards[2:4], cards[4:6]
            LLH_look_up = self.approx_game_likelihood(list(continuous_param), cards, Amy_cards, Ben_cards, order, outcomeArray)
            if answer in self.possible_hand:
                llh += LLH_look_up[num_round][answer]
            else:
                llh += LLH_look_up['']
        return -llh
