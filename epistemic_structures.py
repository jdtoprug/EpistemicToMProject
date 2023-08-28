'''
FILE (minus some comments) FROM CEDEGAO ET AL 2021
I (J.D. Top) DID NOT MAKE THIS!
Novel comments and new functions should be labelled as such
'''
import numpy as np
import networkx as nx 
import itertools as it
import utils as f #import src.utils as f NOTE: changed
import timeit

updateglobal = 0

class modal_model(object):
    '''
    The epistemic model of the task and how to update it
    '''
    def __init__(self):
    #Top: All possible worlds in the Aces and Eights game with 4 Aces, 4 Eights, and 3 players
        self.all_states = ['AAAA88', 'AAA8A8', 'AAA888',
                  'AA88AA', 'AA88A8', 'AA8888',
                  'A8AAA8', 'A8AA88', 'A8A8AA',
                  'A8A8A8', 'A8A888', 'A888AA',
                  'A888A8', '88AAAA', '88AAA8',
                  '88AA88', '88A8AA', '88A8A8',
                  '8888AA'] # by default A before 8 each pair
        self.num_all_states = len(self.all_states) #Number of possible possible worlds
        self.player_number = {'You':1, 'Amy':2, 'Ben':3} # Top: Dictionary mapping names to players/player numbers #'you' means the participant
    #Top: for each player, a list of outgoing non-reflexive, non-symmetric edges (tuples) in the possible world model. Note that indices start at 1.
        self.player_one_uncertainty = [(1, 8), (1, 16), (8, 16),
                           (7, 15), (2, 10), (10, 18),
                           (2, 18), (5, 13), (4, 12),
                           (12, 19), (4, 19), (9, 17),
                           (3, 11)]
        self.player_two_uncertainty = [(1, 3), (1, 6), (3, 6),
                           (7, 10), (10, 13), (7, 13),
                           (14, 19), (14, 17), (17, 19),
                           (9, 12), (18, 15), (2, 5),
                           (8, 11)]
        self.player_three_uncertainty = [(4, 5), (5, 6), (4, 6),
                             (9, 10), (9, 11), (10, 11),
                             (14, 15), (15, 16), (14, 16),
                             (2, 3), (12, 13), (7, 8),
                             (17, 18)]
    #Top: The epistemic_level l SUWEB with update_prob = 1 and noise = 0 needs to solve the puzzle (note that AYB is equivalent to BYA)
        self.inference_level = {'A8A8AA:You,Amy,Ben':4, 'A8A8AA:Amy,Ben,You':4, 'AAA888:You,Amy,Ben':4, 
                                'A8AAA8:Amy,Ben,You':4, 'AA88A8:Amy,Ben,You':4, 'AAA8A8:You,Amy,Ben':3, 
                                'AAA8A8:Amy,Ben,You':3, 'AAA8A8:Ben,You,Amy':3, 'A8A8A8:Amy,Ben,You':3, 
                                'A8A8A8:Ben,You,Amy':3, 'A8A8A8:You,Amy,Ben':3, 'A8A8AA:Ben,You,Amy':2, 
                                'A8AAA8:You,Amy,Ben':2, 'A8AAA8:Ben,You,Amy':2, 'AA88A8:Ben,You,Amy':2, 
                                'AA88A8:You,Amy,Ben':2, 'AAA888:Amy,Ben,You':4, 'AAA888:Ben,You,Amy':2,
                                'A8AA88:Amy,Ben,You':1, 'A8AA88:Ben,You,Amy':1, 'A8AA88:You,Amy,Ben':1, 
                                'AAAA88:Ben,You,Amy':1, 'AAAA88:You,Amy,Ben':1, 'AAAA88:Amy,Ben,You':1, 
                                'AA88AA:Amy,Ben,You':1, 'AA88AA:Ben,You,Amy':1, 'AA88AA:You,Amy,Ben':1,
                                'AA8888:Ben,You,Amy':0, 'AA8888:Amy,Ben,You':0, 'AA8888:You,Amy,Ben':0}
    #Top: e.g. AA8888 and 88AAAA are the same puzzle
        self.iso_map = {'8888AA': 'AAAA88', '88A8AA': 'AAA888', '88AAAA': 'AA8888',
                        '88AAA8': 'AA88A8', '88A8A8': 'AAA8A8', '88AA88': 'AA88AA',
                        'A888AA': 'A8AA88', 'A8A888': 'A8A8AA', 'A888A8': 'A8AAA8',
                        'A8A8A8':'A8A8A8',
                        'AAAA88': 'AAAA88', 'AAA888': 'AAA888', 'AA8888': 'AA8888',
                        'AA88A8': 'AA88A8', 'AAA8A8': 'AAA8A8', 'AA88AA': 'AA88AA',
                        'A8AA88': 'A8AA88', 'A8A8AA': 'A8A8AA', 'A8AAA8': 'A8AAA8'} # maps to equivalent states
    # helper functions
    #Top: A is 1 and 8 is 8
    def map_str_to_int(self,s):
        if s == 'A':
            return 1
        elif s == '8':
            return 8
    #Top: Takes as input players_dict, a dictionary with for each player name (string), their card tuple (ints), and play_order, a list of strings denoting the order of play, and n, a player, and returns whether that player has A8/8A or not
    def a_and_e(self,n, players_dict, play_order):
        cards = players_dict[play_order[n]] #Top: take the cards of player n
        if cards == (1, 8) or cards == (8, 1): #Return True if this player has A8 or 8A, false otherwise
            return True
        else:
            return False
    #Top:
    def convert_default_to_game_order(self, tuple_state_order):
        '''
        The experiment code used a different convention for representing state
        Instead of fixed as PAB, it follows the order
        '''
        state = tuple_state_order[0]
        order = tuple_state_order[1]

        def map_name_to_num(name): #Top: Given a player name, returns the range without a state string that is that player's cards, e.g. 'A8AA88' assigns cards 'A8' to the player, 'You'
            if name == 'You':
                return [0, 2]
            elif name == 'Amy':
                return [2, 4]
            elif name == 'Ben':
                return [4, 6]

        s = ''
        for i in range(len(order)):
            indices = map_name_to_num(order[i])
            s = s + state[indices[0]:indices[1]]
        return list(s)
    #Top: 
    def convert_game_order_to_default(self, tuple_state_order):
        '''
        The experiment code used a different convention for representing state
        Instead of fixed as PAB, it follows the order
        '''
        state = tuple_state_order[0]
        order = tuple_state_order[1]

        def map_name_to_num(name):
            if name == 'You':
                return [0, 2]
            elif name == 'Amy':
                return [2, 4]
            elif name == 'Ben':
                return [4, 6]

        l = [None]*len(state)
        for i in range(len(order)):
            indices = map_name_to_num(order[i])
            l[indices[0]:indices[1]] = state[2*i:2*i+2]
        return l
    # graph update functions
    #Top:  Uses possible worlds and edge lists to generate full model. Does not make the model reflexive/symmetric/transitive!
    def generate_full_model(self):
        '''
        Generate the full initial epistemic structure
        return (nx object): the initial full epistemic structure
        '''
        G = nx.Graph() #Top: Create empty Graph object (note that Graph, unlike DiGraph, is an UNDIRECTED graph)
        index = np.arange(1, self.num_all_states + 1, 1) #Top: Array starting at 1, ending BEFORE number of possible worlds + 1, with interval 1
        for i in range(self.num_all_states): #Top: range 5 returns [0,1,2,4,]. Add ALL possible worlds to the graph.
            G.add_node(index[i], state=self.all_states[i]) #Top: Node is e.g. 0, 18, which has key state and value e.g. 'AA8888'. index[i] increments value by 1.
    #Top: Add each player's edges (from container of 2-tuples) and label them with player number (1=You, 2=Amy, 3=Ben)
        G.add_edges_from(self.player_one_uncertainty, player=1)
        G.add_edges_from(self.player_two_uncertainty, player=2)
        G.add_edges_from(self.player_three_uncertainty, player=3)
        return G

    #Top: Given a graph of current possibilities (corresponding to observations), Amy's OBSERVED cards, and Ben's OBSERVED cards, returns true if the player KNOWS his/her cards
    def compute_my_response(self, graph, Amy_cards, Ben_cards):
        '''
        Compute participant's response 
        return (-1 or bool): 
            -1: eliminated all possible states that are consistent with Amy's and Ben's cards seen by the participant (doesn't happen for DEL)
            True: only one possible state left so respond 'I know my cards'
            False: more than one possible states left so respond 'I don't know my cards'
        '''
        G = graph
        possibilities = [] # possible states for the participant
        for i in list(G.nodes): #Top: Loop over nodes
            if G.nodes[i]['state'][2:4] == Amy_cards and G.nodes[i]['state'][4:6] == Ben_cards: #Top: if input corresponds to this possible world
                possibilities.append(i) #Top: Add it to the list of possibilities
        n = len(possibilities) #Top: Calculate the number of possibilities
        if n == 0: #Top: There are no possibilities. Someone lied or made a mistake.
            return -1
        elif n == 1: #Top: There is exactly one possibility, the player knows his/her cards
            return True
        else: #Top: there are two or more possibilities. The player's cards are not known by the player.
            return False

    def compute_correct_response(self, graph, state, player):
        '''
        Given a model, state of cards ('AA88A8'), and player number (int), 
        return a bool whether the player should know her card
        '''
        G = graph
        state_index = [i for i, d in G.nodes(data=True) if d['state'] == state][0] #Top: Take the first item from a list of node identifiers where the game state corresponds to the input state (identifier for input state)
        players_with_uncertainties = set([G.edges[state_index, v]['player'] for (u, v) in G.edges([state_index])]) #Top: Make a set out of a list of players (owners) of all outgoing edges (u,v) from input node. 
        if player not in players_with_uncertainties:
            return True #Top: true if no outgoing edges for input player, otherwise false
        else:
            return False

    #Top: Given a player, that player's possible world model, and the current observed state, output a list of possible states given this observation
    def compute_possible_states(self, graph, state, player):
        '''
        Give the list of possible states compatible with a model for a player
        player: 1,2 or 3
        '''
        G = graph
        state_index = [i for i, d in G.nodes(data=True) if d['state'] == state][0] #Top: get the identifier of the possible world
        other_states_possible = [G.nodes(data=True)[v]['state'] for (u, v) in G.edges([state_index]) 
                                      if G.edges[state_index, v]['player'] == player] #Top: Get the outgoing edges from the input state, filter out those that belong to the player, and get the state numbers the edges are pointing towards (data=False only returns node, data=True also returns dictionary)
        return [state]+other_states_possible #Top: append other possible worlds to current and return

    #Top: given a possible world model and a player, outputs the updated model if that player states 'I don't know'
    def update_model_neg(self, graph, player):
        '''
        Update the model if a player announces she doesn't know
        '''
        G = graph.copy() #Top: as we don't want to modify the original graph
        nodes_to_del = [] #Top: nodes to be deleted.
        for n in G.nodes(): #Top: loop over nodes.
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])]) #Top: for outgoing edges from that node, make a set of players
            if player not in players_with_uncertainties: #Top: if player is NOT uncertain for that node
                nodes_to_del.append(n) #Top: mark that node for deletion
        print("nodes_to_del: " + str(nodes_to_del))  # REMOVE
        for n in nodes_to_del: #Top: deleted marked nodes
            G.remove_node(n)
        return G

    #Top: given a possible world model and a player, outputs the updated model if that player states 'I know'
    def update_model_pos(self, graph, player):
        '''
        Update the model if a player announces knows
        '''
        G = graph.copy() #Top: as we don't want to modify the original graph
        nodes_to_del = [] #Top: nodes to be deleted.
        for n in G.nodes(): #Top: loop over nodes.
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])]) #Top: make a set of players with outgoing edges from that node
            if player in players_with_uncertainties: #Top: if player IS uncertain for that node
                nodes_to_del.append(n) #Top: mark that node for deletion
        print("nodes_to_del: " + str(nodes_to_del))  # REMOVE
        for n in nodes_to_del: #Top: deleted marked nodes
            G.remove_node(n)
        return G

    #Top: Given a possible worlds model, a player, and that player's announcement (T = know, F = don't know), return the updated model after that announcement
    def update_model(self, graph, announcement, player_number):
        '''
        Update the model according to a player's announcement
        '''
        if announcement: #Top: if the player knows
            G = self.update_model_pos(graph, player_number) #Top: update with that function
        else:
            G = self.update_model_neg(graph, player_number) #Top: otherwise update with the 'I don't know' function
        return G

    #Top: Given the card distribution among players and the order of play for each round, returns a list with, for each round, a list of correct responses for THE player
    def compute_subj_response(self,cards, order):
        '''
        cards: a list of 8 characters representing 8 cards Top: an --> a
        e.g., ['A', '8', 'A', 'A', '8', '8', 'A', '8'] Top: where the first six are the state
        order: ['Ben', 'You', 'Amy']
        
        return (list of lists): whether player 1 (the human subject) should know their cards or not
        Each innerlist represents a round
        
        Important: first two are USER, second two are Amy, third two are Ben
        Amy has pn (player number) 2 and Ben is 3
        '''
        G = modal_model.generate_full_model(self) #Top: create full model G
        state = ''.join(cards)[:6] #Top: "A8AA88" in the example above

        curr_asmt = [self.map_str_to_int(s) for s in cards] #Top: list of ints from list of strings ([1,8,1,1,8,8,1,8] in the example above)
    #Top: tuple of each player's cards as ints
        p1_cards = tuple(curr_asmt[:2])
        p2_cards = tuple(curr_asmt[2:4])
        p3_cards = tuple(curr_asmt[4:6])

        players_cards = [p1_cards, p2_cards, p3_cards] #Top: list of tuples of ints with each player's cards
        players_name = ['You', 'Amy', 'Ben'] #Top: player names (duh)
        players_dict = dict(zip(players_name, players_cards)) #Top: dictionary - for each player name (string), their card tuple (ints)

        play_order = order #Top: order is a list of strings, each of which is a player name
        user_order = order.index('You') + 1 #Top: how manyth turn player gets (1, 2, or 3)
        curr_player_num = -1
        round_num = -1
        game_over = False #Top: stop condition for while loop

        early_stop_cond_1 = play_order[1] != 'You' and self.a_and_e(1, players_dict, play_order) #Top: whether the player is NOT the middle  turn, and the middle player has A8/8A
        early_stop_cond_2 = play_order[1] == 'You' and self.a_and_e(0,
                                                               players_dict, play_order) and self.a_and_e(2, players_dict,
                                                                                                     play_order) #Top: whether the player IS the middle turn, and both OTHER players both have A8/8A
        early_stop = early_stop_cond_1 or early_stop_cond_2 #Top: whether either of this hold
        responses = [] #Top: list of lists of bools, where T is 'I know my cards' and F is 'I don't know my cards', with each sublist being the three responses for a round
        while game_over == False: #Top: each loop corresponds to a single turn ?
            curr_player_num = (curr_player_num + 1) % 3 #Top: current player, loops 0-1-2-0-1-2-0-1-etc
            curr_player = play_order[curr_player_num] #Top: name of the current player as string (You, Amy, Ben)
            if curr_player_num == 0: #Top: if it is a new round
                responses.append(list()) #Top: create new sublist for responses of current round
                round_num += 1 #Top: increment the round number (starting at 0) - a round is one turn for each player
            if curr_player == 'You': #Top: if it is the MODEL's turn
                pn = 1 #Top: player number = 1 (1=You, 2=Amy, 3=Ben)
                correct_response = self.compute_correct_response(G, state, 1) #Top: given model, state ('A8AA88') and player number, compute Bool whether player knows cards, in this case You
                if correct_response:
                    game_over = True #Top: stop if you know
                else:
                    if early_stop and round_num == 2:  # How many rounds? Stop if you are the current player, it is the second round, and the early stop condition holds.
                        game_over = True
                    else:
                        G = modal_model.update_model_neg(self, G, pn) #Top: if you don't stop, update the model with an 'I don't know' response by the player
                responses[round_num].append(correct_response) #Top: always append response to response list of current round
            else: #Top: if it is the turn of one of the perfect reasoners
                if curr_player == 'Amy':
                    pn = 2 #Top: Player number = 2 (1=You, 2=Amy, 3=Bob)
                else:
                    pn = 3
                correct_response = self.compute_correct_response(G, state, 1)
                agent_response = self.compute_correct_response(G, state, pn) #Top: whether the player knows/doesn't know
                if correct_response:
                    game_over = True
                if agent_response:
                    G = modal_model.update_model_pos(self, G, pn) #Top: update possible world model when agent says 'I know'
                else:
                    G = modal_model.update_model_neg(self, G, pn) #Top: update possible world model when agent says 'I don't know'
                responses[round_num].append(correct_response) #Top: always append response to response list of current round (even if it is not the player's turn!)
        return responses

    #Top: Given the card distribution among players (list of strings of 'A'/'8' and the order of play (list of strings like 'You'/'Amy'/'Bob'), returns a list with, for each round, a list of the correct responses for EACH player
    def compute_game_response(self, cards, order):
        '''
        cards: a list of 8 characters representing 8 cards #Top: an list --> a list
        e.g., ['A', '8', 'A', 'A', '8', '8', 'A', '8'] #Top: where the first 6 are the game state, e.g. 'A8AA88' means You = A8, Amy = AA, Bob = 88
        order: ['Ben', 'You', 'Amy']
        player: 1, 2, or 3
        
        return (list of lists): whether each agent knows their cards or not
        Each innerlist represents a round
        
        Important: first two are USER, second two are Amy, third two are Ben
        Amy has pn (player number) 2 and Ben is 3
        '''
        G = self.generate_full_model() #Top: create full possible worlds model for the game
        state = ''.join(cards)[:6] #Top: state is a string such as 'A8AA88'

        curr_asmt = [self.map_str_to_int(s) for s in cards] #Top: list of ints, where 1 is A and 8 is 8 with game state
        p1_cards = tuple(curr_asmt[:2]) #Top: 2-tuple of ints for the player
        p2_cards = tuple(curr_asmt[2:4]) #Top: 2-tuple of ints for Amy
        p3_cards = tuple(curr_asmt[4:6]) #Top: 2-tuple of ints for Bob (Ben?)
        players_cards = [p1_cards, p2_cards, p3_cards] #Top: list of 2-tuples with each player's cards
        players_name = ['You', 'Amy', 'Ben']
        players_dict = dict(zip(players_name, players_cards)) #Top: dictionary mapping player name to two-tuple of ints which are their cards

        play_order = order
        user_order = order.index('You') + 1 #Top: player's position in the game: 1, 2, or 3
        curr_player_num = -1 #Top: 0, 1, 2, 0, 1, 2, 0, 1...
        round_num = -1 #Top: starts at 0, increments for each round

        game_over = False #Top: stop condition of main game loop

        early_stop_cond_1 = play_order[1] != 'You' and self.a_and_e(1, players_dict, play_order) #Top: True if the player does not have the middle turn, and the middle player has A8
        early_stop_cond_2 = play_order[1] == 'You' and self.a_and_e(0,
                                                               players_dict, play_order) and self.a_and_e(2, players_dict,
                                                                                                     play_order) #Top: True if the player DOES have the middle turn, and the other players both have A8

        early_stop = early_stop_cond_1 or early_stop_cond_2 #Top: Early stop condition

        responses = []

        while game_over == False: #Top: main game loop, one loop for each turn

            curr_player_num = (curr_player_num + 1) % 3 #Top: current player, 0, 1, or 2
            curr_player = play_order[curr_player_num] #Top: String of current player, "You", "Amy", or "Ben"

            if curr_player_num == 0: #Top: a new round has started, so
                responses.append(list()) #Top: make a new sublist for responses in this round
                round_num += 1 #Top: increment the round number

            if curr_player == 'You': #Top: if it is the player's turn
                pn = 1 #Top: You = 1, Amy = 2, Ben = 3
                correct_response = self.compute_correct_response(G, state, pn) #Top: True if player pn under card distribution state with possible worlds model G KNOWS his/her cards
                if correct_response: #Top: Stop the game if the player knows
                    game_over = True
                else: #Top: If the current player does not know his/her cards...
                    if early_stop and round_num == 2:
                        game_over = True
                    else: #Top: ...and the early stop condition isn't met...
                        G = self.update_model_neg(G, pn) #Top: update the possible worlds model with an "I don't know" announcement
                responses[round_num].append(correct_response) #Top: Append the player's response

            else: #Top: if it is an AGENT'S turn
                if curr_player == 'Amy':
                    pn = 2 #Top: You = 1, Amy = 2, Ben = 3
                else:
                    pn = 3
                correct_response = self.compute_correct_response(G, state, pn) #Top: Calculate whether player pn KNOWS his/her cards
        #Top: Update the possible worlds model based on the 'know'/'don't know' announcement
                if correct_response: 
                    G = self.update_model_pos(G, pn)
                else:
                    G = self.update_model_neg(G, pn)
                responses[round_num].append(correct_response) #Top: Append the agent's response

        return responses
'''
Bounded modal model - has a set of internal and peripheral states. If the initial possibilities based on observations (e.g. Amy has AA, ben has A8) are R(W) (in this example A8 AA A8 and 88 AA A8), then the peripheral states are all states that are reached from the initial possibilities in l steps, and the internal states are all states that can be reached from the initial possibilities within l-1 steps)
'''
class bounded_modal_model(modal_model):
    '''
    Top: upon object creation, calls the initialization function of modal_model, which gives it:
    all_states - a list of strings like 'A8AA88' with all possible possible worlds in the game
    num_all_states - the length of the above list
    player_number - a dictionary mapping player names (e.g. 'You') to ints (1).
    player_X_uncertainty - with X 'one', 'two', and 'three', a list of 2-tuples of ints where each 2-tuple is an edge, maps possible worlds in all_states to each other (where indices start at 1)
    inference_level - for each unique combination of state and player order (where 'Amy, You, Ben' is equivalent to 'Ben, You, Amy', the inference level l, that is, the minimum parameter epistemic_level the SUWEB model with noise = 0 and update_prob = 1 needs to solve this case
    '''
    def __init__(self):
        super().__init__()

    @staticmethod  # Top: NEW
    def printglobals():
        global updateglobal
        print("Total update time c: " + str(updateglobal))

    #Top: Uses modal model object's states and edges to generate full possible world model as graph. Makes the model symmetric, but does NOT make the model transitive/reflexive
    def generate_full_model(self):
        '''
        Generate the full epistemic structure (as directed graph)
        '''

    #Top: given a list of 2-tuples, returns the list with the two elements of each of the 2-tuples swapped (but does not change the order of the tuples themselves)
        def swap(lst):
            l = []
            for x in lst:
                l.append((x[1], x[0]))
            return l

        G = nx.DiGraph() #Top: unlike generate_full_model, uses a DiGraph (directed graph) instead of a Graph
        index = np.arange(1, self.num_all_states + 1, 1) #Top: List of ints 1, 2, ..., #states - 1, #states
        for i in range(self.num_all_states): #Top: Iteratively add all states (e.g. 'AA88A8' to graph)
            G.add_node(index[i], state=self.all_states[i])
    #Top: Add all edges in the possible world model to the graph, as well as their reverse
        G.add_edges_from(self.player_one_uncertainty, player=1)
        G.add_edges_from(self.player_two_uncertainty, player=2)
        G.add_edges_from(self.player_three_uncertainty, player=3)
        G.add_edges_from(swap(self.player_one_uncertainty), player=1)
        G.add_edges_from(swap(self.player_two_uncertainty), player=2)
        G.add_edges_from(swap(self.player_three_uncertainty), player=3)
        return G

    #Top: given a possible world model and a player (string), return the updated model if that player announced 'I don't know'
    def update_model_neg(self, graph, player):
        verbose = False
        '''
        Update the model if a player announces she doesn't know
        '''
        G = graph.copy() #Top: as we don't want to modify the original graph
        nodes_to_del = [] #Top: lists nodes marked for delition
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']] #Top: list with nods that may be removed (in the algorithm, interior states may be removed but peripheral states may not)
        for n in elimable: #Top: loop over nodes
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])]) #Top: loop over outgoing edges of node n, get its player (label) and construct a set of these player labels
            if player not in players_with_uncertainties: #Top: Mark the node for deletion if it has no outgoing edges (announcement was 'I don't know' so nodes with no outgoing edges are impossible)
                nodes_to_del.append(n)
        if verbose:
            print("nodes_to_del: " + str(nodes_to_del))  # REMOVE
        for n in nodes_to_del: #Top: Remove all nodes marked for deletion from the possible worlds model and return it
            G.remove_node(n)
        return G

    #Top: given a possible world model and a player (string), return the updated model if that player announced 'I know'
    def update_model_pos(self, graph, player):
        verbose = False
        '''
        Update the model if a player announces knows
        '''
        G = graph.copy() #Top: since we don't want to modify the full possible world model
        nodes_to_del = [] #Top: list of nodes marked for deletion
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']] #Top: list of nodes that are interior (not peripheral) and therefore may be deleted)
        for n in elimable: #Top: loop over interior nodes (nodes that may be deleted)
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])]) #Top: for each outgoing edge in that node, copy the edge label (owner/name) into a set
            if player in players_with_uncertainties: #Top: player announced 'I know', so delete any node with outgoing edges.
                nodes_to_del.append(n)
        if verbose:
            print("nodes_to_del: " + str(nodes_to_del))  # REMOVE
        for n in nodes_to_del: #Top: then actually delete those nodes and return the possible worlds model
            G.remove_node(n)
        return G

    #Top: Given a possible worlds model, an announcement (True for 'I know', False for 'I don't know'), and a player number (1 = You, 2 = Amy, 3 = Ben), return the updated model based on that announcement by that player
    def update_model(self, graph, announcement, player_number):
        start1 = timeit.default_timer()
        '''
        Update the model according to a player's announcement
        '''
        if announcement:
            G = self.update_model_pos(graph, player_number) #Top: Update the model with an 'I know' announcement
        else:
            G = self.update_model_neg(graph, player_number) #Top: Update the model with an 'I don't know' announcement
        stop1 = timeit.default_timer()
        global updateglobal
        updateglobal += stop1 - start1
        return G

    #Top: Given what the player sees are Amy's cards and Ben's cards, and epistemic level, creates the partial model with internal and peripheral states (if l is not given l=4)
    def generate_partial_model(self, Amy_cards, Ben_cards, level=4):
        G, full_graph = nx.DiGraph(), self.generate_full_model() #Top: Create an empty directed graph G and a full undirected graph full_graph
        index = np.arange(1, self.num_all_states + 1, 1) #Top: [1,2,3,...,#states]
        initial = [] #Top: states where Amy has Amy_cards and Ben has Ben_cards
        for i in range(self.num_all_states): #Top: loop i over 1 to #states
            if self.all_states[i][2:4] == Amy_cards and self.all_states[i][4:6] == Ben_cards: #Top: add state to initial if Amy_cards and Ben_cards are correct
                initial.append(i)
        for i in initial: #Top: loop over states where Amy has Amy_cards and Ben has Ben_cards
            G.add_node(index[i], state=self.all_states[i], elimable=True) #Top: Add these to empty directed graph G - these are the interior states for l = 0
        if len(list(G.nodes)) > 1:
            perm = it.permutations(list(G.nodes), 2) #Top: print(list(itertools.permutations([1,2,3], 2))) prints "[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]"
            for edge in list(perm): #Top: Make all interior states 'fully' connected (symmetric, transitive, but NOT reflexive), labelled for the PLAYER
                G.add_edge(edge[0], edge[1], player=1)
        for l in range(int(level)): #Top: for each epistemic level (0,1,2,3 if level is 4)
            nodes = list(G.nodes)
            to_remove = [] #Top: unused
            for u in nodes: #Top: loop over current internal nodes
                for v in full_graph.neighbors(u): #Top: loop over neighbours, in the full graph, of internal node (excluding initials)
                    if v not in list(G.nodes):
                        if l < int(level) - 1:
                            G.add_node(v, state=full_graph.nodes[v]['state'], elimable=True) #Top: Add and label as internal node
                        else:
                            G.add_node(v, state=full_graph.nodes[v]['state'], elimable=False) #Top: Add and label as peripheral node
                    G.add_edge(u, v, player=full_graph[u][v]['player']) #Top: add outgoing edge from internal node to peripheral node

        return G
  
    '''
    Top: Imperfect update model object that can be created from a bounded modal model - imperfect update model removes a node with probability update_prob
    '''
class imperfect_update_model(bounded_modal_model):
    '''
    upon object creation, calls the initialization function of modal_model, which gives it:
    all_states - a list of strings like 'A8AA88' with all possible possible worlds in the game
    num_all_states - the length of the above list
    player_number - a dictionary mapping player names (e.g. 'You') to ints (1).
    player_X_uncertainty - with X 'one', 'two', and 'three', a list of 2-tuples of ints where each 2-tuple is an edge, maps possible worlds in all_states to each other (where indices start at 1)
    inference_level - for each unique combination of state and player order (where 'Amy, You, Ben' is equivalent to 'Ben, You, Amy', the inference level l, that is, the minimum parameter epistemic_level the SUWEB model with noise = 0 and update_prob = 1 needs to solve this case
    '''
    def __init__(self):
        super().__init__()
    #Top: Given a possible world model, a player, and update_prob, update the model if that player announces "I don't
        # know"
    def update_model_neg(self, graph, elim_prob, player):
        '''
        Update the model if a player announces she doesn't know
        '''
        G = graph.copy() #Top: we don't want to modify the original graph
        nodes_to_del = [] #Top: list of nodes marked for deletion
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']] #Top: list of interior nodes (that may be deleted)
        for n in elimable: #Top: loop over interior nodes (which may be deleted, while peripheral nodes may not)
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])]) #Top: set of players (e.g. "Amy") that appear on an outgoing edge from this node
            if player not in players_with_uncertainties: #Top: if the announcing player KNOWS in this state
                nodes_to_del.append(n) #Top: mark this node for removal (since he/she announced "I don't know")
        for n in nodes_to_del: #Top: each node is removed with probability update_prob
            if np.random.random_sample() < elim_prob:
                G.remove_node(n)
        return G

    #Top: given a possible world model, a player, and update_prob, update the model if that players announces "I know"
    def update_model_pos(self, graph, elim_prob, player):
        '''
        Update the model if a player announces knows
        '''
        G = graph.copy() #Top: copy the graph so we don't modify the original
        nodes_to_del = [] #Top: nodes marked for deletion
        elimable = [i for i in G.nodes if G.nodes[i]['elimable']] #Top: list of internal nodes
        for n in elimable: #Top: for each internal node
            players_with_uncertainties = set([G.edges[n, v]['player'] for (u, v) in G.edges([n])]) #Top: make a set of players who have one or more outgoing edges from this node
            if player in players_with_uncertainties: #Top: delete nodes where the announce player doesn't know (as he/she announced 'I know')
                nodes_to_del.append(n)
        for n in nodes_to_del: #Top: delete each to-be-deleted node with probability update_prob
            if np.random.random_sample() < elim_prob:
                G.remove_node(n)
        return G

    #Top: Given a possible world model, an update_prob, an announcement (True = "I know", False = "I don't know"), and a player number, return the updated model after that announcement is made by that player
    def update_model(self, graph, elim_prob, announcement, player_number):
        '''
        Update the model according to a player's announcement
        '''
        if announcement: #Top: the announcement is "I know"
            G = self.update_model_pos(graph, elim_prob, player_number)
        else: #Top: the announcement is "I don't know"
            G = self.update_model_neg(graph, elim_prob, player_number)
        return G
