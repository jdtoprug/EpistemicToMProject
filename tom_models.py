import itertools as it  # For permutations
import networkx as nx  # For graphs
import numpy as np

'''
Describes a game and the full perfect model for that game

Note: PerfectModel cannot change!
'''
class PerfectModel:
    def __init__(self, newpcount=0, newhandsize=0, newsymbs="", newvis=None,newconsnum=False):
        if newvis is None:
            newvis = [[]]
        self.playercount = newpcount  # The number of players in this game
        self.handsize = newhandsize  # The number of cards/hats each player has
        self.symbols = newsymbs  # A list of all available hats/cards
        self.visibilities = newvis  # List with for each player a, a list with for each player b, a boolean whether a
        # can see b's hand
        self.handpossibs = []  # List of possible hands
        self.possible_worlds = []  # List with possible worlds (list of Strings)
        self.fmodel = None  # Graph object with full model
        self.edges = []  # List with for each player, that player's edges
        self.consnum = newconsnum

        self.untouchables = []  # Nodes that can't be removed
        self.ends = False  # True if nodes that can't be removed exist

        # Allow "full", "noself", and "facewall" to be inputted
        if isinstance(self.visibilities, str):
            if self.visibilities == "full":
                self.gen_full_visibs()
            else:
                if self.visibilities == "facewall":
                    self.gen_facewall_visibs()
                else:
                    if self.visibilities == "noself":
                        self.gen_noself_visibs()
                    else:
                        assert False, "Visibility algorithm does not exist!"
        self.generate_pw_model(self.consnum)

    def gen_full_visibs(self):
        visibs = []
        for i in range(self.playercount):
            visibs.append([True] * self.playercount)
        self.visibilities = visibs

    '''
    Generate visibility list (list of lists of bools) for games where players can't see themselves, given a player 
    count
    '''

    def gen_noself_visibs(self):
        self.gen_full_visibs()
        for i in range(self.playercount):
            for j in range(self.playercount):
                if i == j:  # You can't see yourself
                    self.visibilities[i][j] = False

    '''
    Generate visibility list for players that are in a row and can only see the symbols of players in front of them,
    given player count. If there are n players, player 0 can see all players except her-/himself, and player n can 
    see noone
    '''

    def gen_facewall_visibs(self):
        self.gen_full_visibs()
        for i in range(self.playercount):
            for j in range(self.playercount):
                if j <= i:  # You can't see yourself or players behind you
                    self.visibilities[i][j] = False

    '''
            Generate the full Kripke possible world model given as object variables, the number of players, 
            a list of available symbols,
            the hand size for each player, and which players each player can see
            '''

    def generate_pw_model(self, consnum=False):
        cardsingame = self.playercount * self.handsize  # Number of cards in the game

        '''
        First, we generate the list of possible worlds for this game
        '''
        symbperms = [list(symbtup[:cardsingame]) for symbtup in list(set(it.permutations(self.symbols)))]
        # it.permutations returns a list of
        # tuples of all permutations of the input list, assuming distinct objects. Since we do not, we have to
        # convert to set and back to list to remove duplicates. Then we restrict a game state to the cards held by
        # players (so no set-aside cards), and convert the tuples to lists

        # At this point, symbperms is a list of lists of characters, where each sublist is a possible world with the
        # same length as playercount * handsize, where the symbols are a subset of self.symbols.

        # Remove isometric possible worlds
        if self.handsize > 1:  # Only needed if players have more than one symbols, e.g. given two symbols A8 a hand A8
            # is isometric to 8A
            newsymbperms = []  # Replacement list to be constructed
            for symblist in symbperms:  # Turn each sublist of characters (possible world) into a sublist of subsublists
                # of characters where the subsublist is a single player's hand
                newsymblist = []  # Possible world to be constructed, list of lists of characters, where each sublist
                # is a hand
                for i in range(0, len(symblist), self.handsize):  # Loop over hands (step size is hand size)
                    tempsublist = symblist[i:i + self.handsize]  # List of characters which is the player's hand
                    tempsublist.sort()  # Sort each player's hand so isometric hands are in the same order
                    temphand = self.listtostring(tempsublist)  # We want to keep track of possible hands
                    if temphand not in self.handpossibs:
                        self.handpossibs.append(temphand)
                    newsymblist.append(tempsublist)
                # Now we have newsymblist, a list (possible world) of lists (hands) of symbols
                newsymblist2 = []
                for ahand in newsymblist:  # Convert back from list of lists to list
                    for symb in ahand:
                        newsymblist2.append(symb)
                if newsymblist2 not in newsymbperms:  # Only add worlds that are not isometric to existing worlds
                    newsymbperms.append(newsymblist2)
            symbperms = newsymbperms
        else:
            for char in self.symbols:  # We want a list of all possible hands
                if char not in self.handpossibs:
                    self.handpossibs.append(char)

        # Convert each of the sublists of characters to string (each of which is a possible world)
        for i in range(len(symbperms)):  # For each possible world
            symbperms[i] = self.listtostring(symbperms[i])  # Replace list of characters with string
        symbperms.sort()  # Sort the possible worlds. At this point we finished generating the list of possible worlds
        # for this game, which is a list of strings

        # No we remove the duplicate possible worlds
        nodups_symbperms = []
        for perm in symbperms:
            if perm not in nodups_symbperms:
                nodups_symbperms.append(perm)
        self.possible_worlds = nodups_symbperms  # Store the possible worlds model in the object

        if consnum and self.handsize == 1 and self.playercount == 2:
            newpossibworlds = []
            max = 0
            for w in self.possible_worlds:
                num1 = int(w[0:1])
                num2 = int(w[1:2])
                if num1 > max:
                    max = num1
                if num2 > max:
                    max = num2
                dif = num1 - num2
                if dif == 1 or dif == -1:
                    newpossibworlds.append(w)
            for w in newpossibworlds:
                if str(max) in w:
                    self.untouchables.append(w)
                    self.ends = True
            self.possible_worlds = newpossibworlds

        '''
        Now we have to create the edges in the possible world model. Recall that we have
        self.possible_worlds - A list of strings with each string a possible world
        self.playercount - The number of players in the game
        self.handsize - The number of symbols each player has
        self.symbols - All symbols that exist in the game
        self.visibilities - For each player a, a list of bools with for each player b, whether player a can see 
        player b's symbols
        '''
        alledges = []  # List with for each player a list of edges (tuples) between possible worlds (ints)
        for p1 in range(self.playercount):  # Loop over all players 1
            p1vis = self.visibilities[p1]  # Get player 1's visibility list
            p1edges = []  # List of edges for player 1
            for pw1index in range(len(self.possible_worlds)):  # Loop over possible worlds 1
                pw1 = self.possible_worlds[pw1index]  # We want to loop over indices to construct the edges,
                # but we need the possible world itself as well
                visiblepw1 = ""  # The portion of this possible world 1 that is visible to player 1. Could be empty!
                for p2 in range(self.playercount):  # Loop over all players 2
                    if p1vis[p2]:  # If player 1 can see this player
                        visiblepw1 += pw1[p2 * self.handsize:(p2 * self.handsize) + self.handsize]  # Add this part of
                        # the possible world
                for pw2index in range(len(self.possible_worlds)):  # Loop over possible worlds 2
                    pw2 = self.possible_worlds[pw2index]  # Same as above
                    if pw1 < pw2:  # One-directional non-reflexive edges
                        visiblepw2 = ""  # The portion of this possible world 2 that is visible to player 1. Could be
                        # empty!
                        for p2 in range(self.playercount):  # Same as above
                            if p1vis[p2]:
                                visiblepw2 += pw2[p2 * self.handsize:(p2 * self.handsize) + self.handsize]
                        if visiblepw1 == visiblepw2:  # If two possible worlds are indistinguishable
                            newedge = (pw1index, pw2index)  # Give this player an edge between those worlds
                            if newedge not in alledges:  # No duplicate edges
                                p1edges.append(newedge)
            alledges.append(p1edges)  # A list of edges for each player
        self.edges = alledges

        '''
        Now we make the actual graph object (basically the same as Cedegao et al, 2021
        '''
        pwmodel = nx.Graph()  # Empty undirected graph

        # Add possible worlds to the graph
        pwids = np.arange(0, len(self.possible_worlds), 1)  # List [0,1,2,...,#possible worlds]
        for i in range(len(self.possible_worlds)):  # Add each of the nodes with identifiers 0, 1, 2, etc
            pwmodel.add_node(pwids[i], state=self.possible_worlds[i])  # State is a string with each player's hand

        # Add edges to the graph

        # First, create a list of unique edges
        uniqueedges = []  # List of unique edges to be constructed
        for p in range(self.playercount):  # Loop over players
            for e in range(len(self.edges[p])):  # Loop over each player's edges
                if self.edges[p][e] not in uniqueedges:
                    uniqueedges.append(self.edges[p][e])  # An edge is a 2-tuple of ints

        # Then, we need to create an accompanying list showing which players access which edges
        edgeplayers = []  # List with for each unique edge, a list of players that has access to it
        for i in range(len(uniqueedges)):  # Loop over edges - for each unique edge, create a list of players that can
            # access it
            edge = uniqueedges[i]  # Current edge
            edgeplayers.append([])  # Start constructing list of players that can access this edge
            for p in range(self.playercount):  # Loop over each player's
                for e in range(len(self.edges[p])):  # Edges
                    if edge == self.edges[p][e]:  # If the player has access to this edge
                        edgeplayers[i].append(p)  # Add that player to the edge access list

        # Now we can create the edges: they consist of a 2-tuple of ints (possible worlds), and a list of ints (
        # players with access to this edge)
        for i in range(len(uniqueedges)):  # Now we add the edges to the model
            if len(edgeplayers[i]) > 0:
                pwmodel.add_edge(uniqueedges[i][0], uniqueedges[i][1], players=edgeplayers[i])

        # Store the constructed graph
        self.fmodel = pwmodel

    '''
    List of characters or strings to string
    '''

    @staticmethod
    def listtostring(inlist):
        outstr = ""
        for substr in inlist:  # Loop over characters in list and append to string
            outstr = outstr + substr
        return outstr

    '''
    Converts an undirected model to a directed model, and draws reflexive edges for each player at each node
    '''
    def fullmodel_to_directed_reflexive(self):
        dipmodel = self.fmodel.to_directed()  # Turn pmodel into a directed graph
        newedges = []
        playernames = list(range(self.playercount))  # List of player names [0, 1, 2, 3, ...
        for p in range(len(self.edges)):   # Loop over players
            newedges.append([])
            for (a,b) in self.edges[p]:  # Loop over that player's edges
                if (a,b) not in newedges[p]:
                    newedges[p].append((a,b))
                if (b,a) not in newedges[p]:
                    newedges[p].append((b,a))
        dipedges = dipmodel.edges(data="players")  # List of edges of full graph, with list of players for each edge
        dipnodes = dipmodel.nodes(data="state")  # List of nodes of full graph, with list of states for each node
        #self.playercount
        nodenums = [x[0] for x in dipnodes]
        for nodenum in nodenums:  # Loop over nodes to check each node for reflexive edges
            for p in range(len(self.edges)):
                if (nodenum, nodenum) not in self.edges[p]:
                    newedges[p].append((nodenum, nodenum))
            if (nodenum,nodenum) in dipmodel.edges.keys():
                for p in playernames:
                    if p not in dipmodel.edges[nodenum, nodenum]:
                        dipmodel.edges[nodenum, nodenum].append(p)
            else:
                dipmodel.add_edge(nodenum, nodenum, players=playernames)
        self.edges = newedges
        self.fmodel = dipmodel

    '''
    Given a player order (list of ints), a possible world model, the actual true world (string), and whether 
    announcements are simultaneous (bool), returns a list with for each round, for each turn, whether that player 
    should answer 'True' or 'False'
    '''

    def perfectanswers(self, playerorder, actualstate, simultaneous):
        assert len(actualstate) == len(self.fmodel.nodes(data="state")[0]), actualstate + " is of length " + str(
            len(actualstate)) + " whereas model states are of length " + str(len(self.fmodel.nodes(data="state")[0]))
        rnd = 0  # Round number for iteration
        equibstate = 0  # The first round where the responses stay the same (equilibrium in responses)
        answerlist = [[]]  # Output list (list of lists (players) of bools (their answers))
        someoneremovednodes = True  # Stop iterating over rounds if noone removed any nodes - we have reached an
        # equilibrium state
        newmodel = self.fmodel.copy()  # Don't want to modify the original
        while someoneremovednodes:  # Keep playing rounds until the possible world models no longer change
            someoneremovednodes = False  # Assume no changes to the model until proven otherwise
            if simultaneous:  # If announcements are simultaneous, collect all player's announcements and update with
                # them all at once
                playerlist = []  # List of players
                knowslist = []  # For each player, whether that player announces 'know' or 'don't know'
            for player in playerorder:  # Loop over turns. Always play a full round even if an equilibrium state is
                # reached before the end of the round
                know, impossible, answer = self.perfectanswer(player, self.playerlook(player, actualstate),
                                                              newmodel)  # Get the correct answer for this round,
                # for this turn/player (ignore the actual answer - we assume players can ONLY announce 'I know'/'I
                # don't know' #MODIFIABLE)
                if impossible:  # Someone detects a lie/mistake - for now we assume we stop playing #MODIFIABLE
                    return answerlist
                if not simultaneous:
                    newmodel, thisplayerremovednodes = self.updatemodel_perfect(player, know, newmodel)  # Update the
                    # model based on the player's correct announcement (if announcements are not simultaneous)
                    if thisplayerremovednodes:
                        someoneremovednodes = True  # If a player removes nodes during a round, play another round
                else:
                    playerlist.append(player)
                    knowslist.append(know)
                answerlist[rnd].append(know)  # Always add player's announcement to output list
            if simultaneous:
                newmodel, someoneremovednodes = self.updatemodel_perfect_simultaneous(playerlist, knowslist, newmodel)
            if rnd > 0:  # Check for equilibrium state
                if answerlist[rnd] != answerlist[equibstate]:  # Current state is not same as supposed equilibrium state
                    equibstate = rnd  # Make the current state the equilibrium state - it will stay the same until a
                    # difference is found
            if someoneremovednodes:  # Get ready for the next round
                answerlist.append([])
            rnd += 1  # Increment round
        return answerlist, equibstate

    '''
    Given a player (int), what the player sees (string), and a possible world model, returns whether the player 
    knows 
    his/her hand or not (bool), whether things are impossible (bool, someone lied or made a mistake), 
    and the answer 
    (string)
    '''

    def perfectanswer(self, player, observed, model):
        newmodel = model.copy()  # Don't want to modify the input
        correctnodes = [x for x in newmodel.nodes(data="state") if observed == self.playerlook(player, x[1])]  # Nodes
        # that correspond to the observed state

        ownhandpossibs = set(
            [x[1][player * self.handsize:player * self.handsize + self.handsize] for x in correctnodes])
        # Unique possibilities for your own hand in the nodes that correspond to the observed state
        if len(ownhandpossibs) > 1:  # There are two or more possibilities for your own hand - you don't know your hand
            return False, False, ""
        if len(ownhandpossibs) == 0:  # There are no possibilities for your own hand! Something went wrong!
            return False, True, ""
        if len(ownhandpossibs) == 1:  # There is exactly one possibility for your own hand - you know your hand
            return True, False, correctnodes[0][1][player * self.handsize:player * self.handsize + self.handsize]

    '''
    Given a player (int) and the actual state (string), returns what that player sees (string, could be empty)
    '''

    def playerlook(self, player, curstate):
        look = ""  # Start output empty
        for p2 in range(len(self.visibilities[player])):  # For each player that this player sees
            if self.visibilities[player][p2]:
                look += curstate[
                        p2 * self.handsize:p2 * self.handsize + self.handsize]  # Add that player's symbols to the  #
                # output
        return look

    '''
    Update on announcement
    Given a possible world model, a player number, and whether the player announces 'I know'(True) or not (
    False), returns the possible world model which adheres to the announcement, as well as whether an update 
    happened,
    assuming a single announcement
    '''

    def updatemodel_perfect(self, player, know, model):
        newmodel = model.copy()  # We don't want to modify the original
        removenodes, nodesremoved = self.updatemodel_perfect_nodes(player, know, model)  # Get nodes that should
        # be removed

        # Actually remove the nodes that should be removed
        for node in model.nodes(data="state"):  # Loop over nodes
            if node[0] in removenodes:
                newmodel.remove_node(node[0])  # Remove it
        return newmodel, nodesremoved

    '''
    Update on announcement, mostly taken from cadegao et al, 2021
    Given a possible world model, a player number, and whether the player announces 'I know'(True) or not (
    False), returns the nodes to be removed from the possible world model which adheres to the announcement, 
    as well as 
    whether an update happened
    '''

    def updatemodel_perfect_nodes(self, player, know, model):
        nodesremoved = False  # Whether any nodes were removed in this turn - needed to recognize equilibrium states
        newmodel = model.copy()  # We don't want to modify the original

        # First we make a list of nodes where the announcing player DOES NOT know his/her symbols (regardless of the
        # announcement)
        uncertainnodes = []  # Nodes where the player isn't certain about his/her HAND
        for node in newmodel.nodes(data="state"):  # Loop over all nodes (and check for uncertainty)
            owncards = node[1][
                       player * self.handsize:player * self.handsize + self.handsize]  # The player's own symbols in
            # this state
            certain = True  # Assume no alternatives for the player's symbols until proven otherwise
            for edge in [[e[0], e[1]] for e in newmodel.edges(data="players") if player in e[2]]:  # Loop over all the
                # player's edges...
                if node[0] in edge:  # ...connected to the node we are checking for uncertainty
                    for node2 in edge:  # Loop over nodes this edge connects to (not optimal)
                        if node[0] != node2:  # Don't compare node to itself
                            if owncards != newmodel.nodes(data="state")[node2][
                                           player * self.handsize:player * self.handsize + self.handsize]:  # If nodes
                                # are connected by player edge and have different player symbols, there is uncertainty
                                certain = False
            if not certain:  # ...and if there is uncertainty, add this node to the list with uncertainties
                uncertainnodes.append(node)

        # Now that we have a list of nodes where the announcing player is uncertain about his/her hand,
        # we can create a list of nodes that should be removed
        removenodes = []
        for node in model.nodes(data="state"):  # Loop over nodes
            if know == (node in uncertainnodes):  # If the announcement is "I know", then remove nodes WITH
                # uncertainty, if the announcement is "I don't know", then remove nodes WITHOUT uncertainty
                nodesremoved = True  # We need to return whether we removed nodes
                removenodes.append(node[0])
        return removenodes, nodesremoved

class MuddyModel:
    """
    Initialization function: always called when object of this class is created
    """

    def __init__(self, newpcount=0, newhandsize=0, newsymbs="", newvis=None, newpws=None, newedges=None, newmodel=None,
                 newhandpossibs=None, newinteriornodes=None, newinteriorstates=None, newstatenodes=None,
                 newstatedict=None, newlevel = 0):
        # Mutable default values cause weird issues so we initialize Nones and then replace those.
        if newvis is None:
            newvis = [[]]
        if newpws is None:
            newpws = []
        if newedges is None:
            newedges = [[]]
        if newhandpossibs is None:
            newhandpossibs = []
        if newinteriornodes == None:
            newinteriornodes = []
        if newinteriorstates == None:
            newinteriorstates = []
        if newstatenodes == None:
            newstatenodes = []
        if newstatedict == None:
            newstatedict = {}
        assert newpcount * newhandsize <= len(newsymbs), "Not enough symbols for all players!"
        for pvis in newvis:
            for player in pvis:
                assert player < newpcount, "Player " + str(player) + " in vislist does not exist!"
        self.possible_worlds = newpws  # List of possible worlds where each possible world is a string
        self.edges = newedges  # List with, for each player, a list of tuples of ints. Each int is the unique
        # identifier of
        # a possible world, and a tuple is an edge indicating the player cannot distinguish these worlds.
        self.playercount = newpcount  # The number of players in this game
        self.handsize = newhandsize  # The number of cards/hats each player has
        self.symbols = newsymbs  # A list of all available hats/cards
        self.visibilities = newvis  # For each player a for each player b, True if player a can observe player b's
        # symbols, False if not
        self.model = newmodel  # Full possible worlds model
        self.handpossibs = newhandpossibs  # All possible hands for all players
        self.interiornodes = newinteriornodes
        self.interiorstates = newinteriorstates
        self.statenodes = newstatenodes
        self.statedict = newstatedict
        self.level = newlevel

    '''
    Generate full visibility list given a player count (every player can see every other player), returns list of 
    lists of Trues
    '''

    def gen_full_visibs(self):
        visibs = []
        for i in range(self.playercount):
            visibs.append([True] * self.playercount)
        self.visibilities = visibs
        return visibs

    '''
    Generate visibility list (list of lists of bools) for games where players can't see themselves, given a player 
    count
    '''

    def gen_noself_visibs(self):
        visibs = self.gen_full_visibs()
        for i in range(self.playercount):
            for j in range(self.playercount):
                if i == j:  # You can't see yourself
                    visibs[i][j] = False
        return visibs

    '''
    Generate the full Kripke possible world model given as object variables, the number of players, 
    a list of available symbols,
    the hand size for each player, and which players each player can see
    '''

    def generate_pw_model(self):
        cardsingame = self.playercount * self.handsize  # Number of cards in the game

        '''
        First, we generate the list of possible worlds for this game
        '''
        symbperms = [list(symbtup[:cardsingame]) for symbtup in list(set(it.permutations(self.symbols)))]
        # it.permutations returns a list of
        # tuples of all permutations of the input list, assuming distinct objects. Since we do not, we have to
        # convert to set and back to list to remove duplicates. Then we restrict a game state to the cards held by
        # players (so no set-aside cards), and convert the tuples to lists

        # At this point, symbperms is a list of lists of characters, where each sublist is a possible world with the
        # same length as playercount * handsize, where the symbols are a subset of self.symbols.

        # Remove isometric possible worlds
        if self.handsize > 1:  # Only needed if players have more than one symbols, e.g. given two symbols A8 a hand A8
            # is isometric to 8A
            newsymbperms = []  # Replacement list to be constructed
            for symblist in symbperms:  # Turn each sublist of characters (possible world) into a sublist of subsublists
                # of characters where the subsublist is a single player's hand
                newsymblist = []  # Possible world to be constructed, list of lists of characters, where each sublist
                # is a hand
                for i in range(0, len(symblist), self.handsize):  # Loop over hands (step size is hand size)
                    tempsublist = symblist[i:i + self.handsize]  # List of characters which is the player's hand
                    tempsublist.sort()  # Sort each player's hand so isometric hands are in the same order
                    temphand = self.listtostring(tempsublist)  # We want to keep track of possible hands
                    if temphand not in self.handpossibs:
                        self.handpossibs.append(temphand)
                    newsymblist.append(tempsublist)
                # Now we have newsymblist, a list (possible world) of lists (hands) of symbols
                newsymblist2 = []
                for ahand in newsymblist:  # Convert back from list of lists to list
                    for symb in ahand:
                        newsymblist2.append(symb)
                if newsymblist2 not in newsymbperms:  # Only add worlds that are not isometric to existing worlds
                    newsymbperms.append(newsymblist2)
            symbperms = newsymbperms
        else:
            for char in self.symbols:  # We want a list of all possible hands
                if char not in self.handpossibs:
                    self.handpossibs.append(char)

        # Convert each of the sublists of characters to string (each of which is a possible world)
        for i in range(len(symbperms)):  # For each possible world
            symbperms[i] = self.listtostring(symbperms[i])  # Replace list of characters with string
        symbperms.sort()  # Sort the possible worlds. At this point we finished generating the list of possible worlds
        # for this game, which is a list of strings

        # No we remove the duplicate possible worlds
        nodups_symbperms = []
        for perm in symbperms:
            if perm not in nodups_symbperms:
                nodups_symbperms.append(perm)
        self.possible_worlds = nodups_symbperms  # Store the possible worlds model in the object

        '''
        Now we have to create the edges in the possible world model. Recall that we have
        self.possible_worlds - A list of strings with each string a possible world
        self.playercount - The number of players in the game
        self.handsize - The number of symbols each player has
        self.symbols - All symbols that exist in the game
        self.visibilities - For each player a, a list of bools with for each player b, whether player a can see 
        player b's symbols
        '''
        alledges = []  # List with for each player a list of edges (tuples) between possible worlds (ints)
        for p1 in range(self.playercount):  # Loop over all players 1
            p1vis = self.visibilities[p1]  # Get player 1's visibility list
            p1edges = []  # List of edges for player 1
            for pw1index in range(len(self.possible_worlds)):  # Loop over possible worlds 1
                pw1 = self.possible_worlds[pw1index]  # We want to loop over indices to construct the edges,
                # but we need the possible world itself as well
                visiblepw1 = ""  # The portion of this possible world 1 that is visible to player 1. Could be empty!
                for p2 in range(self.playercount):  # Loop over all players 2
                    if p1vis[p2]:  # If player 1 can see this player
                        visiblepw1 += pw1[p2 * self.handsize:(
                                                                     p2 * self.handsize) + self.handsize]  # Add  #
                        # this part  # of the possible world
                for pw2index in range(len(self.possible_worlds)):  # Loop over possible worlds 2
                    pw2 = self.possible_worlds[pw2index]  # Same as above
                    if pw1 < pw2:  # One-directional non-reflexive edges
                        visiblepw2 = ""  # The portion of this possible world 2 that is visible to player 1. Could be
                        # empty!
                        for p2 in range(self.playercount):  # Same as above
                            if p1vis[p2]:
                                visiblepw2 += pw2[p2 * self.handsize:(p2 * self.handsize) + self.handsize]
                        if visiblepw1 == visiblepw2:  # If two possible worlds are indistinguishable
                            newedge = (pw1index, pw2index)  # Give this player an edge between those worlds
                            if newedge not in alledges:  # No duplicate edges
                                p1edges.append(newedge)
            alledges.append(p1edges)  # A list of edges for each player
        self.edges = alledges

        '''
        Now we make the actual graph object (basically the same as Cedegao et al, 2021
        '''
        pwmodel = nx.Graph()  # Empty undirected graph

        # Add possible worlds to the graph
        pwids = np.arange(0, len(self.possible_worlds), 1)  # List [0,1,2,...,#possible worlds]
        for i in range(len(self.possible_worlds)):  # Add each of the nodes with identifiers 0, 1, 2, etc
            pwmodel.add_node(pwids[i], state=self.possible_worlds[i])  # State is a string with each player's hand

        # Add edges to the graph

        # First, create a list of unique edges
        uniqueedges = []  # List of unique edges to be constructed
        for p in range(self.playercount):  # Loop over players
            for e in range(len(self.edges[p])):  # Loop over each player's edges
                if self.edges[p][e] not in uniqueedges:
                    uniqueedges.append(self.edges[p][e])  # An edge is a 2-tuple of ints

        # Then, we need to create an accompanying list showing which players access which edges
        edgeplayers = []  # List with for each unique edge, a list of players that has access to it
        for i in range(len(uniqueedges)):  # Loop over edges - for each unique edge, create a list of players that can
            # access it
            edge = uniqueedges[i]  # Current edge
            edgeplayers.append([])  # Start constructing list of players that can access this edge
            for p in range(self.playercount):  # Loop over each player's
                for e in range(len(self.edges[p])):  # Edges
                    if edge == self.edges[p][e]:  # If the player has access to this edge
                        edgeplayers[i].append(p)  # Add that player to the edge access list

        # Now we can create the edges: they consist of a 2-tuple of ints (possible worlds), and a list of ints (
        # players with access to this edge)
        for i in range(len(uniqueedges)):  # Now we add the edges to the model
            if len(edgeplayers[i]) > 0:
                pwmodel.add_edge(uniqueedges[i][0], uniqueedges[i][1], players=edgeplayers[i])

        # Store the constructed graph and return it
        self.model = pwmodel
        return pwmodel

    '''
    Given a player order (list of ints), a possible world model, the actual true world (string), and whether 
    announcements are simultaneous (bool), returns a list with for each round, for each turn, whether that player 
    should answer 'True' or 'False'
    '''

    def perfectanswers(self, playerorder, actualstate, pwmodel, simultaneous):
        assert len(actualstate) == len(pwmodel.nodes(data="state")[0]), actualstate + " is of length " + str(
            len(actualstate)) + " whereas model states are of length " + str(len(pwmodel.nodes(data="state")[0]))
        rnd = 0  # Round number for iteration
        equibstate = 0  # The first round where the responses stay the same (equilibrium in responses)
        answerlist = [[]]  # Output list (list of lists (players) of bools (their answers))
        someoneremovednodes = True  # Stop iterating over rounds if noone removed any nodes - we have reached an
        # equilibrium state
        newmodel = pwmodel.copy()  # Don't want to modify the original
        while someoneremovednodes:  # Keep playing rounds until the possible world models no longer change
            someoneremovednodes = False  # Assume no changes to the model until proven otherwise
            if simultaneous:  # If announcements are simultaneous, collect all player's announcements and update with
                # them all at once
                playerlist = []  # List of players
                knowslist = []  # For each player, whether that player announces 'know' or 'don't know'
            for player in playerorder:  # Loop over turns. Always play a full round even if an equilibrium state is
                # reached before the end of the round
                know, impossible, answer = self.perfectanswer(player, self.playerlook(player, actualstate),
                                                              newmodel)  # Get the correct answer for this round,
                # for this turn/player (ignore the actual answer - we assume players can ONLY announce 'I know'/'I
                # don't know' #MODIFIABLE)
                if impossible:  # Someone detects a lie/mistake - for now we assume we stop playing #MODIFIABLE
                    return answerlist
                if not simultaneous:
                    newmodel, thisplayerremovednodes = self.updatemodel_perfect(player, know, newmodel)  # Update the
                    # model based on the player's correct announcement (if announcements are not simultaneous)
                    if thisplayerremovednodes:
                        someoneremovednodes = True  # If a player removes nodes during a round, play another round
                else:
                    playerlist.append(player)
                    knowslist.append(know)
                answerlist[rnd].append(know)  # Always add player's announcement to output list
            if simultaneous:
                newmodel, someoneremovednodes = self.updatemodel_perfect_simultaneous(playerlist, knowslist, newmodel)
            if rnd > 0:  # Check for equilibrium state
                if answerlist[rnd] != answerlist[equibstate]:  # Current state is not same as supposed equilibrium state
                    equibstate = rnd  # Make the current state the equilibrium state - it will stay the same until a
                    # difference is found
            if someoneremovednodes:  # Get ready for the next round
                answerlist.append([])
            rnd += 1  # Increment round
        return answerlist, equibstate

    '''
    Given a player (int), what the player sees (string), and a possible world model, returns whether the player knows 
    his/her hand or not (bool), whether things are impossible (bool, someone lied or made a mistake), and the answer 
    (string)
    '''

    def perfectanswer(self, player, observed, model):
        newmodel = model.copy()  # Don't want to modify the input
        correctnodes = [x for x in newmodel.nodes(data="state") if observed == self.playerlook(player, x[1])]  # Nodes
        # that correspond to the observed state

        ownhandpossibs = set(
            [x[1][player * self.handsize:player * self.handsize + self.handsize] for x in correctnodes])
        # Unique possibilities for your own hand in the nodes that correspond to the observed state
        if len(ownhandpossibs) > 1:  # There are two or more possibilities for your own hand - you don't know your hand
            return False, False, ""
        if len(ownhandpossibs) == 0:  # There are no possibilities for your own hand! Something went wrong!
            return False, True, ""
        if len(ownhandpossibs) == 1:  # There is exactly one possibility for your own hand - you know your hand
            return True, False, correctnodes[0][1][player * self.handsize:player * self.handsize + self.handsize]

    '''
    Given a player (int) and the actual state (string), returns what that player sees (string, could be empty)
    '''

    def playerlook(self, player, curstate):
        look = ""  # Start output empty
        for p2 in range(len(self.visibilities[player])):  # For each player that this player sees
            if self.visibilities[player][p2]:
                look += curstate[
                        p2 * self.handsize:p2 * self.handsize + self.handsize]  # Add that player's symbols to the  #
                # output
        return look

    '''
    Update on announcement
    Given a possible world model, a player number, and whether the player announces 'I know'(True) or not (
    False), returns the possible world model which adheres to the announcement, as well as whether an update happened,
    assuming a single announcement
    '''

    def updatemodel_perfect(self, player, know, model):
        newmodel = model.copy()  # We don't want to modify the original
        removenodes, nodesremoved = self.updatemodel_perfect_nodes(player, know, model)  # Get nodes that should
        # be removed

        # Actually remove the nodes that should be removed
        for node in model.nodes(data="state"):  # Loop over nodes
            if node[0] in removenodes:
                newmodel.remove_node(node[0])  # Remove it
        return newmodel, nodesremoved

    '''
    Update on announcement, mostly taken from cadegao et al, 2021
    Given a possible world model, a player number, and whether the player announces 'I know'(True) or not (
    False), returns the nodes to be removed from the possible world model which adheres to the announcement, as well as 
    whether an update happened
    '''

    def updatemodel_perfect_nodes(self, player, know, model):
        nodesremoved = False  # Whether any nodes were removed in this turn - needed to recognize equilibrium states
        newmodel = model.copy()  # We don't want to modify the original

        # First we make a list of nodes where the announcing player DOES NOT know his/her symbols (regardless of the
        # announcement)
        uncertainnodes = []  # Nodes where the player isn't certain about his/her HAND
        for node in newmodel.nodes(data="state"):  # Loop over all nodes (and check for uncertainty)
            owncards = node[1][
                       player * self.handsize:player * self.handsize + self.handsize]  # The player's own symbols in
            # this state
            certain = True  # Assume no alternatives for the player's symbols until proven otherwise
            for edge in [[e[0], e[1]] for e in newmodel.edges(data="players") if player in e[2]]:  # Loop over all the
                # player's edges...
                if node[0] in edge:  # ...connected to the node we are checking for uncertainty
                    for node2 in edge:  # Loop over nodes this edge connects to (not optimal)
                        if node[0] != node2:  # Don't compare node to itself
                            if owncards != newmodel.nodes(data="state")[node2][
                                           player * self.handsize:player * self.handsize + self.handsize]:  # If nodes
                                # are connected by player edge and have different player symbols, there is uncertainty
                                certain = False
            if not certain:  # ...and if there is uncertainty, add this node to the list with uncertainties
                uncertainnodes.append(node)

        # Now that we have a list of nodes where the announcing player is uncertain about his/her hand,
        # we can create a list of nodes that should be removed
        removenodes = []
        for node in model.nodes(data="state"):  # Loop over nodes
            if know == (node in uncertainnodes):  # If the announcement is "I know", then remove nodes WITH
                # uncertainty, if the announcement is "I don't know", then remove nodes WITHOUT uncertainty
                nodesremoved = True  # We need to return whether we removed nodes
                removenodes.append(node[0])
        return removenodes, nodesremoved

    '''
    Mirror hands in possible world
    '''

    @staticmethod
    def mirrorhands(instr, hndsize):
        outstr = ""
        for i in range(int(len(instr) / hndsize)):
            substr = instr[i * hndsize:i * hndsize + hndsize]
            substr = substr[::-1]
            outstr = outstr + substr
        return outstr

    '''
    List of characters or strings to string
    '''

    @staticmethod
    def listtostring(inlist):
        outstr = ""
        for substr in inlist:  # Loop over characters in list and append to string
            outstr = outstr + substr
        return outstr

'''
ToM model where each note is annotated with, for each player, each ToM level
'''
class ToMsModel:
    '''
    Initialization function
    '''
    def __init__(self, newpmodel, newmaxtom=0, newfull=None):
        self.pmodel = newpmodel  # PerfectModel object. self.pmodel.fmodel must be a DiGraph where each edge has variable players (list of player
        # numbers), and each node has variable state (String). No edges must be omitted, even reflexive!
        self.ends = self.pmodel.ends
        self.untouchns = []
        for w in self.pmodel.untouchables:
            self.untouchns.append(self.statetonode(w))
        self.maxtom = newmaxtom  # Highest ToM level the model has, int
        self.full = newfull  # Whether the full graph should be used. None if yes, otherwise a list [s,p,t]
        self.nodetomlist = []  # For each node, a list with for each player, a list with for each ToM-level, whether it still exists (True) or not (False)
        self.actualstate = ""  # Actual state's name
        self.player = -1  # Graph owner
        self.level = -1  # Graph owner's ToM level
        self.actualn = -1  # Actual state's id
        self.initialstates = []  # Initial states
        self.generate_nodetomlist()  # Fills nodetomlist appropriately

    '''
    Fills nodetomlist variable
    '''
    def generate_nodetomlist(self):
        nodelist = list(self.pmodel.fmodel.nodes())  # Grab all nodes from the full model
        nodelist.sort()  # Sort nodes by number
        nodelist = nodelist[::-1]  # Reverse so highest number is in front
        highestnode = nodelist[0]  # Grab the highest number
        highestp = self.pmodel.playercount - 1  # Highest player

        if not self.full is None:
            startbool = False
        else:
            startbool = True

        # All tuples exist unless if specified otherwise
        for n in range(highestnode + 1):
            self.nodetomlist.append([])
            for p in range(highestp + 1):
                self.nodetomlist[n].append([])
                for t in range(self.maxtom + 1):
                    self.nodetomlist[n][p].append(startbool)

        if not self.full is None:
            self.actualstate = self.full[0]
            edgelist = self.pmodel.fmodel.edges(data = "players")

            n = self.statetonode(self.full[0])
            self.actualn = n
            p = self.full[1]
            t = self.full[2]
            self.player = p
            self.level = t
            nexttuples = []
            prevtuples = [[n, p, t]]
            havetuples = [[n,p,t]]
            for i in range(t):
                prevtuples.append([n, p, i])
                havetuples.append([n,p,i])
                self.initialstates.append(n)
            # Grow graph outwards
            while len(prevtuples) != 0:

                for [n,p,t] in prevtuples:  # Loop over tuples
                    outedges = [(u,v,l) for (u,v,l) in edgelist if u == n]  # Get outgoing edges from n
                    for (u,v,l) in outedges:  # Loop over edges
                        for edgeplayer in l:  # Loop over players
                            if edgeplayer == p and n != v and [v,p,t] not in havetuples:
                                havetuples.append([v,p,t])
                                nexttuples.append([v,p,t])
                            else:
                                if t-1 >= 0 and [v,edgeplayer,t-1] not in havetuples:
                                    havetuples.append([v, edgeplayer, t-1])
                                    nexttuples.append([v, edgeplayer, t-1])
                prevtuples = nexttuples
                nexttuples = []
            for [n,p,t] in havetuples:
                self.nodetomlist[n][p][t] = True
    '''
    Return all answers up to a certain ToM
    input:
    -node - node number of the actual state
    -player - player id
    -maxtom - Maximum ToM level under consideration
    -lowknow - lower your ToM when you can't find answers?
    '''
    def allanswerlist(self, node, player, maxtom, lowknow=False):
        anss = []
        for i in range(maxtom + 1):
            anss.append(self.answerlist(node, player, i, lowknow=lowknow))
        return anss

    '''
    Takes a node name and return the number
    '''

    def statetonode(self, s):
        return [i for (i,state) in self.pmodel.fmodel.nodes(data="state") if s == state][0]

    '''
    Whether player player answers 'I know my cards' (True) or 'I don't know my cards' (False) if node is the actual node and the player has ToM level tom
    lowknow - if you don't know, but you know in a lower level, switch to lower level? True if yes.

    As output a tuple bool, list, where list is the list of possibilities
    If you answer 'I don't know', you still return a list of possibilities
    '''

    def answerlist(self, node, player, tom, lowknow=False):
        hndsz = self.pmodel.handsize
        statenodes = self.pmodel.fmodel.nodes(
            data="state")  # List of 2-tuples where the first element is the node id and the second element is the node's state as string
        '''
        if not self.nodetomlist[node][player][tom]:  # Tuple doesn't exist!
            print("Tuple doesn't exist!")
            done = False  # Stop looping when true
            while not done:
                tom = tom - 1  # Lower ToM until you find one that still exists
                if tom == -1:
                    return -1, ''
                else:
                    if self.nodetomlist[node][player][tom]:  # Tuple exists!
                        done = True
           # return -1, ''
        '''
        # else:
        nodepossibs = [v for (u, v, l) in
                       self.pmodel.fmodel.edges(data="players") if
                       u == node and player in l]  # Nodes announcing player considers possible in this node
        actualhandpossibs = []  # Only nodes with edge to <player,tom>
        for n in nodepossibs:
            if self.nodetomlist[n][player][tom]:  # If tuple exists
                actualhandpossibs.append(statenodes[n][player * hndsz: (player * hndsz) + hndsz])
        actualhandpossibs = set(actualhandpossibs)  # Remove duplicates
        numpossibs = len(actualhandpossibs)  # Number of possibilities for your own hand
        if numpossibs == 1:  # You know your hand
            actualhandpossibs = list(actualhandpossibs)
            return 1, actualhandpossibs
        else:
            if numpossibs == 0:  # No outgoing edges
                if not lowknow:
                    actualhandpossibs = list(actualhandpossibs)
                    return -1, actualhandpossibs
                else:
                    print("Lower ToM until positive answer is found (1)")
                    done = False  # Stop condition
                    while not done:
                        tom = tom - 1
                        print("Testing tom " + str(tom))
                        if tom == -1:
                            actualhandpossibs = list(actualhandpossibs)
                            return -1, actualhandpossibs
                        actualhandpossibs = []  # Only nodes with edge to <player,tom>
                        for n in nodepossibs:
                            print("Checking node: " + str(self.nodetostate(n)))
                            if self.nodetomlist[n][player][tom]:  # If tuple exists
                                actualhandpossibs.append(statenodes[n][player * hndsz: (player * hndsz) + hndsz])
                        actualhandpossibs = set(actualhandpossibs)
                        numpossibs = len(actualhandpossibs)
                        if numpossibs == 1:  # Found a positive answer!
                            actualhandpossibs = list(actualhandpossibs)
                            return 1, actualhandpossibs
            else:  # 2 or more possibilities, you don't know your hand
                if not lowknow:
                    actualhandpossibs = list(actualhandpossibs)
                    actualhandpossibs.sort()
                    return 0, actualhandpossibs  # Say 'I don't know', but also return what you think is still possible, for model fitting
                else:  # Lower your ToM until you find a positive answer
                    print("Lower ToM until positive answer is found (2)")
                    done = False  # Stop condition
                    while not done:
                        tom = tom - 1
                        print("Testing tom " + str(tom))
                        if tom == -1:
                            actualhandpossibs = list(actualhandpossibs)
                            actualhandpossibs.sort()
                            return 0, actualhandpossibs
                        actualhandpossibs = []  # Only nodes with edge to <player,tom>
                        for n in nodepossibs:
                            print("Checking node: " + str(self.nodetostate(n)))
                            if self.nodetomlist[n][player][tom]:  # If tuple exists
                                print(statenodes[n][player * hndsz: (player * hndsz) + hndsz])
                                actualhandpossibs.append(statenodes[n][player * hndsz: (player * hndsz) + hndsz])
                        actualhandpossibs = set(actualhandpossibs)
                        numpossibs = len(actualhandpossibs)
                        if numpossibs == 1:  # Found a positive answer!
                            actualhandpossibs = list(actualhandpossibs)
                            return 1, actualhandpossibs

    '''
        Update model on announcement

        knows - whether the announcing player knows (True is `I know my cards', False is `I don't know my cards')
        player - player number of announcing player
        reflexivetom - whether reflexive arrows are counted as ToM
        delonempty - Remove tuples with ToM-1 or higher if there are no outgoing edges for the announcement
        confbias - If true, do not remove initial nodes if you know your hand
        lowknow - Assume the announcer, if announcing true, may have lowered their ToM to obtain a true instead of false answer
        '''

    def update(self, knows, player, reflexivetom = True, ans="", delonempty = False, confbias = False, lowknow = False):
        lowknow = False
        newnodetomlist = []
        setfalselist = []
        statenodes = self.pmodel.fmodel.nodes(data = "state")
        hndsz = self.pmodel.handsize
        for n in range(len(self.nodetomlist)):  # Loop over nodes
            newnodetomlist.append([])
            nodepossibs = [v for (u,v,l) in
                                     self.pmodel.fmodel.edges(data = "players") if u == n and player in l]  # Nodes announcing player considers possible in this node
            for p in range(len(self.nodetomlist[n])):  # Loop over players
                newnodetomlist[n].append([])
                if lowknow:
                    truelowerexists = False
                for t in range(len(self.nodetomlist[n][p])):  # Loop over ToM levels
                    value = self.nodetomlist[n][p][t]  # True if it still exists
                    newnodetomlist[n][p].append(value)
                    if value:  # If it exists
                        newtom = t  # Don't change ToM when considering own edges
                        if player != p:  # If node doesn't belong to announcing player, subtract 1 ToM level
                            newtom = newtom - 1
                        actualnodepossibs = []
                        for n2 in nodepossibs:
                            if reflexivetom:
                                if newtom >= 0 and self.nodetomlist[n2][player][newtom]:
                                    actualnodepossibs.append(n2)  # Only consider nodes that still have the tuple
                            else:  # Reflexive doesn't count
                                if player == p:  # No changes on own announcement
                                    if newtom >= 0 and self.nodetomlist[n2][player][newtom]:
                                        actualnodepossibs.append(n2)
                                else:
                                    if n != n2:  # No changes for non-reflexive
                                        if newtom >= 0 and self.nodetomlist[n2][player][newtom]:
                                            actualnodepossibs.append(n2)
                                    else:  # Reflexive!
                                        if newtom >= 0 and self.nodetomlist[n2][player][newtom+1]:
                                            actualnodepossibs.append(n2)
                        handpossibs = []  # Hands player considers possible
                        for n2 in actualnodepossibs:
                            nstate = statenodes[n2]
                            pcards = nstate[player * hndsz: (player * hndsz) + hndsz]
                            handpossibs.append(pcards)
                        handpossibs = set(handpossibs)

                        numpossibs = len(handpossibs)  # Number of unique hand possibilities announcing player has

                        if numpossibs != 0:  # There are outgoing edges
                            if knows:  # If announcement is 'I know', remove node if it has uncertainty or if it does
                                # not correspond to announcement
                                if numpossibs > 1:
                                    if not lowknow:
                                        setfalselist.append([n,p,t])
                                    else:
                                        if not truelowerexists or player != p:
                                            setfalselist.append([n, p, t])
                                else:  # 1 possibility
                                    if ans != "" and numpossibs == 1:  # Announcer gives an answer
                                        if ans != list(handpossibs)[0]:  # Answer does not correspond to node
                                            if not confbias:
                                                setfalselist.append([n, p, t])
                                            else:  # Confirmation bias
                                                if n not in self.initialstates:  # Only remove if you don't know
                                                    setfalselist.append([n, p, t])
                                        else:
                                            if lowknow and player == p:
                                                truelowerexists = True
                                    else:  # No answer
                                        if lowknow and player == p:
                                            if numpossibs == 1:
                                                truelowerexists = True
                            else:  # If announcement is 'I don't know', remove node if it has NO uncertainty
                                if numpossibs == 1:
                                    if not confbias:
                                        setfalselist.append([n, p, t])
                                    else:
                                        if n not in self.initialstates:  # Only remove if you don't know
                                            setfalselist.append([n, p, t])
                        else:  # No outgoing edges
                            if delonempty and t > 0:  # Delete with no outgoing edges if ToM > 0 and variable is set
                                setfalselist.append([n, p, t])
        for [n,p,t] in setfalselist:
            if not self.ends:
                self.nodetomlist[n][p][t] = False
            else:
                if n not in self.untouchns:
                    self.nodetomlist[n][p][t] = False
    '''
    Creates a string for a node
    For example
    0: 0, 1, 2 (agent 0 can have ToM 0, 1 or 2 in this node)
    1: 0, 1, 2 (agent 1 can have ToM 0, 1 or 2 in this node)
    '''
    def nodestring(self,nodenum):
        tomlist = self.nodetomlist[nodenum]
        outstr = ""
        for p in range(self.pmodel.playercount):
            outstr = outstr + str(p) + ": "
            for t in range(self.maxtom + 1):
                if tomlist[p][t]:
                    outstr = outstr + str(t) + ", "
                else:
                    outstr = outstr + (" "*len(str(t))) + "  "
            outstr = outstr[:-2]
            outstr = outstr + "\n"
        outstr = outstr[:-1]
        return outstr