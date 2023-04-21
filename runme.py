# Imports

import timeit
from tom_models import *
from epistemic_structures import *
from agents import *
import csv
import math
import scipy
import matplotlib.pyplot as plt  # Drawing graphs
import os

'''
Read log-likelihoods per player, per ToM from file
input:
-filenamestart - Initial substring of the filename of the to-be-read file
output:
-outlist - The to-be-read .csv file, converted to list of lists of Python objects
'''


def readloglist(filenamestart='tom_refTrue_delonFalse_'):
    name = filenamestart + "likelihoods.csv"  # Construct filename
    outlist = []  # To-be-returned list
    # Read the file and store it in outlist
    with open(name, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            outlist.append(row)
    file.close()
    for i in range(1, len(outlist)):  # Loop over lists in outlist
        for j in range(len(outlist[i])):  # Loop over items in sublist
            if j != 6:  # Column 6 can stay a string
                if isinstance(outlist[i][j], str):  # Otherwise, if it is a string
                    if outlist[i][j] != "":  # And it is non-empty
                        outlist[i][j] = eval(outlist[i][j])  # Evaluate it
                    else:
                        outlist[i][j] = -1
    return outlist

'''
Calculate coherence for random model, taking into account answer
Input:
-k88 - Times answered `I know my cards, I have two Eights'
-k8a - Times answered `I know my cards, I have one Eight and one Ace'
-kaa - Times answered `I know my cards, I have two Aces'
-nk - Times answered `I do not know my cards'

Output:
-l - Proportion of times random model should correspond to participant
'''

def calccorrectraterandom(k88, k8a, kaa, nk):
    n = k88 + k8a + kaa + nk  # Total number of answers given
    l = ((k88 * (k88 / n)) + (k8a * (k8a / n)) + (kaa * (kaa / n)) + (nk * (nk / n))) / n  # Calculate coherence
    return l

'''
Calculate likelihood for random model, taking into account answer
Input:
-k88 - Times answered `I know my cards, I have two Eights'
-k8a - Times answered `I know my cards, I have one Eight and one Ace'
-kaa - Times answered `I know my cards, I have two Aces'
-nk - Times answered `I do not know my cards'

Output:
-l - log-likelihood of random model corresponding to participant
'''
def calcerrorlikelihoodans(k88, k8a, kaa, nk):
    l = 0  # To-be-outputted log-likelihood
    n = k88 + k8a + kaa + nk  # Calculate n, the total number of decision points
    for a in [nk, k88, k8a, kaa]:  # Loop over answers
        if a > 0:  # Prevent ln(0)
            l = l + (a * math.log(a / n))  # Sum over answers of a*ln(a/n)
    return l

'''
For a player, count how many times the player gives each answer
Input:
-predictlist - database with predictions for each ToM level, for each decision point of each participant
-player - player id
-perfect - If true, pretend each player gave perfect answers

Output:
-k88 - Times answered `I know my cards, I have two Eights'
-k8a - Times answered `I know my cards, I have one Eight and one Ace'
-kaa - Times answered `I know my cards, I have two Aces'
-nk - Times answered `I do not know my cards'
'''

def countknowans(predictlist, player, perfect=False):
    playerlist = []  # List of all rows in predictlist that belong to this player
    ansindex = 7  # In which column can we find the player's answer?
    if perfect:  # If we pretend the players gave perfect answers...
        ansindex = 8  # ...look at the column with the perfect answer instead
    for list in predictlist:  # Loop through all decision points for all players
        if list[0] == player:  # If it belongs to the current player
            playerlist.append(list)  # Add it to the list of the current player's decision points

    # Count how often this player gave each answer
    k88 = 0
    k8a = 0
    kaa = 0
    nk = 0
    for list in playerlist:
        if list[ansindex][0] == 0:
            nk += 1
        else:
            if list[ansindex][1][0] == '88':  # If the anser was `I know my cards', also check which cards were reported
                k88 += 1
            else:
                if list[ansindex][1][0] == '8A''':
                    k8a += 1
                else:
                    kaa += 1
    return k88, k8a, kaa, nk

'''
Calculate log-likelihood
Input:
-correct - Times model corresponded to participant
-incorrect - Times model deviated from participant
-penalty - p in n(1-e)*ln(1-e) + ne*ln(pe)

Output:
-l - log-likehood of participant using this model
'''

def calclikelihood(correct, incorrect, penalty=0.5):
    e = incorrect / (correct + incorrect)  # epsilon, error rate - incoherent answers divided by n
    l = 0
    if correct > 0:
        l += correct * math.log(1 - e)  # n(1-e)*ln(1-e)
    if incorrect > 0:
        l += incorrect * math.log(e * penalty)  # ne*ln(pe)
    return l

'''
For a player and a ToM, count how many correct predictions the model makes, as well as incorrect predictions

Input:
-predictlist - For each decision point, predicted answers from each model
-player - player ID
-tom - model's level we want to know accuracy for
-perfect - Whether to pretend the participant gave perfect answers
-emptyincorrect - If False, treat no outgoing edges as `I don't know my cards', if True, treat no outgoing edges as an answer that does not correspond to the participant

Output:
-correct - Times model corresponded to the participant
-incorrect - Times model deviated from the participant
'''

def countmodelacc(predictlist, player, tom, perfect=False, emptyincorrect=False):
    playerlist = []  # List of player's decision points with predicted model answers
    ansindex = 7  # Column with the player's answer
    if perfect:  # If we pretend the participant answered perfectly, use the column with the perfect answer instead
        ansindex = 8
    for list in predictlist:  # Loop over all players, all decision points
        if list[0] == player:  # Get those decision points for the current player
            playerlist.append(list)

    # Count how often model corresponded to player (and how often it did not)
    correct = 0
    incorrect = 0
    for list in playerlist:  # Loop over player's decision points
        if not emptyincorrect:
            if list[6][tom][0] == -1:  # Replace 'impossible' with 'I don't know'
                list[6][tom] = (0, list[6][tom][1])
        if list[6][tom][0] != list[ansindex][0]:  # Predict True when say False and vice versa
            incorrect += 1
        else:
            if list[6][tom][0] == 0:
                correct += 1
            else:
                if list[6][tom][1] == list[ansindex][1]:
                    correct += 1
                else:
                    incorrect += 1

        # list[6] is predictions
        # list[7] is actual
        # list[8] is perfect
    return correct, incorrect

'''
For a player and ToM, count how often that ToM is correct
Input:
-predictlist - For each player's decision point, each level's predicted answer
-player - Player ID
-tom - Level under consideration

Output:
-totcor / totrounds - Proportion of decision points where this level gives the correct answer
'''

def counttcor(predictlist, player, tom):
    totrounds = 0  # Total number of decision points
    totcor = 0  # Decision points where this model is correct
    for list in predictlist:  # Loop over all players' decision points
        if list[0] == player:  # If it belongs to the current player
            totrounds += 1  # Increment number of decision points
            if list[10][tom]:  # This level's answer is correct, increment
                totcor += 1
    return totcor / totrounds  # Return proportion of correct answers for this level

'''
For a player, count how often that player is correct
Input:
-predictlist - For all players' decision points, predictions for each model
-player - Player ID

Output:
-totcar / totrounds - Proportion of decision points where player gave correct answer
'''

def countpcor(predictlist, player):
    totrounds = 0  # Total number of player's decision points
    totcor = 0  # Number of decision points where player was correct
    for list in predictlist:  # Loop over all decision points
        if list[0] == player:  # If it belongs to the player
            totrounds += 1  # Increment rounds
            if list[9]:  # If player was correct, increment
                totcor += 1
    return totcor / totrounds  # Return proportion that player was correct

'''
Read CSV with predictions for each ToM level, for each subject/turn/round
Input:
-name - Name of the file to be read

Output:
-List with for each player, for each decision point, list of predicted answers for each model
'''

def csvtopredictions(name='tompredictions.csv'):
    outlist = []  # List to be outputted

    # Read file
    with open(name, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        for row in reader:
            outlist.append(row)
    file.close()

    # Convert everything from string to Python objects
    for i in range(len(outlist)):
        if i != 0 and i < 3761:  # Only subject data
            for j in [0,1,3,4]:
                outlist[i][j] = int(outlist[i][j])
            for j in [5,6,7,8,9,10]:
                if isinstance(outlist[i][j], str):
                    outlist[i][j] = eval(outlist[i][j])
        else:
            if i > 3760:
                if isinstance(outlist[i][1], str):
                    outlist[i][1] = eval(outlist[i][1])
    return outlist

'''
Returns predicted answers vs. actual answers for a subject
input:
-subj - subject id
-maxtom - maximum ToM under consideration
-pack - a list [lib,cdat] for faster calculations
-verbose - if True, also return state, round, and turn for each decision
'''

def subjpairs(subj, maxtom, pack=None, verbose=False):
    if pack is None:  # If these haven't been passed, make them now
        lib = gamelibrary(maxtom)  # For each combination of state/turn/round/ToM, a predicted answer
        cdat = prunecedegaodata_game(False)  # Cedegao's data
    else:
        lib = pack[0]
        cdat = pack[1]
    outlist = []
    for row in cdat[subj - 1]:  # Loop over subject's played games
        state = row[2]
        turn = row[3]
        anss = row[4]  # List of answers for each round
        for round in range(len(anss)):  # Loop over rounds
            ans = anss[round]
            lookupstr = str(state) + str(round) + str(turn)  # Create a key to use with lib
            predictlist = lib[lookupstr]  # Predicted answers for each ToM level
            know = 1
            cardans = ""
            if ans == "False":
                know = 0
            else:
                cardans = ans[4:6]  # 88, 8A, or AA
            totans = (know, [cardans])
            if not verbose:
                outlist.append((predictlist, totans))
            else:
                outlist.append(((state, round, turn), predictlist, totans))
    return outlist

'''
    Read Cedegao's participant data

    THIS DATA DESCRIPTION COPIED FROM Cedegao et al (2021)'S README:

    ## Data
    Columns:
    1. "age": the subject's age.              
    2. "AmyResponse": Amy's response (I know/I don't know).         
    3. "answer": the subject's answer (what cards she holds. If response is "I don't know", encoded as '').   
    4. "BenResponse": Ben's response (I know/I don't know).         
    5. "cards": game state.               
    6. "cards_iso": game states up to equivalence, e.g., 88AAAA is encoded as AA8888.          
    7. "corAnswer": the subject's correct response (I know or I don't know).           
    8. "exp_time": how long the subject took to finish the experiment, excluding the demographic form.               
    9. "gender": the subject's gender.                         
    10. "inference_level": the minimum level l required to guarantee a SUWEB model (with no stochasticity) with 
        level >= l can solve the game.
    11. "number": the number guess in the p-beauty contest.                  
    12. "order": game order.               
    13. "outcome": won or lost the current game.          
    14. "outcomeArray": the correct annoucements by all players of a game (a list of lists), representing rounds in 
        a game.        
    15. "phase": game number.                  
    16. "points": how many games the subject won out of ten games.              
    17. "response": the subject's actual response (I know/I don't know).          
    18. "round": the round number of a data point.                
    19. "RT": the subject's reaction time for each round (counting from the onset of the previous announcement).  
    20. "should_know": at which turn should the subject know her card (9 turns total, 10 means she never should know).
    21. "subj": the subject ID.       
    22. "numRound": the maximum number of round the game reaches if the subject keeps responding I don't know.

    ACTUAL data structure:
    1. phase - game number
    2. cards - game state, where the first two cards are the participant's, the second two cards are Amy's, 
    the third two cards are Ben's
    3. order - Turn order each round
    4. outcomeArray - Perfect answers for this situation
    5. response - True if the subject says 'I know', False if the subject says 'I don't know'
    6. answer - AA/88/A8, what the subject says he/she has ('' if response is False).
    7. AmyResponse - what Amy said in this round ('' if game ended before Amy got a turn) (True/False)
    8. BenResponse - what Ben said in this round ('' if game ended before Ben got a turn) (True/False)
    9. corAnswer - what the subject should respond (True, False)
    10. round - round number
    11. subj - subject ID
    12. outcome - Did the subject win or lose (0 if lost, 1 if won)
    13. cards_iso - Equivalent game state that starts with A (same if the game state already starts with A)
    14. RT - reaction time
    15. points - how many games subject won out of 10 games (where you win if you make NO mistakes)
    16. exp_time - time to finish entire experiment
    17. number - number guess in p-beauty
    18. inference_level - epistemic level required to solve this case
    19. should_know - At which turn out of 3x3=9 should the subject know his/her cards (10 if never)
    20. numRound - maximum round game reaches if subject keeps responding "I don't know"
    21. age - subject age
    22. gender - subject sex

    ['4', 'A8A8AA', 'Amy,Ben,You', '[[False, False, False], [True, False, True]]', 'False', '', 'False', 'False', 
        'False', '1', '1', '0', 'A8A8AA', '14.664', '5', '27.6', '25', '4', '6', '2', '29', '0']
    ['4', 'A8A8AA', 'Amy,Ben,You', '[[False, False, False], [True, False, True]]', 'False', '', 'True', 'False', 
        'True', '2', '1', '0', 'A8A8AA', '2.14', '5', '27.6', '25', '4', '6', '2', '29', '0']
    '''


def readcedegaodata():
    # First, we need to convert the .csv to usable data types
    toint = [0, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21]  # Convert to int
    tofloat = [13, 15]  # Convert to float
    tobool = [4, 6, 7, 8]  # Conver to boal
    toboollist = [3]  # Convert to list of list of bools
    toeval = toint + tofloat + tobool + toboollist  # Everything we need to convert
    with open('data.csv', newline='') as csvfile:  # Open the .csv
        datareader = csv.reader(csvfile, delimiter=',', quotechar='"')  # Create a .csv reader
        datalist = []  # We need to convert from csv reader to list, as list has indices
        for row in datareader:
            datalist.append(row.copy())
        for i in range(len(datalist)):  # Loop over data and convert each
            if i != 0:  # Don't convert header
                for j in toeval:
                    if datalist[i][j] == "":  # This happens if Amy/Ben didn't get a chance to respond in a round
                        datalist[i][j] = None
                    else:
                        if datalist[i][j] == "NA":  # Convert NA values to -1
                            datalist[i][j] = -1
                        else:
                            datalist[i][j] = eval(datalist[i][j])  # Evaluate everything else as Python code
                datalist[i][2] = datalist[i][2].split(
                    ",")  # Convert string 'Ben,You,Amy' to list ['Ben', 'You', 'Amy']
        return datalist

'''
    Reads cedegao's data and removes columns we are not interested in

        ACTUAL data structure:
    1. phase - game number
    2. cards - game state, where the first two cards are the participant's, the second two cards are Amy's, 
    the third two cards are Ben's
    3. order - Turn order each round
    4. outcomeArray - Perfect answers for this situation
    5. response - True if the subject says 'I know', False if the subject says 'I don't know'
    6. answer - AA/88/A8, what the subject says he/she has ('' if response is False).
    7. AmyResponse - what Amy said in this round ('' if game ended before Amy got a turn) (True/False)
    8. BenResponse - what Ben said in this round ('' if game ended before Ben got a turn) (True/False)
    9. corAnswer - what the subject should respond (True, False)
    10. round - round number
    11. subj - subject ID
    12. outcome - Did the subject win or lose (0 if lost, 1 if won)
    13. cards_iso - Equivalent game state that starts with A (same if the game state already starts with A)
    14. RT - reaction time
    15. points - how many games subject won out of 10 games (where you win if you make NO mistakes)
    16. exp_time - time to finish entire experiment
    17. number - number guess in p-beauty
    18. inference_level - epistemic level required to solve this case
    19. should_know - At which turn out of 3x3=9 should the subject know his/her cards (10 if never)
    20. numRound - maximum round game reaches if subject keeps responding "I don't know"
    21. age - subject age
    22. gender - subject sex

    input:
    -extra: True to return extra data (game, round, turn)
    '''


def prunecedegaodata(extra):
    data = readcedegaodata()
    for i in range(len(data)):
        row = data[i]
        temprow = [row[1], row[2], row[4], row[5], row[9], row[10]]  # Only rows we want
        temprow[0] = temprow[0][0:2][::-1] + temprow[0][2:4][::-1] + temprow[0][4:6][::-1]  # Flip hands

        # Cedegao's states are ordered YYAABB (You, Amy Ben), whereas mine are ordered 001122 (first, second,
        # third player in turn order)
        turnorderedgame = ""
        for player in row[2]:
            if player == "You":
                turnorderedgame += temprow[0][0:2]
            else:
                if player == "Amy":
                    turnorderedgame += temprow[0][2:4]
                else:
                    if player == "Ben":
                        turnorderedgame += temprow[0][4:6]

        # Turn answer into single string
        if temprow[2]:
            answer = "True" + str(temprow[3])[::-1]
        else:
            answer = "False"
        # Figure out which turn (0,1,2) the player was playing
        turn = -1
        if temprow[1][0] == 'You':
            turn = 0
        else:
            if temprow[1][1] == 'You':
                turn = 1
            else:
                if temprow[1][2] == 'You':
                    turn = 2
        round = -1
        if i > 0:
            round = temprow[4] - 1
            game = turnorderedgame + "rnd" + str(round) + "trn" + str(turn)  # Make game string
        else:  # Make sure header makes sense
            game = "game"
            answer = "answer"

        if extra:
            data[i] = [temprow[5], game, answer, turnorderedgame, round, turn]
        else:
            data[i] = [temprow[5], game, answer]
    # ['True88', 3, '8A8A8Arnd2trn2', 0.9, 0.1]
    return data[1:]

'''
   Same as prunecedegaodata, but instead of one row per game round it returns one row for all rounds of each game
   '''


def prunecedegaodata_game(removefirstrow):
    cedat = prunecedegaodata(True)

    newdat = []
    games = []
    # First make a list with a row for each combination of player/state/turn (but NOT round)
    for datrow in cedat:
        game = datrow[1]
        fullgame = str(datrow[0]) + "s" + game[:6] + game[10:]  # Remove 'rnd#' from string, add player ID
        if fullgame not in games:
            games.append(fullgame)
            newdat.append([fullgame, datrow[0], datrow[3], datrow[5], []])  # Unique ID, player ID,
            # state, turn
        datrow.append(fullgame)
    # Now loop through the data and make a list of answers for each player in each game
    for datrow in cedat:
        for newrow in newdat:
            if newrow[0] == datrow[-1]:
                newrow[4].append(datrow[2])
    # Group participant answers
    newerdat = [[]]
    curid = newdat[0][1]
    for datrow in newdat:
        if datrow[1] == curid:
            newerdat[-1].append(datrow)
        else:
            curid = datrow[1]
            newerdat.append([])
            newerdat[-1].append(datrow)
    if removefirstrow:
        for i in range(len(newerdat)):
            for j in range(len(newerdat[i])):
                newerdat[i][j] = newerdat[i][j][1:]
    return newerdat

'''
    Returns answers for aces and eights for all combinations of true states, player turn locations, player turn (first, 
    middle, or last), answering round, assuming error = 0 and update_prob = 1, and all previous answers were perfect,
    using Cedegao et al 2021's original method.

    input:
    -repetitions - number of times each case is repeated
    -error: error in Cedegao's model
    output:
    -totaloutput - a list with 1) the state of these answers, 2) the epistemic level of the agents for these answers, 
    3) the repetition number of this repetition, and 4) a list with for each round, a list with for each turn, 
    the player's answer. Each answer is a 3-tuple with whether the player knows (bool), whether the player's graph is 
    empty (bool), and the actual answer (string)

     This function is used to verify whether my implementations generate the same answers as Cedegao et al's
    '''

def allacesandeights_anyerror_and_update_cedegao(repetitions, error):
    verbose = False  # Set to True to output intermediate steps to console for debugging
    totaloutput = []  # Our to-be-returned list
    a8mm = MuddyModel(3, 2, "AAAA8888")  # The game
    a8mm.gen_noself_visibs()  # Players can't see themselves
    perfecta8model = a8mm.generate_pw_model()  # Full and perfect possible world model for this game
    allstates = a8mm.possible_worlds  # List of all states
    # allstates = ["8A888A"] # REMOVE - Used as a single case for debugging
    a8mmc = bounded_modal_model()  # Initialize a bounded modal
    for state in allstates:  # Loop over all states
        if verbose:  # For debugging
            print(state)
        perfectanswers, _ = a8mm.perfectanswers([0, 1, 2], state, perfecta8model, False)  #
        # Perfect answers for this state
        statec = a8mm.mirrorhands(state, 2)  # Mirror all hands - my method uses e.g. 8A whereas Cedegao et al's
        # uses A8
        if verbose:  # For debugging
            print("statec: " + str(statec))
            print("perfectanswers: " + str(perfectanswers))
        statecs = [statec, statec[2:4] + statec[0:2] + statec[4:6], statec[4:6] + statec[0:2] + statec[2:4]]  #
        # Cedegao et al uses the convention that the first two cards in a state are the
        # player's the second two are Amy's, and the third two are Ben's, whereas we order hands by player order
        # (first two belong to the player first in turn order, et cetera)
        if verbose:
            print("statecs: " + str(statecs))
        for level in range(0, (5 + 1)):  # Loop over levels #[4]:# REMOVE - replace with single level for debugging
            for rep in range(repetitions):  # Loop over repititions
                outputanswers = []  # List of answers an agent of this level should make in this position
                # Create restricted models for all three players
                restrictedmodels = [None, None, None]
                othercards = [None, None, None]
                # orders = [['You','Amy','Ben'],['Amy','You','Ben'],['Amy','Ben', 'You']]
                if verbose:
                    print("Amy: " + statec[2:4] + ", Ben: " + statec[4:6])
                restrictedmodels[0] = a8mmc.generate_partial_model(statec[2:4], statec[4:6], level=level)
                othercards[0] = [statec[2:4], statec[4:6], statec[0:2]]  # Amy, Ben, self
                restrictedmodels[1] = a8mmc.generate_partial_model(statec[0:2], statec[4:6], level=level)
                othercards[1] = [statec[0:2], statec[4:6], statec[2:4]]
                restrictedmodels[2] = a8mmc.generate_partial_model(statec[0:2], statec[2:4], level=level)
                othercards[2] = [statec[0:2], statec[2:4], statec[4:6]]
                if verbose:  # For debugging
                    print("othercards: " + str(othercards))
                for rnd in range(len(perfectanswers)):  # Loop over rounds
                    outputanswers.append([])  # Answers for this round
                    if verbose:  # For debugging
                        print("Current round answers: " + str(perfectanswers[rnd]))
                    for player in range(len(perfectanswers[rnd])):  # Loop over answers/players
                        if verbose:  # For debugging
                            print("")
                            print("Current player: " + str(player))
                            print("statecs[" + str(player) + "]: " + str(statecs[player]))
                            print("Amy_cards: " + str(othercards[player][0]))
                            print("Ben_cards: " + str(othercards[player][1]))
                        answer = SUWEB.agent_by_game_single_game(level=level, update_prob=1, noise=error,
                                                                 state=statecs[player],
                                                                 Amy_cards=othercards[player][0],
                                                                 Ben_cards=othercards[player][1],
                                                                 G=restrictedmodels[player])  # Get the answer
                        # for this player
                        outputanswers[rnd].append((answer[0], answer[1], answer[2]))  # Add it to the output
                        if verbose:  # For debugging
                            print("answer:" + str(answer))
                            print("")
                            print("Model updates:")
                        for i in range(len(restrictedmodels)):  # Then, update all the models
                            playerorder = [2, 3]  # In some cases in Cedegao et al (2021)'s code, order is always
                            # assumed to be You - Amy - Ben. We use the convention 'first player - second player
                            # - third player - etc', so we need to translate between them
                            playerorder.insert(i, 1)  # 1-2-3 if the player is first, 2-1-3 if the player is
                            # second, 2-3-1 if the player is third
                            if verbose:  # For debugging
                                print("playerorder " + str(playerorder))
                                print("Player " + str(i) + " acts as 1 and updates with " + str(
                                    answer[0]) + " from " + str(player) + " (acting as " + str(
                                    playerorder[player]) + ")")
                            restrictedmodels[i] = a8mmc.update_model(restrictedmodels[i],
                                                                     perfectanswers[rnd][player], playerorder[
                                                                         player])  # Update the model based on  #
                            # announced PERFECT answer, not the ACTUAL answer
                totaloutput.append([state, level, rep, outputanswers])  # Add current case to output
    return totaloutput

'''
Library of states, turns, and sequence of announcements
input:
-maxtom - Maximum ToM level under consideration
-reftom - Does reflexive count as ToM? True for yes, False for no.
-confbi - Do agents have confirmation bias?
-lono - Do agents lower their ToM to find an answer?
-usecedegao - If true, use epistemically bounded models instead of ToM models
'''


def gamelibrary(maxtom, reftom=True, delon=False, confbi=False, lono=False, usecedegao=False):
    a8pm = PerfectModel(3, 2, "8888AAAA", "noself")  # Make perfect model for Aces and Eights
    a8pm.fullmodel_to_directed_reflexive()  # Turn non-directed graph without reflexive arrows in directed graph with reflexive arrows

    if usecedegao:
        cedanss = allacesandeights_anyerror_and_update_cedegao(1, 0)  # Get answers from epistemically bounded models
        # Columns are: state, level, list of answers for each round/turn
        # Each answer is a three-tuple with
        # A bool for Know/don't know,
        # A bool for there are no outgoing edges
        # The actual answer
    statedict = {}  # Start building a dictionary with a key for each combination of state, round, turn, and as values a list of predicted answers for each ToM

    statelist = a8pm.possible_worlds  # List of all states in the game (as strings)
    for state in statelist:  # Loop over states
        tsm = ToMsModel(a8pm, maxtom)  # New ToM model
        panss, _ = a8pm.perfectanswers([0, 1, 2], state, False)  # Perfect answers
        for round in range(len(panss)):  # Loop over rounds
            for turn in range(len(panss[round])):  # Loop over turns
                statedict[str(state) + str(round) + str(turn)] = tsm.allanswerlist(tsm.statetonode(state), turn, maxtom,
                                                                                   lowknow=lono)  # Save answers
                tsm.update(panss[round][turn], turn, reflexivetom=reftom, delonempty=delon, confbias=confbi,
                           lowknow=lono)  # Update the model
                if usecedegao:
                    statecdans = [x[3] for x in cedanss if state == x[0]]  # Predicted answers for each EL for this combination of state/round/turn

                    # Loop over EL levels and convert them to our writing convention (two-tuple, first element is 0 for `I don't know', 1 for `I know', second element is string with answer
                    anslist = []
                    for i in range(len(statecdans)):
                        canswer = statecdans[i][round][turn]
                        if canswer[1] == True:
                            outans = (-1, [])
                        else:
                            if canswer[0] == False:
                                outans = (0, [''])
                            else:
                                outans = (1, [canswer[2]])
                        anslist.append(outans)
                    statedict[str(state) + str(round) + str(turn)] = anslist

    return statedict

'''
Returns predicted answers vs. actual answers for all subjects
input:
-maxtom - maximum ToM under consideration
-verbose - if False, output list only contains subject id, predicted answers for each ToM, and actual answer
    if True, also contains state, turn, and round for each answer
'''


def allsubjpairs(maxtom, verbose=False, reftom=True, delon=False, confbi=False, lono=False, usecedegao=False):
    lib = gamelibrary(maxtom, reftom=reftom, delon=delon, confbi=confbi, lono=lono,
                      usecedegao=usecedegao)  # Get predicted answers for all states, turns, rounds
    cdat = prunecedegaodata_game(False)  # Prune cedegao data
    pack = [lib, cdat]  # Pack the above together to pass to subjpairs
    outlist = []  # output list to construct
    # Cedegao has subjects 1 through 211
    for i in range(211 + 1):  # Loop over all subjects
        if i != 0:  # 0 is not a subject
            subjlist = subjpairs(i, maxtom, pack=pack, verbose=verbose)  # Get predicted answers for this subject
            outlist.append((i, subjlist))  # Add them to the output
    return outlist

'''
Write CSV with predictions for each ToM level, for each subject/turn/round
Input:
-name - Filename to write
-reftom - parameter, whether reflexive arrows count as ToM
-delon - parameter, whether tuples should be deleted if there are no outgoing edges
-confbi - parameter, confirmation bias. If true, do not delete tuples if you KNOW your symbols.
-lono - parameter. If true, if there are no outgoing edges, lower your ToM until you find one where there are
-maxtom - Maximum level under consideration
-usecedegao - If true, generate predictions for epistemically bounded models instead of ToM models
'''

def predictionstocsv(name='tompredictions.csv', reftom=True, delon=False, confbi=False, lono=False, maxtom=5,
                     usecedegao=False):
    outlist = allsubjpairs(maxtom, verbose=True, reftom=reftom, delon=delon, confbi=confbi, lono=lono,
                           usecedegao=usecedegao)  # List with, for each participant, decision point, and ToM, a predicted answer for that combination, as well as the participant's actual answer
    a8pm = PerfectModel(3, 2, "8888AAAA", "noself")  # Make perfect model for Aces and Eights
    a8pm.fullmodel_to_directed_reflexive()  # Turn non-directed graph without reflexive arrows in directed graph with reflexive arrows
    writelist = []  # List of rows to be written to file
    header = ['subj', 'decisionnum', 'state', 'turn', 'round', 'precedingannouncements', 'tompredictions',
              'actualanswer', 'perfectanswer', 'playercorrect', 'tomcorrects']  # Header
    writelist.append(header)
    for subj in outlist:  # Loop over subjects
        for i in range(len(subj[1])):  # Loop over decision points
            state = subj[1][i][0][0]  # Distribution of cards
            round = subj[1][i][0][1]
            turn = subj[1][i][0][2]
            panss, _ = a8pm.perfectanswers([0, 1, 2], subj[1][i][0][0], False)  # Perfect answers for this game
            perfectansbool = panss[round][turn]  # True if `I know my cards', False if `I do not know my cards'
            perfectans = (0, [''])  # Assume answer is `I do not know my cards' until shown otherwise
            if perfectansbool:  # If answer is `I know my cards'
                perfectansstring = state[turn * 2:(turn * 2) + 2]  # Get the actual, correct answer
                perfectans = (1, [perfectansstring])  # Construct perfect answer (0 = `I don't know', `1 = I know')
            panss = panss[0:subj[1][i][0][1] + 1]  # Restrict to relevant rounds
            panss[subj[1][i][0][1]] = panss[subj[1][i][0][1]][0:subj[1][i][0][2]]  # Restrict to relevant preceding turns. Now we have all the preceding announcements.

            tomcorrects = []  # Which ToM levels made the correct prediction for this subject, for this decision point?
            for pred in subj[1][i][1]:  # Loop over ToM predictions
                tomcor = True  # Assume it's correct until proven otherwise
                ans = pred[0]  # Answer (`I know' or `I don't know')
                ansstring = pred[1]  # Symbols this player claims to be holding
                if ans == -1:  # Replace no outgoing edges with `I don't know'
                    ans = 0
                if ans != perfectans[0]:  # If participant answer does not correspond to actual answer...
                    tomcor = False  # ...it's incorrect
                    if ans == 1:  # If answer is `I know', claimed symbols also need to match
                        if ansstring != perfectans[1]:
                            tomcor = False
                tomcorrects.append(tomcor)  # Add to list
            #In order: participant ID, decision point number, actual state, round, turn, preceding answers, ToM predictions, participant's answer, perfect answer, whether participant was correct, which ToM levels were correct
            subjlist = [subj[0], i + 1, subj[1][i][0][0], subj[1][i][0][2], subj[1][i][0][1], panss, subj[1][i][1],
                        subj[1][i][2], perfectans, subj[1][i][2] == perfectans, tomcorrects]

            writelist.append(subjlist)
    writelist.append(["reftom", reftom])  # Also add parameter settings to output
    writelist.append(["delon", delon])
    writelist.append(["confbi", confbi])

    # Write list of predictions to file
    with open(name, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(writelist)
    file.close()

'''
Write, to a file, log-likelihoods per player, per ToM, as well as the random model.
Input:
-maxtom - Maximum level under consideration
-filenamestart - First string in filenames for created files
-reftom - True if reflexive counts as ToM
-delon - True if tuples are deleted when there are no outgoing edges
-penalty - p in n(1-e) * ln(1-e) + ne * ln(pe) where e is error rate and n is number of decision points
-perfect - Use perfect answers instead of participant's answers
-emptyincorrect - Model always gives incorrect prediction if the graph is empty (instead of `I don't know')
-usecedegao - Set to true to use epistemically bounded model instead of ToM models
'''

def writeloglistans(maxtom=5, filenamestart='tom_refTrue_delonFalse_', reftom=True, delon=False, penalty=0.5,
                    perfect=False, emptyincorrect=False, usecedegao=False):
    predictionstocsv(name=filenamestart + 'predictions.csv', reftom=reftom, delon=delon, confbi=False, lono=False,
                     maxtom=maxtom, usecedegao=usecedegao)  # Make list of predicted answers
    predictions = csvtopredictions(filenamestart + 'predictions.csv')  # Read list predictions answers
    outlist = []  # List of rows that need to be written to a file
    outlist.append(["subjnum", "model", "loglikelihood", "reftom", "delon", "C/T", "88/8A/AA/NK", "randomcorrectrate",
                    "playeraccuracy", "tomaccuracy"])  # header
    for player in range(1, 211 + 1):  # Loop over all 211 players
        pscore = countpcor(predictions, player)  # How good is a player?
        for t in range(maxtom + 1):  # Loop over ToM levels
            tscore = counttcor(predictions, player, t)  # How good is this ToM?
            correct, incorrect = countmodelacc(predictions, player,
                                               t, perfect=perfect,
                                               emptyincorrect=emptyincorrect)  # Read how often the player corresponded to the model
            l = calclikelihood(correct, incorrect, penalty=penalty)  # Get log-likelihood
            outlist.append([player, t, l, reftom, delon, str(correct / (correct + incorrect)),
                            "", "", pscore, tscore])  # Save to output list
        k88, k8a, kaa, nk = countknowans(predictions,
                                         player,
                                         perfect=perfect)  # Once for each player, count how often player said 'I (don't) know'
        le = calcerrorlikelihoodans(k88, k8a, kaa, nk)  # Use those to calculate the likelihood of the random model
        rcr = calccorrectraterandom(k88, k8a, kaa, nk)  # Random coherence
        outlist.append([player, maxtom + 1, le, reftom, delon, rcr,
                        str(k88) + "/" + str(k8a) + "/" + str(kaa) + "/" + str(nk), rcr,
                        pscore])  # Add data to output list, reserve maxtom + 1 as random model
    with open(filenamestart + "likelihoods.csv", 'w', encoding='UTF8', newline='') as file:  # Write to file
        writer = csv.writer(file)
        writer.writerows(outlist)
    file.close()

'''
Run RFX-BMS on a model of Aces and Eights, using the data from Cedegao et al. (2021)

Input variables:
maxtom - Maximum ToM (or epistemic) level under consideration
filenamestart - Beginning of filename to save likelihoods and predictions
reftom - True if following a reflexive arrow for another player counts as a ToM step
delon - True if you should delete tuples if there are no outgoing edges
convergediff - If, between iterations, each element in alpha has changed by this value or less, stop iterating
penalty - p in n(1-e)*ln(1-e) + ne*ln(p*e) where e is error rate and n is number of decision points
perfect - If true, fit RFX-BMS on 'fake data' where all participants answered perfectly (for testing)
emptyincorrect - If true, the model's prediction is always incoherent if there are no outgoing edges. If false, the model answers `I do not know my cards' when there are no outgoing edges.
usecedegao - If true, run RFX-BMS on epistemically bounded models. If false, run RFX-BMS on ToM models
'''
def rfxbms(maxtom=5, filenamestart = 'tom_refTrue_delonFalse_', reftom = True, delon = False, convergediff=0.001, penalty=0.5, perfect = False, emptyincorrect = False, usecedegao = False):
    verbose = False  # Set to true for debugging
    writeloglistans(maxtom=maxtom, filenamestart=filenamestart, reftom=reftom, delon=delon, penalty=penalty, perfect = perfect, emptyincorrect=emptyincorrect, usecedegao = usecedegao)  # Write likelihoods
    loglist = readloglist(filenamestart=filenamestart)  # Read likelihoods. loglist is a list of lists, where each sublist contains player ID, model level, and log-likelihood of that combination, followed by several diagnostic columns
    correctratedict = {}  # Dictionary with coherence for each player-model combination (where the random model is coded as maxtom + 1)
    logdict = {}  # Dictionary of log-likelihoods for each player-model combination

    for row in loglist[1:]:  # Fill dictionary, skipping the table header
        logdict[(row[0],row[1])] = row[2]  # Key is a tuple (participant ID, level), value is log-likelihood
        correctratedict[(row[0], row[1])] = row[5]  # Same key, value is coherence

    # For each ToM, and in general, find the correct/total values
    correctratesums = [0]*(maxtom+2)  # Sum of coherence for each ToM, where ToM is best, last one is random
    correctratelists = [[] for _ in range(maxtom+2)] # List of best coherences for each ToM. DO NOT USE [[]]*(maxtom+2). Last element is over everything
    bestcounts = [0]*(maxtom+2)  # Number of players where each ToM is best. Last one is random model
    playercount = 0  # Number of players
    totalbestsum = 0  # Sum of best coherences over all players
    totalbestlist = []  # List of all best coherences
    randombestlist = []  # Coherences when random is best
    totalbestdict = {}  # For each subject, best ToMs (as list) and correct rate
    for player in range(1, 211 + 1):  # Loop over all 211 players
        bestcorrectrate = 0
        besttoms = []
        for tom in range(maxtom + 2):  # Loop over ToM levels, last one is random, to find model with best coherence for this player
            curcorrectrate = correctratedict[(player, tom)]  # Coherence for this player-model combination

            if curcorrectrate > bestcorrectrate:  # If coherence for this model is better, replace it as the best model
                bestcorrectrate = curcorrectrate
                besttoms = [tom]
            else:
                if curcorrectrate == bestcorrectrate:  # If coherence for this model is equal to the best coherence, add it to the list of best models
                    besttoms.append(tom)
        totalbestsum += bestcorrectrate  # Add best coherence to sum of all best coherences
        playercount += 1
        totalbestlist.append(bestcorrectrate)  # Add best coherence to list of all best coherences
        if maxtom + 1 in besttoms:  # If the random model fits best for this participant
            randombestlist.append(bestcorrectrate)  # Add it to the list of coherences where the random model is best
        for t in besttoms:  # For each model that is best for this participant
            bestcounts[t] += 1  # Increment counter of how many times this model is best
            correctratesums[t] += bestcorrectrate  # Increment sum of this model's total coherence
            correctratelists[t].append(bestcorrectrate)  # Add it to the list of each model's list of best coherences
        totalbestdict[player]=[bestcorrectrate,besttoms]  # Add this participant's best coherence and models to a dictionary

    outlist = []  # List of lists of coherences, to be written to file
    # Each row in the file is corresponds to a level, starting with 0, and contains all coherences where that level fits best
    # The last two rows are, first, a concatenation of all previous lists, and second, the list for the random model
    for list in correctratelists:
        outlist.append(list)
    outlist.append(totalbestlist)
    outlist.append(randombestlist)
    with open("correctrates_usecedegao" + str(usecedegao) + ".csv", 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(outlist)
    file.close()

    #Algorithm for RFX-BMS as detailed in 'Bayesian model selection for group studies' (Stephan et al., 2009)
    a0 = [1] * (maxtom + 2)  # alpha_0, with one element for levels 0 through level maxtom, and an additional element for the random model
    a = a0.copy()  # We don't change a0
    # Until convergence:
    cont = True
    while(cont):
        prev = a.copy()  # Previous alpha
        sumdg = scipy.special.digamma(sum(a))  # Digamma of sum over k of alpha_k
        if verbose:
            print("a: " + str(a))
            print("sum(a): " + str(sum(a)))
            print("scipy.special.digamma(sum(a)) (sumdg): " + str(sumdg))
        listdg = []  # For each k, Psi(alpha_k) - Psi(sum over k of alpha_k)
        for k in range(len(a)):  # For each model
            if verbose:
                print("scipy.special.digamma("+str(a[k])+") = " + str(scipy.special.digamma(a[k])))
            listdg.append(scipy.special.digamma(a[k]) - sumdg)  # Digamma of alpha_k - digamma of sum over k of alpha_k
        # Loop over models and subjects
        b = [0] * (maxtom + 2)  # For each k a beta_k
        if verbose:
            print("scipy.special.digamma(a[k]) - sumdg for each k: " + str(listdg))
            print("b before adding: " + str(b))
        for player in range(1, 211 + 1):  # Loop over all 211 players
            if verbose:
                print("player: " + str(player))
            sumunk = 0  # Sum over k of player's u_{nk}'s
            unklist = []
            for k in range(len(a)):  # Loop over models
                unk = math.exp(logdict[(player,k)] + listdg[k])  # Calculate u_{nk}
                if verbose:
                    print("model " + str(k))
                    print("listdg[k]: " + str(listdg[k]))
                    print("logdict[(player,k)]: " + str(logdict[(player,k)]))
                    print("unk: " + str(unk))
                unklist.append(unk)
                sumunk = sumunk + unk
            for k in range(len(a)):  # Loop over models again
                b[k] = b[k] + (unklist[k]/sumunk)  # Update b_k
            if verbose:
                print("unklist: " + str(unklist))
                print("sumunk: " + str(sumunk))
                print("new b: " + str(b))
        for k in range(len(a)):  # Update alpha
            a[k] = a0[k] + b[k]  # alpha = alpha_0 + beta
        if verbose:
            print("final b: " + str(b))
        cont = False

        # Check whether convergence has been achieved
        highestdiff = 0
        for k in range(len(a)):
            if abs(a[k] - prev[k]) > highestdiff:
                highestdiff = abs(a[k] - prev[k])
            if abs(a[k] - prev[k]) > convergediff:
                cont = True
        if verbose:
            input("Press enter to continue")
            print("%\n\n\n@")
    # Normalize final alpha so elements sum to 1
    suma = 0
    for ak in a:
        suma += ak
    for k in range(len(a)):
        a[k] = a[k] / suma

    print("alpha after convergence:")
    names = [x for x in range(maxtom+1)]
    names.append("random")
    print("level        estimated frequency")
    for i in range(len(a)):
        if i < len(a) - 1:
            print(str(names[i]) + "            " + str(a[i]))
        else:
            print(str(names[i]) + "       " + str(a[i]))
    print("String for copying into R:")
    print("<- c(" + str(a)[1:-1] + ")")

'''
Draw a ToM model and save it as an image
Input variables:
tomsmodel - ToMsModel object to be drawn
savename - Filename of saved image
layout - Node layout algorithm for the graph. Can be one of "fruchterman_reingold", "spectral", "spring", "kamada_kawai", and "planar". See networkx.drawing.layout documentation.
pos - Dictionary of node positions. Useful for printing the same graph multiple times to investigate how it changes between updates
anss - String of answers to print in the upper right corner of the graph
correct - String with correct answer to print in the upper right corner of the graph
drawnodes - Whether to draw nodes
drawreflexive - Set to true to draw reflexive edges
drawtoms - Set to true to draw ToM tuples below each state
statefont - Font size for state names
edgefntsz - Font size for agent names on edges
fntsz - General font size used for everything else

Output variables:
pos - Dictionary of node positions. Useful for printing the same graph multiple times to investigate how it changes between updates
'''
def drawmodel_toms(tomsmodel, savename, layout, pos=None, anss=None, correct=None, drawnodes = True, drawreflexive = False, drawtoms = False, statefont = 20, edgefntsz = 18, fntsz = 16):
    todraw = tomsmodel.pmodel.fmodel.copy()  # Don't want to modify the input model, just in case
    ndsz = 300 #300  # Size of nodes
    fgsz = 15  # Figure height/width
    ndshp = 'o'  # Node shape
    ndlbcl = 'black'  # Node label colour
    edcl = 'black'  # Edge colour
    colorlist = ["orange", "yellow", "cyan"]  # Node colors
    edgeoffset = 0.1 # Offset for edge labels to ensure they are drawn on the edge itself
    plt.figure(1, figsize=(fgsz, fgsz))  # Create empty drawing area
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # Remove most margins

    edgelist = [(x[0], x[1]) for x in todraw.edges()]  # List of all edges
    # Generate coordinates for all nodes
    if pos is None:
        if layout == "fruchterman_reingold":
            pos = nx.fruchterman_reingold_layout(todraw)
        if layout == "spectral":
            pos = nx.spectral_layout(todraw)
        if layout == "spring":
            pos = nx.spring_layout(todraw)
        if layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(todraw)
        if layout == "planar":
            pos = nx.planar_layout(todraw)
    if pos is None:
        return

    minh = 0
    maxh = 0
    for i in range(len(pos)):  # Find horizontal range of nodes
        if pos[i][0] < minh:
            minh = pos[i][0]
        if pos[i][0] > maxh:
            maxh = pos[i][0]

    minv = 0
    maxv = 0
    for i in range(len(pos)):  # Find vertical range of nodes
        if pos[i][1] < minv:
            minv = pos[i][1]
        if pos[i][1] > maxv:
            maxv = pos[i][1]
    vdiff = maxv - minv  # Vertical distance between highest and lowest node
    nodelist = [(a, b) for (a, b) in todraw.nodes(data="state")]  # List of nodes (id, state) tuples
    # Draw nodes. Give different colours to true state, initial nodes, and all other nodes (if the former two are specified)
    if drawnodes:
        nx.draw_networkx_nodes(todraw, pos, [x[0] for x in nodelist if
                                             x[0] != tomsmodel.actualn and x[0] not in tomsmodel.untouchns],
                               node_color=colorlist[0],
                               node_size=ndsz, node_shape = ndshp)

        nx.draw_networkx_nodes(todraw, pos, [x[0] for x in nodelist if x[0] == tomsmodel.actualn],
                               node_color=colorlist[1],
                               node_size=ndsz)

        nx.draw_networkx_nodes(todraw, pos, [x[0] for x in nodelist if x[0] in tomsmodel.untouchns],
                               node_color=colorlist[2],
                               node_size=ndsz)

    uniquestates = list(set([x[1] for x in todraw.nodes(data="state")]))  # Remove duplicate states
    uniquestates.sort()
    statedict = dict([(a, str(b)) for (a, b) in todraw.nodes(data="state")])  # Dictionary of node states. Key is node ID, value is state
    alledges = todraw.edges(data="players")  # List of all edges
    pcount = tomsmodel.pmodel.playercount  # Player count

    #Code to find edges that are draw on top of each other
    overlapedges = []  # List of edges that go through a node, to be filled
    for p in range(pcount):  # Loop over players
        playeredges = [x for x in alledges if p in x[2] and x[0] != x[1]]  # Get player's edges
        for (i,j,ps) in playeredges:  # Loop over player's edges (i,j)
            overlap = False
            # Get coordinates for i and j
            icorx = pos[i][0]
            icory = pos[i][1]
            jcorx = pos[j][0]
            jcory = pos[j][1]
            #Figure out the minimum and maximum coordinates
            if icorx < jcorx:
                minx = icorx
                maxx = jcorx
            else:
                minx = jcorx
                maxx = icorx
            if icory < jcory:
                miny = icory
                maxy = jcory
            else:
                miny = jcory
                maxy = icory
            slope = (jcory - icory)/(jcorx - icorx)
            playernodes = []
            for (k, l, _) in playeredges:  # Loop over player's edges again
                if (k,l) != (i,j) and (k,l) != (j,i):   # They're not part of the original edge
                    if k not in playernodes:
                        playernodes.append(k)
                    if l not in playernodes:
                        playernodes.append(l)
            for n in playernodes: # Loop over both nodes for that edge
                # That node's coordinates
                ncorx = pos[n][0]
                ncory = pos[n][1]
                if ncorx > minx and ncorx < maxx and ncory > miny and ncory < maxy:  # Node is in the 'box'
                    newslope = (ncory - icory)/(ncorx - icorx)
                    if round(slope,4) == round(newslope,4):
                        overlap = True
            if overlap:
                if (i,j,ps) not in overlapedges:
                    overlapedges.append((i,j,ps))


    edgedict = dict(
        [((x[0], x[1]), str(x[2])[1:-1]) for x in todraw.edges(data="players") if x[0] != x[1] and x not in overlapedges])  # Dictionary of edge labels. str()[1:-1] removes list brackets, non-reflexive, not overlapping
    overlapdict = dict([((x[0], x[1]), str(x[2])[1:-1]) for x in overlapedges])  # Dictionary of edge labels for overlapping edges
    edgerefldict = dict([((x[0], x[1]), str(x[2])[1:-1]) for x in todraw.edges(data="players") if x[0] == x[1]])  # Dictionary of edge labels for reflexive edges

    nx.draw_networkx_labels(todraw, pos, statedict, font_size=statefont, font_color=ndlbcl)  # Draw node labels
    if tomsmodel.player != -1:
        nx.draw_networkx_labels(todraw, {0: [maxh, maxv + 0.1]}, {0: "Turn: " + str(tomsmodel.player)},
                                font_size=fntsz)  # If model belongs to a specific agent, print it in the corner
    if tomsmodel.level != -1:
        nx.draw_networkx_labels(todraw, {0: [maxh, maxv + 0.1]}, {0: " \n\n  Level: " + str(tomsmodel.level)},
                                font_size=fntsz)  # If model has a specific level, print it in the corner
    if not anss is None:  # Print anss in corner if it exists
        nx.draw_networkx_labels(todraw, {0: [maxh - 0.25, maxv + 0.1]}, {0: " \n\n\n\nAnswers: " + str(anss)},
                                font_size=fntsz)
    if not correct is None:  # Print correct in corner if it exists
        nx.draw_networkx_labels(todraw, {0: [maxh, maxv + 0.1]}, {0: " \n\n\n\n\n\nCorrect: " + str(correct)},
                                font_size=fntsz)

    # Separate symmetrical overlapping edges
    newoverlapdict = {}
    overlaplist1 = []
    overlaplist2 = []
    for (i,j) in overlapdict:
        ps = overlapdict[(i,j)]
        if (i,j) not in newoverlapdict and (j,i) not in newoverlapdict:
            newoverlapdict[(i,j)] = ps
            overlaplist1.append((i,j))
            if (j,i) in overlapdict:
                overlaplist2.append((j,i))

    if drawreflexive == True:
        edgelist = [x for x in edgelist if x not in overlaplist1 and x not in overlaplist2]  # Draw reflexive edges
    else:
        edgelist = [(a,b) for (a,b) in edgelist if (a,b) not in overlaplist1 and (a,b) not in overlaplist2 and a != b]  # Don't draw reflexive edges
    nx.draw_networkx_edges(todraw, pos, edgelist, edge_color = edcl)  # Draw edges
    rad = 0.6
    nx.draw_networkx_edges(todraw, pos, overlaplist1, connectionstyle='arc3,rad=0.6', edge_color = edcl)  # Draw curved edges for overlapping edges
    nx.draw_networkx_edges(todraw, pos, overlaplist2, connectionstyle='arc3,rad=-0.6', edge_color = edcl)  # Draw curved edges for overlapping edges

    # Math to make sure edge labels match up with curved edges
    for (i,j) in newoverlapdict:
        label = newoverlapdict[(i,j)]

        #degree = radian*(180/pi)  just in case

        #The curve is created such that the middle control point (C1) is located at the same distance from the start
        # (C0) and end points(C2) and the distance of the C1 to the line connecting C0-C2 is rad times the distance
        # of C0-C2. (times 0.5 but it doesn't say that)
        xi = pos[i][0]
        yi = pos[i][1]
        xj = pos[j][0]
        yj = pos[j][1]
        x = xi - xj
        y = yi - yj
        xm = (xi + xj)/2
        r = -1*(x/y)  # Breaks if y is 0
        edgelen = math.sqrt((x*x)+(y*y))
        linetocurve = edgelen*rad*0.5
        xn = math.sqrt((linetocurve*linetocurve)/(1+(r*r)))+xm
        yoffset = r*(xn-xm)
        xoffset = xn-xm
        if y > 0:
            yoffset = yoffset * -1
            xoffset = xoffset * -1
        for k in pos:
            pos[k][0] = pos[k][0] + xoffset
            pos[k][1] = pos[k][1] + yoffset
        nx.draw_networkx_edge_labels(todraw, pos, {(i,j):label}, label_pos=0.5, font_size=edgefntsz)
        for k in pos:
            pos[k][0] = pos[k][0] - xoffset
            pos[k][1] = pos[k][1] - yoffset

    nosymedgedict = {}
    for (w1,w2) in edgedict:
        if (w1,w2) not in nosymedgedict and (w2,w1) not in nosymedgedict:
            nosymedgedict[(w1,w2)] = edgedict[(w1,w2)]
    nx.draw_networkx_edge_labels(todraw, pos, nosymedgedict, label_pos=0.5, font_size=edgefntsz)  # Draw edge labels
    # Reflexive edge labels are drawn on the nodes themselves, so we pretend the nodes are at a higher y position

    if tomsmodel.actualn != -1:
        nx.draw_networkx_labels(todraw, pos, {tomsmodel.actualn: "_" * len(tomsmodel.actualstate)},
                                font_size=fntsz + 3)  # Highlight actual node, if required

    for i in pos:
        pos[i][1] = pos[i][1] + (edgeoffset * vdiff)
    if drawreflexive:
        nx.draw_networkx_edge_labels(todraw, pos, edgerefldict, label_pos=0,
                                 font_size=edgefntsz)  # Draw reflexive edge labels

    for i in pos:
        pos[i][1] = pos[i][1] - (1.35 * edgeoffset * vdiff)

    tomdict = {}
    for node in nodelist:
        tomdict[node[0]] = tomsmodel.nodestring(node[0])
    if drawtoms:
        nx.draw_networkx_labels(todraw, pos, tomdict, font_size=fntsz, font_color="red")  # Draw ToM stuff

    plt.savefig(savename + ".png")  # Save the image
    plt.clf()  # Clean the drawing space for future drawings

    for i in pos:
        pos[i][1] = pos[i][1] + (0.35 * edgeoffset * vdiff)
    return pos  # Return node positions for future drawings

def runtask():
    print("Drawing figures...")
    a8pm = PerfectModel(3, 2, "8888AAAA", "noself")  # Make perfect model for Aces and Eights
    a8pm.fullmodel_to_directed_reflexive()  # Turn non-directed graph without reflexive arrows into directed graph with reflexive arrows
    tsm = ToMsModel(a8pm, 5)  # Create a ToM model for Aces and Eights, maximum level 5
    layout = "spectral"
    pos = drawmodel_toms(tsm, "Figure1", layout, drawnodes = False)  # Draw and save model for Figure 1
    drawmodel_toms(tsm, "Figure2a", layout, pos=pos, drawreflexive=True, drawtoms=True, statefont=9,
                   edgefntsz=9, fntsz=9)  # Draw and save model for Figure 2a
    tsm.update(False, 0)  # Update ToM model with announcement `I do not know my cards' by player 0
    drawmodel_toms(tsm, "Figure2b", layout, pos=pos, drawreflexive=True, drawtoms = True, statefont = 9, edgefntsz = 9, fntsz = 9)  # Draw and save model for Figure 2b
    print("")
    print("Calculating proportions of SUWEB levels...")
    cwd = os.getcwd()  # Get working directory
    npz_directory = cwd + '/fitted_parameters/ModFit_'
    ModFit = np.load(npz_directory + "SUWEB.npz", allow_pickle=True)  # Load SUWEB's fit
    Params = ModFit['Params']  # Get SUWEB's fitted parameters
    lvls = [0, 0, 0, 0, 0]  # List with, for each epistemic level, how often a participant was fitted to it
    for i in range(len(Params)):  # Loop over participants
        lvls[int(Params[i][0])] += 1  # Increment best-fitting level by 1
    for i in range(len(lvls)):  # Loop over levels
        lvls[i] = lvls[i] / 211  # calculate proportion of this level in the population
    print("String for copying into R:")
    print("suweb <-c(" + str(lvls)[1:-1] + ")")

    print("")
    print("Running RFX-BMS on epistemically bounded models...")
    rfxbms(4, "tom_refTrue_delonFalse_", True, False, convergediff=0.001, penalty=0.5, emptyincorrect=False,
           usecedegao=True)  # 0.001, 0.5 default
    print("")
    print("Running RFX-BMS on ToM models...")
    rfxbms(5, "tom_refTrue_delonFalse_", True, False, convergediff=0.001, penalty=0.5, emptyincorrect=False,
           usecedegao=False)  # 0.001, 0.5 default
    print("")

start = timeit.default_timer()
runtask()
stop = timeit.default_timer()
print('Total time required: ', stop - start)