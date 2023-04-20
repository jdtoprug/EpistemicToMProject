EpistemicToMProject contains the code use in the paper "Title" (Top, , , , 202)

The following files were copied from the paper "Does Amy know Ben knows you know your cards? A computational model of higher-order epistemic reasoning" (Cedegao, Ham, Holliday (2021)):
-The folder fitted_parameters and contents.
-utils.py
-epistemic_structures.py
-data.csv
-agents.py

Some of these files have received additional comments to improve readability, or have been modified for integration with our new code. This should be consistently marked with 'Top'.

To obtain the graphs used in the paper, run runme.py, then run plotcode.R.
Versions used:
Python 3.10
R 4.1.2

runme.py generates several data files:

correctrates_XXX.csv
Each row corresponds to a model, starting with level 0. Each item is the coherence of a participant where this model had the best coherence. The last and third-to-last rows are the random model.
The second-to-last row is the contatenation of each model's row.

tom_refTrue_delonFalse_predictions.csv
Contains the predictions for each model, for each participant's decision point.
The first row is the header, and the last three rows are the parameter settings of the parameters reftom, delon, and confbi. For all remaining rows, columns are as follows:
1 - Participant ID
2 - Decision pount number
3 - Distribution of cards
4 - Turn the participant has in this game (starting at 0)
5 - Current round
6 - A list of lists containing the previous announcements the participant heard (including their own) before the current decision point. `True' is `I know my cards', `False' is `I do not know my cards'.
7 - A list with, for each model (starting at level 0), a tuple indicating the model's prediction. The first element in the tuple is the answer, 
	0: `I do not know my cards', 1: `I know my cards', -1: There are no outgoing edges. The second element are the cards the model believes it can hold.
8 - The participant's actual answer
9 - The correct answer
10 - Whether the participant's answer was correct
11 - Which models would have given the correct answer (starting at 0)

tom_refTrue_delonFalse_likelihoods.csv
Contains the likelihoods for each model for each participant
The first row is the header. For all remaining rows, columns are as follows:
1 - Participant ID
2 - Model level
3 - Log-likelihood
4 - Parameter setting of reftom
5 - Parameter setting of delon
6 - Coherence
7 - Proportion with which the participant gave each answer, ordered as 'I know I have 88', 'I know I have 8A', 'I know I have AA', 'I don't know my cards'. Only printed in the row corresponding to the random model.
8 - Random model's coherence
9 - Proportion of player's decision points where they got the correct answer
10 - Proportion of player's decision points where this model got the correct answer