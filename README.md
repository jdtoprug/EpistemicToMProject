# EpistemicToMProject

This archive contains the code used in the article "Predictive Theory of Mind Models Based on Public Announcement Logic" (Top, Jonker, Verbrugge,  De Weerd (2023)).

## Requirements

python		3.10<br>
R		4.1.2<br>
matplotlib	3.5.1<br>
networkx	3.0<br>
numpy		1.21.5<br>
pandas		1.5.2<br>
scipy		1.8.0

## Installation

Open a terminal in `.../EpistemicTomProject` and run the following command:

`$ pip install -r requirements.txt`

## Running

Open a terminal in `.../EpistemicTomProject` and run the following command:

`$ python runme.py`

This generates several `.csv` filesNext, run `plotcode.R` (e.g. in RStudio) to import these save files and create the data and figures used in the article.

## Credit

The following files were copied from the paper "Does Amy know Ben knows you know your cards? A computational model of higher-order epistemic reasoning" (Cedegao, Ham, Holliday (2021)):

- The folder `fitted_parameters` and its contents
- `utils.py`
- `epistemic_structures.py`
- `data.csv`
- `agents.py`

Some of these files have received additional comments to improve readability, or have been modified for integration with our new code. This should be consistently marked with 'Top'.

## Directory tree


```bash
.
├── Figure1.png
├── Figure2a.png
├── Figure2b.png
├── README.md
├── agents.py
├── data.csv
├── epistemic_structures.py
├── fitted_parameters
│   └── ModFit_SUWEB.npz
├── plotcode.R
├── requirements.txt
├── runme.py
├── tom_models.py
└── utils.py
```

<div style="page-break-after: always; visibility: hidden"> 
&#x200B;
</div>

## Files

Ours:

- **runme.py** - Runs RFX-BMS and generates the data used in R.
- **tom_models.py** - Implements Theory of Mind models as networkx graph objects.
- **plotcode.R** - Reads the generated data and creates the images used in the article.
- **FigureX.png** - One of the figures used in the article.
- **requirements.txt** - Python modules needed to run the code.
- **README.md** - This file.

Cedegao's:

- **epistemic_structures.py** - Implements epistemically bounded models.
- **agents.py** - Implements Cedegao's agents.
- **utils.py** - Helper functions.
- **data.csv** - Cedegao's behavioral data.
- **ModFit_SUWEB.npz** - Fits for SUWEB models.


## Output files:

runme.py generates several data files:

**correctrates_XXX.csv**

Each row corresponds to a model, starting with level 0. Each column is a participant. Each item is the coherence of a participant where this model had the best coherence. The last and third-to-last rows are the random model.
The second-to-last row is the contatenation of each model's row.

**tom_refTrue_delonFalse_predictions.csv**

Contains the predictions for each model, for each participant's decision point.<br>
The first row is the header, and the last three rows are the parameter settings of the parameters reftom, delon, and confbi. For all remaining rows, columns are as follows:

1. Participant ID
2. Decision pount number
3. Distribution of cards
4. Turn the participant has in this game (starting at 0)
5. Current round
6. A list of lists containing the previous announcements the participant heard (including their own) before the current decision point. 'True' is 'I know my cards', 'False' is 'I do not know my cards'.
7. A list with, for each model (starting at level 0), a tuple indicating the model's prediction. The first element in the tuple is the answer, 
	0: 'I do not know my cards', 1: 'I know my cards', -1: There are no outgoing edges. The second element are the cards the model believes it can hold.
8. The participant's actual answer
9. The correct answer
10. Whether the participant's answer was correct
11. Which models would have given the correct answer (starting at 0)

**tom_refTrue_delonFalse_likelihoods.csv**

Contains the likelihoods for each model for each participant<br>
The first row is the header. For all remaining rows, columns are as follows:

1. Participant ID
2. Model level
3. Log-likelihood
4. Parameter setting of reftom
5. Parameter setting of delon
6. Coherence
7. Proportion with which the participant gave each answer, ordered as 'I know I have 88', 'I know I have 8A', 'I know I have AA', 'I don't know my cards'. Only printed in the row corresponding to the random model.
8. Random model's coherence
9. Proportion of player's decision points where they got the correct answer
10. Proportion of player's decision points where this model got the correct answer
