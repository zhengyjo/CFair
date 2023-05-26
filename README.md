# CFair

This repository is the official implementation of *Counterpart Fairness – Addressing Systematic between-group Differences in Fairness Evaluation*.

## Abstract
When using machine learning (ML) to aid decision-making, it is critical to ensure that an algorithmic decision is fair, i.e., it does not discriminate against specific individuals/groups, particularly those from underprivileged populations. Existing group fairness methods require equal group-wise measures, which however fails to consider systematic between-group differences. The confounding factors, which are non-sensitive variables but manifest systematic differences, can significantly affect fairness evaluation. To mitigate this problem, we believe that a fairness measurement should be based on the comparison between **counterparts** (i.e., individuals who are similar to each other with respect to the task of interest) from different groups, whose group identities cannot be distinguished algorithmically by exploring confounding factors. We have developed a propensity-score-based method for identifying counterparts, which prevents fairness evaluation from comparing "oranges" with "apples". In addition, we propose a counterpart-based statistical fairness index, termed Counterpart-Fairness (CFair), to assess fairness of ML models. Empirical studies on the Medical Information Mart for Intensive Care (MIMIC)-IV database were conducted to validate the effectiveness of CFair.

## Dataset
We are using Sepsis patients from MIMIC-IV dataset.

0. User may need to apply for the access to MIMIC dataset by going to [this page](https://physionet.org/content/mimiciv/2.0/)
1. Please use the files in sql folder to extract the raw data of sepsis patients from MIMIC-IV 
2. Please use the file from pre-processing folder to process the data

## Code Structure
The codebase is organized as follows:

- `sql/`: This directory contains the sql files to extract the raw data of sepsis patients from MIMIC-IV dataset

- `preprocess/`: This directory contains the preprocessing file for sepsis patients.

- `matching/`: This directory contains the main source code for CFair matching.
  - `calculation.py`: The functions that we use in all kind of calculation in matching.
  - `sampling.py`: Other utility function.
  - `weighted_health_matching_helper.py` : The helper functions for second stage matching in terms of Mahalanobis distance with health conditions
  - `propensity_score_training.py`: The example script to train a propensity score model. 
  - `propensity_score_matching.py`: The example script to perform pair-wise propensity score matching.
  - `propensity_score_filtering.py`: The example script to perform pair-wise propensity score filtering.
  - `weighted_health_matching_run.py`: The example script to perform pair-wise health condition matching with Mahalanobis distance.

- `Ventilation_classification/`: This directory contains the helper function and a jupyter notebook demo about trainining and evaluating a model for predicting ventilation status, based on the counterparts and group-based fairness metric.

- `run_learned_matrix/`: This directory contains how we update the matrix that we used in Mahalanobis distance


- `README.md`: This file you are currently reading, providing an overview of the project and instructions.

## Usage
- Run the scripts from sql and pre-processing to process the data.
- Perform CFair matching by running the scripts in matching folders as following order:
  - Run `propensity_score_training.py` to train a propensity score model and give each patient a
  propensity score by `python propensity_score_training.py [Path to csv file including two ethic groups] `
  - Run `propensity_score_matching.py` to perform pair-wise propensity score matching between any combination of minority and majority patients by `python propensity_score_training.py [Path to csv file including minority group] [Path to csv file including majority group] [The output name of all the pair-wise distance] [The output name that includes the distance between each minor and respective major] `
  - Run `propensity_score_filtering.py` to perform propensity score filtering to remove candidates that is too far away from the respective minority patients in terms of propensity score by `python propensity_score_filtering.py [The file including all the pair-wise distance] [The file including respective the pair-wise distance] [The chosen p-value threshold] [output directory of qualified propensity score candidates index] `
  - Run `weighted_health_matching_run.py` to perform health condition matching to find the counterparts by `python weighted_health_matching_run.py [Path to directory of qualified propensity score candidates index] [Path to minority csv file] [Path to majority csv file] --cov [a file for the learned covariance matrix]`
- Load the counterparts from last step in the training and evaluation of Ventilation prediction, just as what we show in the jupyter notebook from the folder of Ventilation_classification.
  



