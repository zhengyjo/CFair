import pandas as pd
import numpy as np
from datetime import timedelta
from pandas.api.types import CategoricalDtype
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
import pickle
import argparse
import random

# stats test
from scipy.stats import f
import scipy.stats as stats

drive_dir = './'

# The helper function
from calculation import *
import sampling
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression


# create the parser object
parser = argparse.ArgumentParser(description='Example argparse script')
parser.add_argument('input_file', help='The file including all the pair-wise distance')
parser.add_argument('input_file_2', help='The file including respective the pair-wise distance')
parser.add_argument('p_value', help='The chosen p-value threshold')
parser.add_argument('name', help='Save qualified delta-group index')

all_pair_distance_lst = args.input_file
all_pair_distance = args.input_file_2
p_value = float(args.delta)
name = args.name

# Pass the original candidates list and the p-value threshold to filter out unqualified candidates 
candidate_index_lst = filter_with_delta_threshold(all_pair_distance, all_pair_distance_lst, p_value)

with open(name, "wb") as fp:  # Pickling the results
    pickle.dump(candidate_index_lst, fp)
