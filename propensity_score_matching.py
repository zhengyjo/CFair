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

# create the parser object
parser = argparse.ArgumentParser(description='Example argparse script')

# add arguments to the parser
parser.add_argument('minority_file', help='Path to csv file including minority group')
parser.add_argument('majority_file', help='Path to csv file including majority group')
parser.add_argument('output_name', help='The output name of all the pair-wise distance')
parser.add_argument('output_name_2', help='The output name that includes the distance between each minor and respective major.')

black_file = args.minority_file
white_file = args.majority_file
name = args.output_name
name_pair = args.output_name_2

df_black = pd.read_csv(black_file)
df_white = pd.read_csv(white_file)

all_pair_distance = calculate_all_pair_difference(df_black, df_white, 'propensity_logit')

all_pair_distance_lst = []
for individual in all_pair_distance:
    all_pair_distance_lst.append(individual)

with open(name, "wb") as fp:  # Pickling
    pickle.dump(all_pair_distance_lst, fp)

with open(name_pair, "wb") as fp:  # Pickling
    pickle.dump(all_pair_distance, fp)
