import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

# stats test
from scipy.stats import f
import scipy.stats as stats


def dp_t_calculation_two_group(df_black, df_white, features):
    """
    Calculate the Dp Gap and paired t-test for two groups
    Parameters:
    -------
    df_black: DataFrame. The data frame of minority group
    df_white: DataFrame. The pandas data frame of majority group
    features: list. The selected features to use in calculating DP gap and paired-t test


    Return:
    -------
    dict. The keys are respective features from parameters, and the corresponding values are
          the lists that contains DP gap and paired-t test results.
          The list from value is in the following form:
                1. The first element of value is DP gap.
                2. The second element of value is the result of shapiro test,which is to test whether the difference of
                The selected feature between two group belongs to a normal distribution.
                3. The third element of value is the result of Wilcoxon-Sign test between two groups.
                4. The forth element of value is the result of paired-t test between two groups.
    """
    res_dict = {}
    for ele in features:
        res_dict[ele] = [[], [], [], []]
        ran = df_white[ele]
        ran_black = df_black[ele]
        res_dict[ele][1].append(stats.shapiro(ran_black - ran)[1])
        res_dict[ele][2].append(stats.ttest_rel(ran_black, ran)[1])
        res_dict[ele][3].append(stats.wilcoxon(ran_black, ran)[1])
        res_dict[ele][0].append(abs(ran_black.mean() - ran.mean()))
    return res_dict



def plot_standardized_change(df_white_black_selected_filtered, df_black_selected, df_white_selected,
                             measure_columns, treatment):
    """
    Draw the comparison figure about the changed of standardized mean of selected features before and after matching .
    Parameters:
    -------
    df_white_black_selected_filtered: DataFrame.The panda data frame of the whole population before matching
    df_black_selected: DataFrame. The panda data frame of minority group after matching
    df_white_selected: DataFrame. The panda data frame of majority group after matching/sampling
    measure_columns: list. The selected features to compare
    treatment: str. The target variable for propensity score


    Return:
    -------
    figure. The comparison figure between before and after matching/sampling.
    """
    df_original = df_white_black_selected_filtered
    df_black_selected[treatment] = 1
    df_white_selected[treatment] = 0
    df_matched = pd.concat([df_white_selected.reset_index(), df_black_selected.reset_index()])
    data = []
    for cl in measure_columns:
        try:
            data.append([cl, 'Before matching', cohenD(df_original, treatment, cl)])
        except:
            data.append([cl, 'Before matching', 0])
        try:
            data.append([cl, 'After matching', cohenD(df_matched, treatment, cl)])
        except:
            data.append([cl, 'After matching', 0])
    res = pd.DataFrame(data, columns=['variable', 'matching', 'effect_size'])
    sns.set_style("white")
    sn_plot = sns.barplot(data=res, y='variable',
                          x='effect_size', hue='matching', orient='h')
    title = 'Standardized Mean differences accross covariates before and after matching'
    sn_plot.set(title=title)


# Use Euclidian distance to calculate difference
def calculate_all_pair_difference(df_black, df_white, feature):
    """
    Calculate the euclidian distance of selected feature between each possible pair with one candidate in majority
    group and the other one from minority group.
    Parameters:
    -------
    df_black: DataFrame. The data frame of minority group
    df_white: DataFrame. The data frame of majority group
    feature: str. The feature name that you want to calculate the difference.

    Return :
    -------
    list.  The respective distance of selected feature of all the possible pairs. This 2d-list is in the following
           form:
                size of 0-axis: # of minority group
                size of 1-axis: # of majority group
    """
    distance_psm_nor_all = []
    for index_black, row_black in df_black.iterrows():
        black_prop = row_black[feature]
        temp_dis = []
        for index_white, row_white in df_white.iterrows():
            white_prop = row_white[feature]
            dis = math.sqrt((black_prop - white_prop) * (black_prop - white_prop))
            temp_dis.append(dis)
        distance_psm_nor_all.append(temp_dis)
    return distance_psm_nor_all


def filter_with_delta_threshold(distance_psm_all, all_pair_distance_lst, delta):
    """
    First stage of counter-part filtering with given delta
    Parameters:
    -------
    distance_psm_all: list.2D-list
      - size of 0-axis: # of minority group
      - size of 1-axis: # of majority group
    all_pair_distance_lst: list. 1D-list. It contains all the difference of possible pair.
    delta: float. The p-value threshold for null distribution of all the pair-wise difference among
    pairs with one candidate from major group and the other one from minority group

    Return :
    -------
    list.  The indices of qualified neighbor for each minority candidate.
    """
    threshold = np.quantile(all_pair_distance_lst, delta)
    candidate_index_lst = []
    for i in range(len(distance_psm_all)):
        temp_lst = np.where(np.array(distance_psm_all[i]) < threshold)
        candidate_index_lst.append(temp_lst)
    return candidate_index_lst


def filter_with_std(df_whole, df_black, df_white, measure_columns, index_col, candidate_index_lst, coef=1):
    """
    Second stage of counter-part filtering. In this stage, we only keep neighbors, the difference between whom and
    counter element is less than coef * standard deviation for each selected feature
    Parameters:
    -------
    df_whole: DataFrame. The data frame combining majority group and minority group. It is used to calculate standard deviation.
    df_black: DataFrame. The data frame of minority group.
    df_white: DataFrame. The data frame of majority group.
    measure_columns: list. Features that we need to consider for pair-wise distance constrain.
    index_col: str. The unique index like 'subject_id'
    candidate_index_lst: list. Qualified index list from the delta-selection.
    coef: float. Scalar for standard deviation

    Return :
    -------
    list.  The indices of qualified counter-part for each minority counter-element after second-stage filtering.
    """
    std = df_whole.std()[measure_columns]
    qualified_ids_lst = []
    for index_black, row_black in df_black.iterrows():
        black_id = row_black[index_col]
        black_features = df_whole[df_whole[index_col] == black_id][measure_columns]

        qualified_ids = []
        for index_white in candidate_index_lst[index_black][0]:
            white_id = df_white.iloc[index_white][index_col]
            white_features = df_whole[df_whole[index_col] == white_id][measure_columns]
            flag = 1
            for ele in measure_columns:
                white_ele = float(white_features[ele])
                black_ele = float(black_features[ele])
                std_ele = float(std[ele])
                if (abs(black_ele - white_ele)) > coef * std_ele:
                    flag = 0
                    break
            if flag == 1:
                qualified_ids.append(white_id)
        qualified_ids_lst.append(qualified_ids)
    return qualified_ids_lst
