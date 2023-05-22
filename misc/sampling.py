import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

# stats test
from scipy.stats import f
import scipy.stats as stats

def dp_t_calculation_random(df_black_selected, df_white_whole, repeated_times, features):
    """
    Random Sample and calculate the Dp Gap and paired t-test for two groups
    Parameters:
    -------
    df_black: DataFrame. The panda data frame of the selected minority group
    df_white: DataFrame. The panda data frame of the whole majority group
    repeated_times: int. Times the user want to apply the random sampling
    features: list. the selected features to use in calculating DP gap and paired-t test


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
    n_sample = len(df_black_selected)
    for i in range(repeated_times):
        df_sample_white = df_white_whole.sample(n=n_sample)
        for ele in features:
            ran = df_sample_white[ele]
            ran_black = df_black_selected[ele]
            res_dict[ele][1].append(stats.shapiro(ran_black-ran)[1])
            res_dict[ele][2].append(stats.ttest_rel(ran_black, ran)[1])
            res_dict[ele][3].append(stats.wilcoxon(ran_black, ran)[1])
            res_dict[ele][0].append(abs(ran_black.mean() - ran.mean()))
    return res_dict


def dp_t_calculation_stratified(df_black_selected, df_white_whole, repeated_times, features, ratio_feature):
    """
    Stratified Sample and calculate the Dp Gap and paired t-test for two groups
    Parameters:
    -------
    df_black: DataFrame. The data frame of the selected minority group
    df_white: DataFrame. The pandas data frame of the whole majority group
    repeated_times: int. Times the user want to apply the random sampling
    features: list. The selected features to use in calculating DP gap and paired-t test
    ratio_feature: str. The feature you want the stratified sampling base on.


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
    test_dict = df_black_selected[ratio_feature].value_counts().to_dict()
    for i in range(repeated_times):
        df_sample_white = pd.DataFrame()
        for key in test_dict.keys():
            temp = df_white_whole[df_white_whole[ratio_feature] == key].sample(n=test_dict[key])
            df_sample_white = pd.concat([df_sample_white, temp], ignore_index=True)
        for ele in features:
            ran = df_sample_white[ele]
            ran_black = df_black_selected[ele]
            res_dict[ele][1].append(stats.shapiro(ran_black-ran)[1])
            res_dict[ele][2].append(stats.ttest_rel(ran_black, ran)[1])
            res_dict[ele][3].append(stats.wilcoxon(ran_black, ran)[1])
            res_dict[ele][0].append(abs(ran_black.mean() - ran.mean()))
    return res_dict


def output_sampling_result(res_dict, conf_level=0.05):
    """
    Output the sampling result in the form of panda data frame.
    -------
    Parameters:
    res_dict: dict. The resulting dictionary from sampling.
    conf_level: float. p-value confidence level for shapiro test. If the p-value of a shapiro test is less than
    the confidence level, we reject the null hypothesis that the corresponding distribution is normal. We use this to
    judge whether to use paire-t test result or Wilcoxon-Sign test result.


    Return:
    -------
    Panda data frame. It contains five columns: 'feature', 'dp_mean', 'dp_std', 'p_value_mean', 'p_value_std'
            feature: str. Respective feature name.
            dp_mean: float. The mean of DP among all the sampling results.
            dp_std: float. The standard deviation of DP among all the sampling results.
            p_value_mean: float. The mean of DP among all the sampling results.
            p_value_std: float. The standard deviation of DP among all the sampling results.
    """
    df_res = pd.DataFrame(columns=[['feature', 'dp_mean', 'dp_std', 'p_value_mean', 'p_value_std']])
    for key in res_dict.keys():
        p_value = []
        for ind in range(len(res_dict[key][1])):
            if (res_dict[key][1][ind] < conf_level):
                p_value.append(res_dict[key][3][ind])
            else:
                p_value.append(res_dict[key][2][ind])
        p_value_np = np.array(p_value)
        dp_np = np.array(res_dict[key][0])
        df_res.loc[len(df_res)] = [key, dp_np.mean(), dp_np.std(), p_value_np.mean(), p_value_np.std()]
    return df_res
