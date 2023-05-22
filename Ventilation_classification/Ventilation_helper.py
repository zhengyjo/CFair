import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix



"""
Plot the distribution of two different group
"""

def plot_distribution(df_black,df_white,ele):
    minority_feature = df_black[ele]
    match_feature = df_white[ele]
    bins=np.histogram(np.hstack([minority_feature,match_feature]), bins=15)[1]
    plt.figure(figsize=(8,6))
    plt.hist(minority_feature,bins = bins,weights=np.ones(len(minority_feature)) / len(minority_feature)*100,alpha=0.5,label="Black")
    plt.hist(match_feature,bins = bins,weights=np.ones(len(match_feature)) / len(match_feature)*100,alpha=0.5,label="White")
    plt.xlabel("%s"%(ele), size=14)
    plt.ylabel("Percentage", size=14)
    plt.title("%s Distribution of Black and white"%(ele))
    plt.legend(loc='center right')

def dp_t_calculation_two_group_no_pair(df_black, df_white, features):
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
        print(ele)
        res_dict[ele] = [[], [], [],[], []]
        ran = df_white[ele]
        ran_black = df_black[ele]
        res_dict[ele][1].append(stats.shapiro(ran_black)[1])
        res_dict[ele][2].append(stats.shapiro(ran)[1])
        res_dict[ele][3].append(stats.ttest_ind(ran_black, ran)[1])
        res_dict[ele][4].append(stats.mannwhitneyu(ran_black, ran)[1])
        res_dict[ele][0].append(abs(ran_black.mean() - ran.mean()))
    return res_dict
  
     
    
def Abs_mean_diff_normalized(df_minor,df_major,df_whole,cl):
    mean_minor = df_minor[cl].mean()
    mean_major = df_major[cl].mean()
    difference = abs(mean_minor - mean_major)
    feature_mean = df_whole[cl].mean()
    return difference/feature_mean
  
def split_with_match(df_minor_one_to_one,df_major_one_to_one,n_fold):
    if (len(df_minor_one_to_one) != len(df_major_one_to_one)):
        print("The size of counterparts are not equal!")
        return
    whole_lst = np.arange(len(df_minor_one_to_one))
    df_counter_train_lst = []
    df_counter_test_lst = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(whole_lst)):
        df_minor_counterpart_train = df_minor_one_to_one.iloc[train_idx]
        df_minor_counterpart_test =  df_minor_one_to_one.iloc[test_idx]
        df_major_counterpart_train = df_major_one_to_one.iloc[train_idx]
        df_major_counterpart_test = df_major_one_to_one.iloc[test_idx]
        
        df_train = pd.concat([df_minor_counterpart_train,df_major_counterpart_train])
        df_test = pd.concat([df_minor_counterpart_test,df_major_counterpart_test])
        
        df_counter_train_lst.append(df_train)
        df_counter_test_lst.append(df_test)

    return df_counter_train_lst,df_counter_test_lst
  
def split_without_match(df,n_fold):
    whole_lst = np.arange(len(df))
    df_train_lst = []
    df_test_lst = []
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(whole_lst)):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
        df_train_lst.append(df_train)
        df_test_lst.append(df_test)

    return df_train_lst,df_test_lst
  
def vent_paired_t_test (countert_minor,counter_major):
    differences = []
    for a,b in zip (countert_minor,counter_major):
        temp = np.linalg.norm(a - b,ord=1)
        differences.append(temp)
    # Calculate the mean difference
    mean_difference = np.mean(differences)

    # Calculate the standard deviation of the differences
    std_difference = np.std(differences, ddof=1)

    # Calculate the standard error of the mean difference
    n = len(differences)
    sem_difference = std_difference / np.sqrt(n)

    # Calculate the t-statistic
    t_statistic = mean_difference / sem_difference

    # Calculate the degrees of freedom
    df = n - 1

    # Calculate the p-value (two-tailed test)
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))
    
    return t_statistic,p_value
  
  
def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    res = {}
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Overall Accuracy: {:.2f}".format(accuracy))
    res['Overall_Accuracy'] = accuracy 

    # Calculate confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion)
    res['Confusion_Matrix'] = confusion 

    # Calculate fairness metrics
    num_groups = len(np.unique(sensitive_features))

    for group in np.unique(sensitive_features):
        group_indices = np.where(sensitive_features == group)[0]
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        # Calculate group-specific accuracy
        group_accuracy = accuracy_score(group_y_true, group_y_pred)
        print("Group '{}' Accuracy: {:.2f}".format(group, group_accuracy))
        res["Group_'{}'_Accuracy".format(group)] = group_accuracy 

        # Calculate disparate impact
        group_positive_ratio = np.mean(group_y_pred == 1)
        overall_positive_ratio = np.mean(y_pred == 1)
        disparate_impact = group_positive_ratio / overall_positive_ratio
        print("Group '{}' Disparate Impact: {:.2f}".format(group, disparate_impact))
        res["Group_'{}'_Disparate_Impact".format(group)] = disparate_impact 

        # Calculate equal opportunity
        equal_opportunity = []
        for i in range(len(np.unique(y_true))):
            print(i)
            group_true_positive = np.sum((group_y_true == i) & (group_y_pred == i))
            print(group_true_positive)
            overall_true_positive = np.sum((y_true == i) & (y_pred == i))
            print(overall_true_positive)
            equal_opportunity.append(group_true_positive / overall_true_positive)
            print(equal_opportunity)
        equal_opportunity = np.mean(equal_opportunity)
        print("Group '{}' Equal Opportunity: {:.2f}".format(group, equal_opportunity))
        res["Group_'{}'_Equal_Opportunity".format(group)] = equal_opportunity

        # Calculate predictive parity
        predictive_parity = []
        for i in range(len(np.unique(y_pred))):
            group_positive_ratio = np.mean(group_y_pred == i)
            overall_positive_ratio = np.mean(y_pred == i)
            predictive_parity.append(group_positive_ratio / overall_positive_ratio)
        predictive_parity = np.mean(predictive_parity)
        print("Group '{}' Predictive Parity: {:.2f}".format(group, predictive_parity))
        res["Group_'{}'_Predictive_Parity".format(group)] = predictive_parity

        # Calculate equalized odds
        equalized_odds = []
        for i in range(len(np.unique(y_true))):
            group_true_positive = np.sum((group_y_true == i) & (group_y_pred == i))
            group_true_negative = np.sum((group_y_true != i) & (group_y_pred != i))
            overall_true_positive = np.sum((y_true == i) & (y_pred == i))
            overall_true_negative = np.sum((y_true != i) & (y_pred != i))
            equalized_odds.append((group_true_positive / overall_true_positive) / (group_true_negative / overall_true_negative))
        equalized_odds = np.mean(equalized_odds)
        print("Group '{}' Equalized Odds: {:.2f}".format(group, equalized_odds))
        res["Group_'{}'_Equalized_Odds".format(group)] = equalized_odds

        # Calculate equalized opportunity
        equalized_opportunity = []
        for i in range(len(np.unique(y_true))):
            group_true_positive = np.sum((group_y_true == i) & (group_y_pred == i))
            group_false_negative = np.sum((group_y_true == i) & (group_y_pred != i))
            overall_true_positive = np.sum((y_true == i) & (y_pred == i))
            overall_false_negative = np.sum((y_true == i) & (y_pred != i))
            group_false_negative_rate = group_false_negative / (group_false_negative)
        print("Group '{}' Equal Opportunity: {:.2f}".format(group, equal_opportunity))
        res["Group_'{}'_Equal_Opportunity".format(group)] = equal_opportunity
        
    return res
