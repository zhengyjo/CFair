# Read me
## Function illustration

### calculation.dp_t_calculation_two_group(df_black, df_white, features):
 Calculate the Dp Gap and paired t-test for two groups
    
* Parameters:
    - df_black: DataFrame. The panda data frame of minority group 
    - df_white: DataFrame. The panda data frame of majority group 
    - features: list. The selected features to use in calculating DP gap and paired-t test

* Return:
    * dict. The keys are respective features from parameters, and the corresponding values are the lists that contains DP gap and paired-t test results. The list from value is in the following form:
      - The first element of value is DP gap.
      - The second element of value is the result of shapiro test,which is to test whether the difference of
      The selected feature between two group belongs to a normal distribution.
      - The third element of value is the result of Wilcoxon-Sign test between two groups.
      - The forth element of value is the result of paired-t test between two groups.

### calculation.plot_standardized_change(df_white_black_selected_filtered, df_black_selected, df_white_selected,measure_columns, treatment):
Draw the comparison figure about the changed of standardized mean of selected features before and after matching .
* Parameters:
    - df_white_black_selected_filtered: DataFrame. The panda data frame of the whole population before matching
    - df_black_selected: DataFrame. The panda data frame of minority group after matching
    - df_white_selected: DataFrame. The panda data frame of majority group after matching/sampling
    - measure_columns: list. the selected features to compare
    - treatment: str. The target variable for propensity score

* Return:
    - figure. The comparison figure between before and after matching/sampling.

### calculation.calculate_all_pair_difference(df_black, df_white, feature):
 Calculate the euclidian distance of selected feature between each possible pair with one candidate in majority
 group and the other one from minority group.    

* Parameters:
    - df_black: DataFrame. The data frame of minority group
    - df_white: DataFrame. The data frame of majority group
    - feature: str. The feature name that you want to calculate the difference.

* Return:
    - list.The respective distance of selected feature of all the possible pairs. This 2d-list is in the following form:
      - size of 0-axis: # of minority group
      - size of 1-axis: # of majority group
    

### calculation.filter_with_delta_threshold(distance_psm_all, all_pair_distance_lst, delta):
First stage of counter-part filtering with given delta
* Parameters:
    - distance_psm_all: list.2D-list 
      - size of 0-axis: # of minority group
      - size of 1-axis: # of majority group
    - all_pair_distance_lst: list. 1D-list. It contains all the difference of possible pairs.
    - delta: float. The p-value threshold for null distribution of all the pair-wise difference among
    pairs with one candidate from major group and the other one from minority group

* Return :
    
    - list.  The indices of qualified neighbor for each minority candidate.

### calculation.filter_with_std(df_whole, df_black, df_white, measure_columns, index_col, candidate_index_lst, coef=1):
Second stage of counter-part filtering. In this stage, we only keep neighbors, the difference between whom and
counter element is less than coef * standard deviation for each selected feature
* Parameters:
    - df_whole: DataFrame. The data frame combining majority group and minority group. It is used to calculate standard deviation.
    - df_black: DataFrame. The data frame of minority group.
    - df_white: DataFrame. The data frame of majority group.
    - measure_columns: list. Features that we need to consider for pair-wise distance constrain.
    - index_col: str. The unique index like 'subject_id'
    - candidate_index_lst: list. Qualified index list from the delta-selection.
    - coef: float. Scalar for standard deviation. Default=1.

* Return :
    - list.  The indices of qualified counter-part for each minority counter-element after second-stage filtering.


### dp_t_calculation_random(df_black_selected, df_white_whole, repeated_times, features):
Random Sample and calculate the Dp Gap and paired t-test for two groups
* Parameters:
    - df_black: DataFrame. The panda data frame of the selected minority group
    - df_white: DataFrame. The panda data frame of the whole majority group
    - repeated_times: int. Times the user want to apply the random sampling
    - features: list. the selected features to use in calculating DP gap and paired-t test

* Return:
  - dict. The keys are respective features from parameters, and the corresponding values are
        the lists that contains DP gap and paired-t test results.
        The list from value is in the following form:
      - The first element of value is DP gap.
      - The second element of value is the result of shapiro test,which is to test whether the difference of
      The selected feature between two group belongs to a normal distribution.
      - The third element of value is the result of Wilcoxon-Sign test between two groups.
      - The forth element of value is the result of paired-t test between two groups.

### sampling.dp_t_calculation_stratified(df_black_selected, df_white_whole, repeated_times, features, ratio_feature):
Stratified Sample and calculate the Dp Gap and paired t-test for two groups
* Parameters:
    - df_black: DataFrame. The data frame of the selected minority group
    - df_white: DataFrame. The pandas data frame of the whole majority group
    - repeated_times: int. Times the user want to apply the random sampling
    - features: list. The selected features to use in calculating DP gap and paired-t test
    - ratio_feature: str. The feature you want the stratified sampling base on.

* Return:
    - dict. The keys are respective features from parameters, and the corresponding values are
          the lists that contains DP gap and paired-t test results.
          The list from value is in the following form:
        - The first element of value is DP gap.
        - The second element of value is the result of shapiro test,which is to test whether the difference of The selected feature between two group belongs to a normal distribution.
        - The third element of value is the result of Wilcoxon-Sign test between two groups.
        - The forth element of value is the result of paired-t test between two groups.

### output_sampling_result(res_dict, conf_level=0.05):
 Output the sampling result in the form of panda data frame.
 * Parameters:
    - res_dict: dict. The resulting dictionary from sampling.
    - conf_level: float. p-value confidence level for shapiro test. If the p-value of a shapiro test is less than
    the confidence level, we reject the null hypothesis that the corresponding distribution is normal. We use this to
    judge whether to use paire-t test result or Wilcoxon-Sign test result. Default = 0.05


* Return:
   data frame. It contains five columns: 'feature', 'dp_mean', 'dp_std', 'p_value_mean', 'p_value_std'
            feature: str. Respective feature name.
  - dp_mean: float. The mean of DP among all the sampling results.
  - dp_std: float. The standard deviation of DP among all the sampling results.
  - p_value_mean: float. The mean of DP among all the sampling results.
  - p_value_std: float. The standard deviation of DP among all the sampling results.

        
    

    


    