import pandas as pd
import numpy as np
from datetime import timedelta
from pandas.api.types import CategoricalDtype

def get_year_month_day_hour(dynamic_df):
    """
    returns the hourly average value of the dynamic variable dataframe
    """
    dynamic_df.loc[:, 'charttime_year'] = pd.DatetimeIndex(dynamic_df['charttime']).year
    dynamic_df.loc[:, 'charttime_month'] = pd.DatetimeIndex(dynamic_df['charttime']).month
    dynamic_df.loc[:, 'charttime_day'] = pd.DatetimeIndex(dynamic_df['charttime']).day
    dynamic_df.loc[:, 'charttime_hour'] = pd.DatetimeIndex(dynamic_df['charttime']).hour

    by_hour_new = dynamic_df.groupby(['subject_id', 'stay_id', 'charttime_year', 'charttime_month', 'charttime_day', 'charttime_hour']).mean()
    by_hour_new = by_hour_new.reset_index()
    return by_hour_new

drive_dir = './'

# Read the corresponding data file
s3_dynamic_vital = pd.read_csv(drive_dir + 'raw_files/deci_s3_hourly_no_lab.csv') # adjusted sbp, dbp, mbp
s3_ventilation = pd.read_csv(drive_dir + 'raw_files/s3_ventilation.csv')
# add the hour column
# keeps track progress in ICU
s3_dynamic_vital['hour'] = s3_dynamic_vital.groupby(['subject_id']).cumcount() # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html

# read in each patient's demographics
s3_static = pd.read_csv(drive_dir + "raw_files/s3_static.csv")


# Concatenate Variables
bg =  pd.read_csv(drive_dir + 'raw_files/s3_bg.csv')
cbc =  pd.read_csv(drive_dir + 'raw_files/s3_cbc.csv')

# Preprocessing
s3_vent_play = s3_ventilation.copy()

# some pt not ventilation during ICU stay
pt_on_vent = s3_vent_play[~s3_vent_play['starttime'].isna()]

pt_on_vent['starttime'] = pt_on_vent.apply(lambda d: pd.date_range(d['starttime'],
                                                    d['endtime'],
                                                    freq='h')[:-1], axis=1)
pt_on_vent_hourly = pt_on_vent.explode('starttime', ignore_index=True)

# create ranked order of ventilation severity
vent_type = CategoricalDtype(categories=["None", "SupplementalOxygen", "HFNC", "NonInvasiveVent", "Tracheostomy", "InvasiveVent"], ordered=True)
s3_vent_play = s3_ventilation.copy()

# integer code ventilation based on severity
pt_on_vent_hourly = pt_on_vent_hourly[['subject_id', 'stay_id', 'starttime', 'ventilation_status']]
pt_on_vent_hourly['ventilation_status'] = pt_on_vent_hourly['ventilation_status'].astype(vent_type)
pt_on_vent_hourly['ventilation_status'] = pt_on_vent_hourly['ventilation_status'].cat.codes

# extract hours from the exploded df
# dummy code the ventilation status with order severity
# merge (left) with the dynamic array

# rename to match other dynamic columns and charttime extraction variables
pt_on_vent_hourly.rename(columns={'starttime': 'charttime'}, inplace=True)

pt_on_vent_hourly = get_year_month_day_hour(pt_on_vent_hourly)

s3_dynamic_vital_vent['ventilation_missing'] = s3_dynamic_vital_vent.apply(lambda x: 1 if np.isnan(x['ventilation_status']) else 0, axis=1)

s3_dynamic_vital_vent['ventilation_status'] = s3_dynamic_vital_vent.groupby('subject_id')['ventilation_status'].ffill().fillna(0).astype(int)

s3_dynamic_vital_vent['norepinephrine_equivalent_dose'] = s3_dynamic_vital_vent['norepinephrine_equivalent_dose'].fillna(0)

# Static Variable
# encode gender into dummy variable. Female = 1, male = 0
s3_static = pd.get_dummies(s3_static, columns=['gender'])
s3_static = s3_static.drop(columns = 'gender_M')
s3_static = s3_static.rename(columns = {"gender_F":"gender"})

# rrt is NaN if dialysis_present != 1. Manually adjusting to 0
s3_static[['rrt']] = s3_static[['rrt']].fillna(value=0)

# taking median since BMI is not normal distribution
median_bmi = s3_static['bmi'].median()
na_bmi = s3_static['bmi'].isna()
s3_static.loc[na_bmi, 'bmi'] = median_bmi

# encode ethnicity into categorical variables.
# s3_static["ethnicity"].dtype
# categories=['AMERICAN INDIAN/ALASKA NATIVE', 'ASIAN',
# 'BLACK/AFRICAN AMERICAN', 'HISPANIC/LATINO', 'OTHER',
# 'UNABLE TO OBTAIN', 'UNKNOWN', 'WHITE']
s3_static["ethnicity"] = s3_static["ethnicity"].astype('category')
ethnicity_code = s3_static["ethnicity"].cat.codes
s3_static["ethnicity_cat"] = ethnicity_code

#### merging staic and device data
DEVICE_COL = ['subject_id', 'gcs', 'norepinephrine_equivalent_dose',
       'sofa_24hours', 'heart_rate', 'sbp_art', 'dbp_art', 'mbp_cuff', 'resp_rate', 'temperature', 'spo2',
       'glucose', 'ventilation_status', 'hour','ventilation_missing']
STATIC_COL = ['subject_id', 'deathtime', 'admission_age', 'charlson_comorbidity_index', 'apsiii', 'rrt', 'bmi',
       'gender', 'ethnicity_cat']

s3_device = s3_dynamic_vital_vent[DEVICE_COL]
s3_static_final = s3_static[STATIC_COL]

s3_device_static = s3_device.merge(s3_static_final, on = 'subject_id')

# Use forward and backward filling to fill NA
s3_device_static_filter = s3_device_static.ffill(axis = 0)
s3_device_static_filter = s3_device_static_filter.bfill(axis = 0)
s3_device_static_filter = s3_device_static_filter.ffill(axis = 0)

# hour info not needed for logistic regression
vital = ['subject_id','gcs', 'norepinephrine_equivalent_dose', 'sofa_24hours',
       'heart_rate', 'sbp_art', 'dbp_art', 'mbp_cuff', 'resp_rate',
       'temperature', 'spo2', 'glucose', 'ventilation_status','ventilation_missing']
static = ['subject_id', 'gender', 'admission_age','charlson_comorbidity_index', 'apsiii', 'rrt', 'bmi', 'ethnicity_cat']

# Only extract the 0th hour info for matching
s3_device_static_matching = s3_device_static_filter[s3_device_static_filter['hour'] == 0]

# White Vs Black
df_white_black = s3_device_static_matching[(s3_device_static_matching['ethnicity_cat'] == 2) |(s3_device_static_matching['ethnicity_cat'] == 7)]

selected_feature = ['subject_id','norepinephrine_equivalent_dose','ventilation_status','rrt','gcs', 'sofa_24hours',
       'heart_rate', 'sbp_art', 'dbp_art', 'mbp_cuff', 'resp_rate',
       'temperature', 'spo2', 'glucose', 'gender', 'admission_age','charlson_comorbidity_index',
                    'apsiii', 'bmi', 'ethnicity_cat','ventilation_missing']
## eliminate treatment variable
df_white_black_selected = df_white_black[selected_feature]

df_white_black_selected['race_white_black'] = df_white_black_selected.apply(lambda x: 1 if x['ethnicity_cat']==2 else 0,axis=1)

# Remove irregular glucose value according to suggestion
df_white_black_selected = df_white_black_selected[df_white_black_selected['glucose'] < 2000]

df_whole = df_white_black_selected[(df_white_black_selected['ventilation_status'] == 0) |(df_white_black_selected['ventilation_status'] == 1)| (df_white_black_selected['ventilation_status'] == 5)]

df_whole.to_csv("Black_vs_White.csv")
