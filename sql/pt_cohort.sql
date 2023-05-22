select icu.*, charlson.*, apsiii, SOFA,
vaso_amount, weight_admit, height, (weight_admit / ((0.01*height) * (0.01*height))) AS bmi
-- remove weight and height, keeping only bmi next time.
from physionet-data.mimic_derived.icustay_detail as icu
--defines the sepsis population
inner join physionet-data.mimic_derived.sepsis3 as s3
on s3.stay_id = icu.stay_id

--study population filter
and icu.first_icu_stay is true
and icu.first_hosp_stay is true
and s3.sepsis3 is true

--Charlson
left join physionet-data.mimic_derived.charlson as charlson
on icu.hadm_id = charlson.hadm_id

--weight and height
left join physionet-data.mimic_derived.first_day_height as ht
on icu.stay_id = ht.stay_id --ensure only first stay
left join physionet-data.mimic_derived.first_day_weight as wt
on icu.stay_id = wt.stay_id

--APACHE III 
left join physionet-data.mimic_derived.apsiii as aps
on icu.stay_id = aps.stay_id

--SOFA
left join physionet-data.mimic_derived.first_day_sofa as sofa_d1
on icu.stay_id = sofa_d1.stay_id

--vasopressors
--only want one record per stay. vp may have multiple records, group by within to achieve single-record
left join (
  select icu.stay_id, avg(vaso_amount) as vaso_amount from physionet-data.mimic_derived.icustay_detail as icu
--defines the sepsis population
inner join physionet-data.mimic_derived.sepsis3 as s3
on s3.stay_id = icu.stay_id
--study population filter
and icu.first_icu_stay is true
and icu.first_hosp_stay is true
and s3.sepsis3 is true
--vaso meta info
left join physionet-data.mimic_derived.vasopressin as vp
on icu.stay_id = vp.stay_id
group by icu.stay_id --each stay has one vaso_amount
) as vp_table

on icu.stay_id = vp_table.stay_id

where icu.los_icu >= 1 --cohort filtering

ORDER BY icu.subject_id

 