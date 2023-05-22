select icu.subject_id, charttime, sepsis3, heart_rate, sbp, dbp, sbp_ni, dbp_ni, resp_rate, temperature, spo2, glucose
from physionet-data.mimic_derived.icustay_detail as icu
--defines the sepsis population
inner join physionet-data.mimic_derived.sepsis3 as s3
on s3.stay_id = icu.stay_id
and icu.first_icu_stay is true
and icu.first_hosp_stay is true
and s3.sepsis3 is true
-- extracting vital signs
left join physionet-data.mimic_derived.vitalsign as vs
on icu.stay_id = vs.stay_id

WHERE icu.los_icu >= 1
ORDER BY icu.subject_id, charttime

 