import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import scipy
from scipy import stats
import vonMisesMixtures as vonmises


def str_to_nparray(string):
    if not isinstance(string, str):
        return None
    from ast import literal_eval
    try:
        s = string.replace('\n', '').strip()
        s = s.replace('     ', ' ')
        s = s.replace('    ', ' ')
        s = s.replace('   ', ' ')
        s = s.replace('  ', ' ')
        s = s.replace('  ', ' ')
        s = s[:1] + s[2:]
        s = s.replace(' ', ',')
        lst = literal_eval(s)
    except:
        import code; code.interact(local=locals())
    return np.array(lst)

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def load_id_to_keep(list_pupil): # pupil data contain only 43 subjects
    id_tokeep = [ int(s.split('_')[-1].split('.')[0].split('l')[1]) for s in list_pupil ]
    return id_tokeep


data_path_fix = 'data/fixations_feat_OU_and_ROI/'
data_path_sac = 'data/saccades_feat_OU_and_ROI/'
path_Q = 'data/'
data_path_pupil = 'data/pupil_data/'

sbj_feats = []

X_lst = []
actually_kept = []

list_fixations = sorted_nicely(os.listdir(data_path_fix))
list_saccades = sorted_nicely(os.listdir(data_path_sac))
list_pupil = sorted_nicely(os.listdir(data_path_pupil))

id_tokeep = load_id_to_keep(list_pupil)


for index, s in enumerate(list_fixations): # Iterating over the various subjects
    curr_sbj = int(s.split('_')[-1].split('.')[0])

    if curr_sbj in id_tokeep:
        print('Subj: ' + str(curr_sbj))
        df_fix = pd.read_csv(data_path_fix+s)
        df_fix['feature'] = df_fix['feature'].apply(str_to_nparray)

        #Extracting fixation's durations list
        fix_dur = [df_fix['feature'][i][-1] for i in range(len(df_fix['feature'])) if df_fix['feature'][i] is not None]
        
        #Deleting the last feature from the vector
        df_fix['feature'] = [ x[:12] if x is not None else None for x in df_fix['feature'] ]

        grouped_fix = df_fix.groupby(['ROI'])['feature'].mean()
        eye_fix = grouped_fix['eye']
        mouth_fix = grouped_fix['mouth_nose']

        #Fitting a gamma distribution for fixations duration
        gamma_fix = list(stats.gamma.fit(fix_dur))

        df_sac = pd.read_csv(data_path_sac+list_saccades[index])
        df_sac['feature'] = df_sac['feature'].apply(str_to_nparray)

        #Extracting saccades's classic features lists
        sac_dur = [df_sac['feature'][i][-1] for i in range(len(df_sac['feature'])) if df_sac['feature'][i] is not None]
        sac_ampl = [df_sac['feature'][i][-2] for i in range(len(df_sac['feature'])) if df_sac['feature'][i] is not None]
        sac_angle = [df_sac['feature'][i][-3] for i in range(len(df_sac['feature'])) if df_sac['feature'][i] is not None]
        
        #Deleting the last three feature from the vector
        df_sac['feature'] = [ x[:12] if x is not None else None for x in df_sac['feature'] ]

        grouped = df_sac.groupby(['End_ROI'])['feature'].mean()
        eye_sac = grouped['eye']
        mouth_sac = grouped['mouth_nose']

        #Fitting gamma, levy and von mises distributions for saccades' durations, amplitude and angle
        gamma_sac = list(stats.gamma.fit(sac_dur))
        levy_sac = list(stats.levy_stable.fit(sac_ampl))
        von_sac = vonmises.mixture_pdfit( sac_angle , n=2).flatten().tolist()

        #Extracting pupil features
        df_pupil = pd.read_csv(data_path_pupil+'tmp_csv_tmp_pupil'+str(curr_sbj)+'.csv')
        pupil_list = df_pupil.values.flatten().tolist()
        lognorm_pupil = list(stats.lognorm.fit(pupil_list))

        df_sac['feature'] = [ x[:12] if x is not None else None for x in df_sac['feature'] ]

        print('eye_fix ', len(eye_fix))
        print('mouth_fix ', len(mouth_fix))
        print('gamma_fix ', len(gamma_fix))
        print('eye_sac ', len(eye_sac))
        print('mouth_sac ', len(mouth_sac))
        print('gamma_sac ', len(gamma_sac))
        print('von_sac ', len(von_sac))
        print('levy_sac ', len(levy_sac))
        print('lognorm_pupil ', len(lognorm_pupil))

        sbj = np.array([curr_sbj])
        svect = np.hstack([sbj, eye_fix, mouth_fix, gamma_fix,
                           eye_sac, mouth_sac, gamma_sac, levy_sac, von_sac,
                           lognorm_pupil])
        X_lst.append(svect)
        actually_kept.append(curr_sbj)


df = pd.DataFrame(X_lst)
df.to_csv('subject_features.csv')