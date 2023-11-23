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

def get_ou_fix_features(df_fix):
    #Extracting fixation's durations list
    fix_dur = [(df_fix['Trial'][i], df_fix['feature'][i][-1]) for i in range(len(df_fix['feature'])) if df_fix['feature'][i] is not None]

    #Deleting the last feature from the vector
    df_fix['feature'] = [ x[:12] if x is not None else None for x in df_fix['feature'] ]

    grouped_fix = df_fix.groupby(['Trial', 'ROI'])['feature'].mean().reset_index()

    group_counts = grouped_fix.groupby(['Trial']).size().reset_index(name='count')
    valid_trials = group_counts[group_counts['count'] >= 2]['Trial']
    grouped_fix = grouped_fix[grouped_fix['Trial'].isin(valid_trials)]

    result = {}
    for t in valid_trials.tolist():
        try:
            features0 = grouped_fix[grouped_fix['Trial'] == t].iloc[0]['feature']
            features1 = grouped_fix[grouped_fix['Trial'] == t].iloc[1]['feature']
            concatenated_features = np.concatenate([features0, features1])
            result[t] = concatenated_features
        except Exception as e:
            print(f"An error occurred for Trial {t}: {e}")

    return result, fix_dur, valid_trials

def get_ou_sac_features(df_sac):
    # Extracting saccades's classic features lists
    sac_dur = [ (df_sac['Trial'][i], df_sac['feature'][i][-1]) for i in range(len(df_sac['feature'])) if df_sac['feature'][i] is not None]
    sac_ampl = [ (df_sac['Trial'][i], df_sac['feature'][i][-2]) for i in range(len(df_sac['feature'])) if df_sac['feature'][i] is not None]
    sac_angle = [(df_sac['Trial'][i], df_sac['feature'][i][-3]) for i in range(len(df_sac['feature'])) if df_sac['feature'][i] is not None]

    # Deleting the last three feature from the vector
    df_sac['feature'] = [ x[:12] if x is not None else None for x in df_sac['feature'] ]

    grouped_sac = df_sac.groupby(['Trial','End_ROI'])['feature'].mean().reset_index()
    grouped_sac = grouped_sac[grouped_sac['End_ROI'] != "out"]
    group_counts = grouped_sac.groupby(['Trial']).size().reset_index(name='count')
    valid_trials = group_counts[group_counts['count'] >= 2 ]['Trial']
    grouped_sac = grouped_sac[ grouped_sac['Trial'].isin(valid_trials) ]

    result = {}
    for t in valid_trials.tolist():
        try:
            features0 = grouped_sac[grouped_sac['Trial'] == t].iloc[0]['feature']
            features1 = grouped_sac[grouped_sac['Trial'] == t].iloc[1]['feature']
            concatenated_features = np.concatenate([features0, features1])
            result[t] = concatenated_features
        except Exception as e:
            print(f"An error occurred for Trial {t}: {e}")

    return result, sac_dur, sac_ampl, sac_angle, valid_trials


def get_pupil_lists(curr_sbj, valid_trials):
    pupil_lists = []

    pupil_dilation = pd.read_csv(data_path_pupil+'tmp_csv_tmp_pupil'+str(curr_sbj)+'.csv')
    timestamps = np.load(data_path_gazetime+'gaze_data_timeLook_'+str(curr_sbj)+'.npy', allow_pickle=True)

    for t in valid_trials:
        pupil_lists.append((t, pupil_dilation.iloc[int(t)]))

    return pupil_lists


data_path_fix = 'data/fixations_feat_OU_and_ROI/'
data_path_sac = 'data/saccades_feat_OU_and_ROI/'
data_path_gazetime = 'data/gazetime/'
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

        # FIXATIONS
        ou_fix, fix_dur, valid_trials_fix = get_ou_fix_features(df_fix)

        df_fix_dur = pd.DataFrame(fix_dur, columns=['Trial', 'Value'])
        # Group the data by 'Trial'
        groups_fix_dur = df_fix_dur.groupby('Trial')

        # Fit a gamma distribution to each group
        gamma_fix = {}
        for name, group in groups_fix_dur:
            try:
                gamma_fix[name] = list(scipy.stats.gamma.fit(group['Value'].tolist()))
            except:
                continue


        # SACCADES

        df_sac = pd.read_csv(data_path_sac+list_saccades[index])
        df_sac['feature'] = df_sac['feature'].apply(str_to_nparray)

        ou_sac, sac_dur, sac_ampl, sac_angle, valid_trials_sac = get_ou_sac_features(df_sac)

        df_sac_dur = pd.DataFrame(sac_dur, columns=['Trial', 'Value'])
        # Group the data by 'Trial'
        groups_sac_dur = df_sac_dur.groupby('Trial')

        df_sac_ampl = pd.DataFrame(sac_ampl, columns=['Trial', 'Value'])
        # Group the data by 'Trial'
        groups_sac_ampl = df_sac_ampl.groupby('Trial')

        df_sac_angle = pd.DataFrame(sac_angle, columns=['Trial', 'Value'])
        # Group the data by 'Trial'
        groups_sac_angle = df_sac_angle.groupby('Trial')


        # Fitting gamma, levy and von mises distributions for saccades' durations, amplitude and angle

        gamma_sac = {}
        for name, group in groups_sac_dur:
            try:
                gamma_sac[name] = list(scipy.stats.gamma.fit(group['Value'].tolist()))
            except:
                continue


        """levy_sac = {}
        for name, group in groups_sac_ampl:
            try:
               levy_sac[name] = list(scipy.stats.levy_stable.fit(group['Value'].tolist()))
            except:
               continue

            
        vonmises_sac = {}
        for name, group in groups_sac_ampl:
            try:
                vonmises_sac[name] = vonmises.mixture_pdfit( group['Value'].tolist() , n=2).flatten().tolist()
            except:
                continue


        # Extracting pupil features
        valid_trials = set(valid_trials_sac).intersection(set(valid_trials_fix))
        pupil_lists = get_pupil_lists(curr_sbj, valid_trials)

        lognorm_pupil = {}
        for t, pupil_list in pupil_lists:
            lognorm_pupil[t] = list(stats.lognorm.fit(pupil_list))
        """
        
        valid_trials = set(valid_trials_sac).intersection(set(valid_trials_fix))

        sbj = np.array([curr_sbj])
        for trial in valid_trials:
            #if trial in ou_fix and trial in gamma_fix and trial in ou_sac and trial in gamma_sac and trial in lognorm_pupil and trial in levy_sac and trial in vonmises_sac:
            #    svect = np.hstack([ sbj, trial, ou_fix[trial], gamma_fix[trial],
            #                        ou_sac[trial], gamma_sac[trial], levy_sac[trial], vonmises_sac[trial],
            #                        lognorm_pupil[trial] ])
            #    X_lst.append(svect)

            if trial in ou_fix and trial in gamma_fix and trial in ou_sac and trial in gamma_sac:
                svect = np.hstack([ sbj, trial, ou_fix[trial], gamma_fix[trial],
                                    ou_sac[trial], gamma_sac[trial]])
                X_lst.append(svect)




        #print('X_lst ', X_lst)
        #print('lognorm_pupil ', lognorm_pupil)
        #print('eye_fix ', len(eye_fix))
        #print('mouth_fix ', len(mouth_fix))
        #print('gamma_fix ', len(gamma_fix))
        #print('eye_sac ', len(eye_sac))
        #print('mouth_sac ', len(mouth_sac))
        #print('gamma_sac ', len(gamma_sac))
        #print('von_sac ', len(von_sac))
        #print('levy_sac ', len(levy_sac))
        #print('lognorm_pupil ', len(lognorm_pupil))

        

df = pd.DataFrame(X_lst)
df.to_csv('trial_features.csv')