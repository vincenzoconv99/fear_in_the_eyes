# Using ML on Gaze Data in a Fear Generalization setup

The system performs prediction of subject- or trial-related information, i.e., Social Anxiety (SIAS), Linear Deviation Score (LDS), Electrotactile stimulus presence, and Shock Expectancy Rating, based on eye movements modeled as an Ornstein-Uhlenbeck process.

## Installation

Install required libraries with Anaconda:

```bash
conda create --name mlgaze -c conda-forge --file requirements.txt
conda activate mlgaze
```
Install [NSLR-HMM](https://gitlab.com/nslr/nslr-hmm)

```bash
python -m pip install git+https://gitlab.com/nslr/nslr
```

Install [Mixture of von Mises disttributions](https://framagit.org/fraschelle/mixture-of-von-mises-distributionss)

```bash
git clone https://framagit.org/fraschelle/mixture-of-von-mises-distributions.git mixture
```

### Features extraction
Extract event-related Ornstein-Uhlenbeck features from [Diagnostic Facial Features & Fear Generalization dataset](https://osf.io/4gz7f/) (`datasets/Reutter`) launching the module `extract_OU_params.py`, results will be saved in `features/Reutter_OU_posterior_VI`.

After that, extract the subject- and trial-related features by launching the modules `extract_subject_features.py` and `extract_trial_features.py` (which average the event-related features extracted in the step before).


### Train and test
Modules `clf_predict_SIAS.py`, `clf_predict_LDS.py`, `clf_predict_SHOCK_EXP.py` and `clf_predict_STIMULUS.py`  exploit different classifiers for labels prediction on the extracted features.
