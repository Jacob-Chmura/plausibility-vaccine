import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Load the dataset from: https://github.com/google-deepmind/svo_probes
file_path = os.path.join('..', 'data', 'verb_understanding_data', 'svo_probes.csv')
df = pd.read_csv(file_path)

# Drop unnecessary columns (We may use neg_triplet in the future)
df = df.drop(columns=['sentence', 'neg_triplet', 'pos_url'])

verb_subject_counts = defaultdict(int)
verb_object_counts = defaultdict(int)
subject_counts = defaultdict(int)
object_counts = defaultdict(int)
verb_counts = defaultdict(int)

# Process the pos_triplet column
for triplet in df['pos_triplet']:
    if isinstance(triplet, str):
        subject, verb, obj = triplet.split(',')

        # Count occurrences
        verb_subject_counts[(verb, subject)] += 1
        verb_object_counts[(verb, obj)] += 1
        subject_counts[subject] += 1
        object_counts[obj] += 1
        verb_counts[verb] += 1

total_subjects = sum(subject_counts.values())
total_objects = sum(object_counts.values())

# Compute probabilities
p_s_given_v = {
    (verb, subject): count / verb_counts[verb]
    for (verb, subject), count in verb_subject_counts.items()
}
p_s = {subject: count / total_subjects for subject, count in subject_counts.items()}

p_o_given_v = {
    (verb, obj): count / verb_counts[verb]
    for (verb, obj), count in verb_object_counts.items()
}
p_o = {obj: count / total_objects for obj, count in object_counts.items()}


# Optionally we can save results to CSV
# pd.DataFrame.from_dict(p_s_given_v, orient="index", columns=["P(S|V)"]).to_csv("../data/results/kl-results/p_s_given_v.csv")
# pd.DataFrame.from_dict(p_s, orient="index", columns=["P(S)"]).to_csv("../data/results/kl-results/p_s.csv")
# pd.DataFrame.from_dict(p_o_given_v, orient="index", columns=["P(O|V)"]).to_csv("../data/results/kl-results/p_o_given_v.csv")
# pd.DataFrame.from_dict(p_o, orient="index", columns=["P(O)"]).to_csv("./data/results/kl-results/p_o.csv")


##Selectional Association

selectional_association_subject = defaultdict(float)
selectional_association_object = defaultdict(float)

# Compute S_R(v) for subjects
s_r_subject = {}
for verb in verb_counts.keys():
    s_r_subject[verb] = sum(
        p_s_given_v[(verb, subject)]
        * np.log(p_s_given_v[(verb, subject)] / p_s.get(subject, 1e-10))
        for (v, subject) in p_s_given_v.keys()
        if v == verb
    )

# Compute S_R(v) for objects
s_r_object = {}
for verb in verb_counts.keys():
    s_r_object[verb] = sum(
        p_o_given_v[(verb, obj)]
        * np.log(p_o_given_v[(verb, obj)] / p_o.get(obj, 1e-10))
        for (v, obj) in p_o_given_v.keys()
        if v == verb
    )

# Compute selectional association for subjects
for (verb, subject), p_s_v in p_s_given_v.items():
    if s_r_subject[verb] > 0:
        selectional_association_subject[(verb, subject)] = (
            p_s_v * np.log(p_s_v / p_s.get(subject, 1e-10)) / s_r_subject[verb]
        )

# Compute selectional association for objects
for (verb, obj), p_o_v in p_o_given_v.items():
    if s_r_object[verb] > 0:
        selectional_association_object[(verb, obj)] = (
            p_o_v * np.log(p_o_v / p_o.get(obj, 1e-10)) / s_r_object[verb]
        )

selectional_association_df_subject = pd.DataFrame.from_dict(
    selectional_association_subject, orient='index', columns=['Selectional_Association']
).reset_index()
selectional_association_df_subject.columns = ['Verb-Subject', 'Selectional_Association']

selectional_association_df_object = pd.DataFrame.from_dict(
    selectional_association_object, orient='index', columns=['Selectional_Association']
).reset_index()
selectional_association_df_object.columns = ['Verb-Object', 'Selectional_Association']

# File path

file_path_association = os.path.join('..', 'data', 'verb_understanding_data')
# Define paths for saving CSV files
selectional_association_subject_path = os.path.join(
    file_path_association, 'selectional_association_subject.csv'
)
selectional_association_object_path = os.path.join(
    file_path_association, 'selectional_association_object.csv'
)

selectional_association_df_subject.to_csv(
    selectional_association_subject_path, index=False
)
selectional_association_df_object.to_csv(
    selectional_association_object_path, index=False
)


##KL (Unused as of now)
def compute_KL():
    kl_subject = {}
    kl_object = {}

    # Compute KL Divergence for subjects
    for verb in verb_counts.keys():
        kl_s = 0
        for (v, subject), p_s_v in p_s_given_v.items():
            if v == verb:
                prob_s = p_s.get(subject, 0)
                if prob_s > 0:  # Avoid division by zero
                    kl_s += p_s_v * np.log(p_s_v / prob_s)
        kl_subject[verb] = kl_s

    # Compute KL Divergence for objects
    for verb in verb_counts.keys():
        kl_o = 0
        for (v, obj), p_o_v in p_o_given_v.items():
            if v == verb:
                prob_o = p_o.get(obj, 0)
                if prob_o > 0:  # Avoid division by zero
                    kl_o += p_o_v * np.log(p_o_v / prob_o)
        kl_object[verb] = kl_o

    kl_df = pd.DataFrame(
        {
            'Verb': kl_subject.keys(),
            'KL_Divergence_Subject': kl_subject.values(),
            'KL_Divergence_Object': kl_object.values(),
        }
    )

    kl_df.to_csv('kl_divergence_results.csv', index=False)
