from collections import defaultdict
import numpy as np
from pandas import DataFrame
from pyclts import CLTS
from scipy.stats import chi2_contingency
from typing import List

clts = CLTS()
bipa = clts.bipa


def get_sound_successions(words: List[str], sound_classes) -> DataFrame:
    successions = defaultdict(lambda: defaultdict(lambda: 0))
    for word in words:
        sounds = [segment for segment in word if bipa[segment].type in sound_classes]
        if len(sounds) > 1:
            for i in range(len(sounds)-1):
                s1 = bipa[sounds[i]]
                s2 = bipa[sounds[i+1]]
                # if not (s1.type == 'diphthong' or s2.type == 'diphthong'):
                successions[sounds[i]][sounds[i+1]] += 1
            # successions[sounds[0]][sounds[1]] += 1
    df_observed = DataFrame.from_dict(successions)
    df_observed.replace(np.NaN, 0, inplace=True)
    df_observed = df_observed.transpose()
    return df_observed


def get_phi_values(df: DataFrame, signed=False) -> DataFrame:
    contingency_table = chi2_contingency(df)[3]
    df_expected = DataFrame(
        data=contingency_table[:,:],
        index=df.index,
        columns=df.columns
    )

    df_phi = DataFrame()
    for idx1, sound1 in enumerate(df.columns):
        phis = []
        for idx2, sound2 in enumerate(df.index):
            A = df.loc[sound2, sound1]
            B = df.loc[:, sound1].sum() - A
            C = df.iloc[idx2].sum() - A
            D = df.sum().sum() - (A + B + C)
            # print(idx1, sound1, idx2, sound2, A, B, C, D)

            if signed:
                # Thomas Mayer 2010
                phi = (A*D-B*C) / (np.sqrt((A+C)*(B+D)*(A+B)*(C+D)))
            else:
                contingency_matrix = [[A, B],[C, D]]
                df_observed = DataFrame(data=contingency_matrix)
                contingency_table = chi2_contingency(contingency_matrix)[3]
                df_expected = DataFrame(data=contingency_table[:,:])
                df_diff = df_observed-df_expected
                chi2 = (np.square(df_diff)/df_expected).sum().sum()
                phi = np.sqrt(chi2/(A+B+C+D))

            phis.append(phi)

        df_phi.insert(0, sound1, phis)


    return df_phi