from matplotlib import pyplot as plt
import numpy as np
from typing import List

# 1 = VH
# 2 = Umlaut
# 3 = Neither VH nor Umlaut

lang2vh = {
    'fin' : 1,
    'krl' : 1,
    'olo' : 1,
    'vep' : 1,
    'ekk' : 0,
    'liv' : 0,
    'sma' : 2,
    'smj' : 2,
    'sme' : 2,
    'smn' : 2,
    'sms' : 2,
    'sjd' : 2,
    'mrj' : 1,
    'mhr' : 1,
    'mdf' : 1,
    'myv' : 1,
    'udm' : 0,
    'koi' : 0,
    'kpv' : 0,
    'hun' : 1,
    'kca' : 1,
    'mns' : 1,
    'sel' : 0,
    'yrk' : 0,
    'enf' : 0,
    'nio' : 1,
    'ben' : 0,
    'hin' : 0,
    'pbu' : 0,
    'pes' : 0,
    'kmr' : 0,
    'oss' : 0,
    'hye' : 0,
    'ell' : 0,
    'sqi' : 0,
    'bul' : 0,
    'hrv' : 0,
    'slv' : 0,
    'slk' : 0,
    'ces' : 0,
    'pol' : 0,
    'ukr' : 0,
    'bel' : 0,
    'rus' : 0,
    'lit' : 0,
    'lav' : 0,
    'isl' : 2,
    'nor' : 0,
    'swe' : 0,
    'dan' : 0,
    'deu' : 2,
    'nld' : 0,
    'eng' : 0,
    'gle' : 2,
    'cym' : 2,
    'bre' : 0,
    'lat' : 0,
    'fra' : 0,
    'cat' : 0,
    'spa' : 0,
    'por' : 0,
    'ita' : 0,
    'ron' : 0,
    'tur' : 1,
    'azj' : 1,
    'uzn' : 1,
    'kaz' : 1,
    'bak' : 1,
    'tat' : 1,
    'sah' : 1,
    'chv' : 1,
    'khk' : 1,
    'bua' : 1,
    'xal' : 1,
    'evn' : 1,
    'mnc' : 1,
    'gld' : 1,
    'ket' : 0,
    'ykg' : 0,
    'yux' : 0,
    'itl' : 0,
    'ckt' : 1,
    'niv' : 0,
    'ain' : 0,
    'kor' : 1,
    'jpn' : 0,
    'ale' : 0,
    'ess' : 0,
    'kal' : 0,
    'kan' : 0,
    'mal' : 0,
    'tam' : 0,
    'tel' : 1,
    'bsk' : 0,
    'kat' : 0,
    'eus' : 0,
    'abk' : 0,
    'ady' : 0,
    'ava' : 0,
    'ddo' : 0,
    'lbe' : 0,
    'lez' : 0,
    'dar' : 0,
    'che' : 0,
    'arb' : 0,
    'heb' : 0,
    'cmn' : 0
}

color_scheme = {
    0 : 'r',
    1 : 'b',
    2 : 'g'
}

# needed for legend
label2color = {
    'None': 'r',
    'Vowel Harmony': 'b',
    'Umlaut': 'g'
}


# def plot_phi(result):
#     plt.figure(figsize=(18, 12))

#     results_sorted = dict(sorted(result.items(), key=lambda item: item[1]))
#     languages = list(results_sorted.keys())
#     for x, (lang, phi_vowel) in enumerate(results_sorted.items()):
#         color = color_scheme[lang2vh[lang]]
#         labels = ['None', 'Umlaut', 'Vowel Harmony']
#         handles = [plt.Rectangle((0,0),1,1, color=label2color[label]) for label in labels]
#         plt.bar(x, phi_vowel, color=color)
#     plt.xticks(np.arange(len(languages)), languages, rotation=90)
#     plt.legend(handles, labels)
#     plt.show()


def set_lang_ax(axis, data: List[List[float]], xlabels: List[str], ylabel: str, title: str, boxcolor: str, \
    fontsize: int, markercolor='black'):
    """ Plots surprisal values in boxplots for a specific language. 

    Parameters
    ----------
    axis : 
        The pyplot axis object to use for plotting
    data : List[List[float]]
        List of Lists of surprisal values for different conditions
    xlabels : List[str]
        List of condition labels
    ylabel : str
        Either 'feature surprisal' or 'average surprisal'
    title : str
        Plot title
    boxcolor : str
        Internal color of the boxes
    markercolor : str, optional
        Color of box edges, whiskers, bars, by default 'black'
    """
    assert len(data) == len(xlabels) # suprisal distributions and number of conditions have to be identical.
    bp = axis.boxplot(data, patch_artist=True, showmeans=True)
    for element in ['boxes', 'whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black', linewidth=3)
    plt.setp(bp['boxes'], facecolor=boxcolor)
    plt.setp(bp['medians'], linestyle='-', linewidth=3, color=markercolor)
    plt.setp(bp['means'], linestyle='-', marker='o', markerfacecolor=markercolor, markeredgecolor=markercolor,
        markersize=8)
    axis.set_ylabel(ylabel, fontsize=fontsize)
    axis.tick_params(axis='y', labelsize=fontsize)
    axis.set_xticklabels(xlabels, fontsize=fontsize)
    axis.set_title(title, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

