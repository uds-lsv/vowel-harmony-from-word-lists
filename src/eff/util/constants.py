import torch


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = 'cpu'


sos = 'SOS'
eos = 'EOS'
pad = 'PAD'
mask = 'MASK'

marker = '+'

word_boundary = '_'

tones_mandarin = ['¹', '²', '³', '⁴', '⁵']

long_segment_marker = 'ː'
mid_long_segment_marker = 'ˑ'
wikt_stress_marker = 'ˈ'
wikt_secondary_stress_marker = 'ˌ'
wikt_velar_fricative_high = 'ˣ'
wikt_left_brace = '('
wikt_right_brace = ')'
wikt_semivowel_marker = '̯'

wikt_chars = [wikt_semivowel_marker, wikt_left_brace, wikt_right_brace, \
    wikt_stress_marker, wikt_secondary_stress_marker, wikt_velar_fricative_high]

color_encoding = {
    0: '#929591',
    1: '#069AF3',
    2: '#FC5A50'
}

# needed for legend
colors = {
    'None': '#929591',
    'Vowel Harmony': '#069AF3',
    'Umlaut': '#FC5A50'
}

random_seed = 0

lang_encoding = {
    'fin' : 1,
    'olo' : 1,
    'vep' : 1,
    'ekk' : 0,
    'krl' : 1,
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