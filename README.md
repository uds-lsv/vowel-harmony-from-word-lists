# Information-Theoretic Characterization of Vowel Harmony: A Cross-Linguistic Study on Word Lists

## Requirements
Implementation was done in a conda environment with the latest version of Python 3.8. After installing Anaconda or Miniconda, create a conda environment via:

```
conda create -n env python=3.8
conda activate env
```

Most required packages can be installed via the ```requirements.txt``` file calling pip recursively:
```bash
pip install -r requirements.txt
```

Unzip the `clts` zip folder from the OSF project or download version [2.2.0](https://zenodo.org/record/5583682)  CLTS:

```bash
wget https://github.com/cldf-clts/clts/archive/refs/tags/v2.2.0.zip
unzip v2.2.0.zip
```

Furthermore, version [0.9](https://zenodo.org/record/5121268) of the Lexibank version of the NorthEuraLex dataset has to be installed. It is also provided with some adjustments made to the Manchu data (these changes are also available when downloading the git tree rather than the release version). This is how you would obtain it from GitHub:

```bash
wget https://github.com/lexibank/northeuralex/archive/refs/tags/v4.0.zip
unzip v4.0.zip
cd northeuralex-4.0
pip install .
```

As soon as the dataset is installed, configure the path to the CLTS data by running ```cldfbench catconfig``` and then entering the absolute path into the config file in ```/home/$USER/.config/cldf/catalog.ini```:
```
[clones]
clts = /path/to/clts
```

Then, cd back to the location of this package and install it:

```bash
cd path/to/repo
pip install .
```

And you're set!

## Training

The models can be trained by running the notebook [train_nelex.ipynb](https://github.com/digling/eff/blob/8d302a195621773f52d475d1ea13de563b7ea190/notebooks/train_nelex.ipynb). The models along with the data used for training & testing will be saved automatically in ```notebooks/out/```. Otherwise you can use the pretrained models by downloading the ```nelex_unique``` zip folder and extracting it into the ```out``` folder. Only models for NELEX10 are provided.

## Analysis

One notebook performs the analysis for a single language. The results are output in the form of latex tables and plots (used with minimal changes in the thesis document itself). The table below maps the notebooks to the type of analysis:

|Notebook|Experiment|Description|
|:-|:-:|:-|
|[all.ipynb](https://github.com/digling/eff/blob/8d302a195621773f52d475d1ea13de563b7ea190/notebooks/all.ipynb)|Masking|Compares mean surprisal in the vowel-only and consonant-only condition for all languages in NorthEuraLex|
|[nelex10.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/nelex10.ipynb)|Masking|Evaluates surprisal in the masking experiments for the languages in NELEX10|
|[finnish.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/finnish.ipynb)|Harmony|Feature surprisal for Finnish +-BACK feature|
|[hungarian.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/hungarian.ipynb)|Harmony|Feature surprisal for Hungarian +-BACK feature|
|[turkish.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/turkish.ipynb)|Harmony|Feature surprisal for Turkish +-BACK and +-ROUND features|
|[manchu.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/manchu.ipynb)|Harmony|Feature surprisal for Manchu +-BACK feature|
|[khalkha_mongolian](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/khalkha_mongolian.ipynb)|Harmony|Feature surprisal for Khalkha Mongolian +-ATR and +-ROUND features|
|[non_vh_langs.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/non_vh_langs.ipynb)|Harmony|Feature surprisal for languages without vowel harmony, for +-BACK and +-ROUND features|
|[surprisal_reduction.iypnb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/surprisal_reduction.ipynb)||Plots difference between harmonic and disharmonic distribution for all feature-language combinations for NELEX10|
|[vh_vs_non_vh.ipynb](https://github.com/digling/eff/blob/15a7a73687005692fb21fa2f030f7ca590e31293/notebooks/vh_vs_non_vh.ipynb)||Plots mean differences for the 5 vowel harmony languages for +-BACK and +-ROUND _by feature_|
