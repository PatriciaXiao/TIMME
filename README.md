# TIMME

This repository contains the code implementation and the data of *TIMME: Twitter Ideology-detection via Multi-task Multi-relational Embedding*.

Please cite the [paper](./TIMME_for_KDD2020_cameraready.pdf) to use the [dataset](./data/).
```

```

## Environment

Language: Tested on Python 3.6, 3.7 or 3.8. All worked well.

Pre-requisites (other versions might apply as well, these are the developing environment we've used):

| Python | torch | pandas | numpy  | scipy | scikit-learn |
|:------:| :---: | :----: | :----: | :---: | :----------: |
| 3.8    | 1.4.0 |  1.0.3 | 1.18.2 | 1.4.1 |  0.23.1      |
| 3.7    | 1.4.0 | 0.6.3  | 1.17.2 | 1.3.1 |  0.20.2      |
| 3.6    | 1.3.1 | 0.23.4 | 1.15.4 | 1.1.0 |  0.20.2      |

All other dependencies should be automatically installed once you get these packages installed. Otherwise please follow the instruction to install missing packages, and welcome to give us feedback if we gave the wrong version.

## Related Reporitories

Baseline models, and the geography visualization code, are given in separate repositories.

### Baseline Models

Pending: repository to be released by Haoyan.

by Haoyan Xu

### Geography Visualization

Geography visualization tasks envolve two parts: first, interprete the location from the chaotic Twitter user profiles; second, plot the maps according to our results.

* [TIMME-formatted-location](https://github.com/franklinnwren/TIMME-formatted-location)
* [TIMME-data-visualization](https://github.com/franklinnwren/TIMME-data-visualization)

by Zhicheng Ren.

## Usage

### Sample Usage


### Hyperparameters

#### The variations of the model

- TIMME-single
- TIMME
- TIMME-hierarchical

| Model Name (in paper) | Command Line Flag (in [main](./code/main.py)) | Model Name (in [model](./code/model/model.py)) |
|:---------------------:| :-------------------------: | : -----------------: | : ------------------:|
| N/A                   | Classification              | ClassificationTask   | Classification       |
| N/A                   | LinkPrediction              | LinkPredictionTask   | LinkPrediction       |
| TIMME-single          | TIMME_SingleLink            | TIMMEManager         | TIMMEsingle          |
| TIMME                 | TIMME                       | TIMMEManager         | TIMME                |
| TIMME-hierarchical    | TIMME_hierarchical          | TIMMEManager         | TIMMEhierarchical    |

