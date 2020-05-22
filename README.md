# TIMME

This repository contains the code implementation and the data of *TIMME: Twitter Ideology-detection via Multi-task Multi-relational Embedding*.

Please cite the [paper](./TIMME_for_KDD2020_cameraready.pdf) to use the [dataset](./data/).
```

```

## Environment


## Usage

## Related Reporitories

### Baseline Models

Pending: repository to be released by Haoyan.

by Haoyan Xu

### Geography Visualization

Geography visualization tasks envolve two parts: first, interprete the location from the chaotic Twitter user profiles; second, plot the maps according to our results.

* [TIMME-formatted-location](https://github.com/franklinnwren/TIMME-formatted-location)
* [TIMME-data-visualization](https://github.com/franklinnwren/TIMME-data-visualization)

by Zhicheng Ren.

## Hyperparameters

### The variations of the model

- TIMME-single
- TIMME
- TIMME-hierarchical

| Model Name (in paper) | Task Name (in code) |
|:---------------------:| :-----------------: |
| N/A                   | Classification      |
| N/A                   | LinkPrediction      |
| TIMME-single          | SingleLink          |
| TIMME                 | MultiTask           |
| TIMME-hierarchical    | MultitaskConcat     |

