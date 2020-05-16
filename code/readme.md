# GCN with relations

## What

- This code is adapted & simplified from [Spring 2019 version](../rGCN/src/)
- Works with Python3.7 and current latest dependencies

## Why

- The siplest code as a starting point of any modification.

## How

A proof that it works with reasonable performance:

```shell
python main.py -e 20 -d twitter_1hop
```

## Details
- entity-classification and link-prediction both implemented
- propose to have more efficient link prediction by multi-process sampling like in [Weiping's code](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/sequentialRec/markovChains/sampler.py)

## Scai Server issues

### CUDA availability
- **ScAi1** don't have GPUs, it only has CPUs. But using Docker you could do anything, require the resources you need.
- **ScAi2** is the one with GPUs. Please be careful not to block all the GPUs.
- If in your environment pytorch+CUDA support is not properly handled, consider using anaconda, and then:
    ```shell
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
    ```
    this is the current version of our lab's CUDA

