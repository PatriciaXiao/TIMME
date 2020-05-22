# TIMME Dataset Documentation

## Relational Dataset (PureP, P50, P_20_50, P_all)

These datasets are of the same format, corresponding to the four dataset we've used in our paper.

To use our dataset and/or code, please cite our paper
```

```

The components in our datasets:
* dict.csv: the ground-truth labels of the politicians.
* friend_list.csv: the relation of A follow B.
* favorite_list.csv: the relation of A like B's tweet.
* mention_list.csv: the relation of A mention B in her/his tweet (retweeting's automatic @ is excluded).
* reply_list.csv: the relation of A replying to B's tweet.
* retweet_list.csv: the relation of A retweet from B. This retweet include retweeting with / without comments. In other places, *retweet with comment* is sometimes referred to as "quote". The reason why we treated them the same is that, from a Twitter user's side these two options are almost the same.
* tweet_features.npz
* features.npz

All relations are in the format of ```<from>	<to>	<count>```, where *from* is where this relation starts, *to* is whom involved on the other side, *count* is how many times these two person have this relationship in between in the time window we observe. (**note:** all relations are directed --- from A to B and from B to A are treated differently.)

Something to note is that, in order to safely contianing the politicians' names and probably some special characters, we decided to use ```\t``` as our separator, for almost all the data files we use.

## Additional Labels

Additional labels are extracted by reading the [user profile information](./data/simplified_user_info.json)'s **description** part, done by [Zhicheng Ren](https://github.com/franklinnwren) manually.

The additional labels we have are stored under [additional_labels](./additional_labels) folder.

## Formatted Location

Locations are extracted from [user profile information](./data/simplified_user_info.json) by [Zhicheng Ren](https://github.com/franklinnwren).

The location information associated with the accounts are self-reported by some of the Twitter users. It is neither mandatory nor fully-reliable. Zhicheng cross-check the self-reported locations with the USA states and city names, wrote an automatic script of extracting their locations. The result is as shown in [formatted_location](./formatted_location).

## Yupeng's Dataset

Yupeng's dataset is collected by [Yupeng Gu](https://scholar.google.com/citations?user=11jDFV8AAAAJ&hl=en) at around 2013, 2014.

The data size is much smaller than ours, relation types are fewer, but the format is mostly the same. To understand and use his data, please read [his paper](https://arxiv.org/abs/1612.08207), and cite it:
```
@article{gu2016ideology,
  title={Ideology detection for twitter users with heterogeneous types of links},
  author={Gu, Yupeng and Chen, Ting and Sun, Yizhou and Wang, Bingyu},
  journal={arXiv preprint arXiv:1612.08207},
  year={2016}
}
```