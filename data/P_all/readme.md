# P_all
filtered from dataset ../twitter_2019/ according to:
- all politicians are kept
- followers and followees are filtered based on the lower threshold 20
- 3000 political-issue outliers are randomly kept
- number of nodes = 20811
- number of links = 6496107
- number of relations = 5

## relations
- relation counts:
    ```shell
    args.relations = ['friend_list.csv', "reply_list.csv", "retweet_list.csv", "favorite_list.csv", "mention_list.csv"]
    ```
    - 915438
    - 530598
    - 1684023
    - 1794111
    - 1571937

## About **features.npz**

That file was too big to be uploaded onto github, so I did:
```shell
zip features.zip --out features.npz.zip -s 50m
```
where features.zip is a folder containing **features.npz**.

When unzipping it, please unzip **features.npz.zip**, and then you'll see a folder containing **features.npz**.