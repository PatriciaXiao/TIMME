# PureP
filtered from dataset ../twitter_2019/ according to:
- all politicians are kept
- randomly-selected followers and followees are all kept
- 0 political-issue outliers are randomly kept
- number of nodes = 583
- number of links = 122347
- number of relations = 5

## relations
- scale for picking 90% for training:
    ```shell
    args.relations = ['friend_list.csv', "reply_list.csv", "retweet_list.csv", "favorite_list.csv", "mention_list.csv"]
    ```
    - 50212
    - 1233
    - 16795
    - 12223
    - 23529