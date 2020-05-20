# P_20_50
filtered from dataset ../twitter_2019/ according to:
- all politicians are kept
- followers and followees are filtered based on the lower threshold 20
- followers and followees are filtered based on the upper threshold 50
- 10000 political-issue outliers are randomly kept
- number of nodes = 12103
- number of links = 1976985
- number of relations = 5

## relations
- scale for picking 90% for training:
    ```shell
    args.relations = ['friend_list.csv', "reply_list.csv", "retweet_list.csv", "favorite_list.csv", "mention_list.csv"]
    ```
    - 134934
    - 102963
    - 505775
    - 478121
    - 458642