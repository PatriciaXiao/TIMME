# P50
filtered from dataset ../twitter_2019/ according to:
- all politicians are kept
- followers and followees are filtered based on the lower threshold 50
- 0 political-issue outliers are randomly kept
- number of nodes = 5435
- number of links = 1593721
- number of relations = 5

## relations
- counts:
    ```shell
    args.relations = ['friend_list.csv', "reply_list.csv", "retweet_list.csv", "favorite_list.csv", "mention_list.csv"]
    ```
    - 529448
    - 96757
    - 311359
    - 302571
    - 353586