# twitter_2019_20
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