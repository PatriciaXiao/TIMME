link files:
  friend_list.tsv
  mention_list.tsv
  retweet_list.tsv

  Each line is separated by tab ("\t"), and each field is a Twitter ID. 
  A  \t  B   means A follows/mentions/retweets from B.

  Mention and retweet files may contain duplicate lines, which means the 
  relation happens multiple times. 

dict file:
  dict.txt:   information for congress members. 

  Each line is separated by tab. 
  The first field is the real name (First name, Last name);
  The second field is the Twitter ID, corresponding to the IDs in link files.
  The third field is the Twitter screen name, as in @XXX.
  The last field is label. R = republican; D = democrat.

