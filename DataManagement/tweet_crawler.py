import tweepy
import time
import codecs
from collections import defaultdict

consumer_key= ''
consumer_secret = ''
auth_token = ''
auth_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(auth_token, auth_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def crawl_tweets():

    with open("C:/Users/Sumeet Singh/Documents/Code-Mixed/EMNLPTweetIds.csv", 'r') as fr:
        with codecs.open("C:/Users/Sumeet Singh/Documents/Code-Mixed/EMNLPTweets.csv", 'w', encoding='utf-8') as fw:
            tweet_ids = fr.read().split()
            while len(tweet_ids)>0:
                for n_ids in [tweet_ids[i:i+100] for i in range(0, len(tweet_ids), 100)]:
                    successDownload = False
                    while not successDownload:
                        try:
                            n_tweets = api.statuses_lookup(n_ids)
                            for tw in n_tweets:
                                fw.write(tw.text.strip() + '\n')
                            successDownload = True
                        except tweepy.TweepError:
                            print("Encountered Tweepy error... sleeping for 300")
                            # sleep for sometime to overcome request drop by twitter.
                            time.sleep(60 * 5)
                tweet_ids = fr.read().split()


# parses tweet TSV in CoNLL format
# returns dict of 2 lists (tweet texts and labels)
# lbl: list of labels for each tweet
def parse_twitter(tsvfile):
	txt_lst = defaultdict(list)
	lbl_lst = defaultdict(list)
	with open(tsvfile) as f:
		for line in f.readlines():
			all_info = line.replace('\n', '').split('\t')
			tweet_id = all_info[0]
			txt = all_info[4]
			lbl = all_info[5]

			txt_lst[tweet_id].append(txt)
			lbl_lst[tweet_id].append(lbl)
                
    return {'txt': txt_lst, 'lbl': lbl_lst}
                
if __name__=='__main__':
    crawl_tweets()
