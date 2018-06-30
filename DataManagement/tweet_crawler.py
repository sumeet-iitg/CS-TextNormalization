import tweepy
import time
import codecs
from collections import defaultdict
from filters import tweetFilterCollection, dumbFilterCollection
import emoji

consumer_key= 'H7fTnyqbP8VWASr2R6hofb1kF'
consumer_secret = 'IHQvllXFss7RWpy5fxAOdByDBcJG7kU2Dgy9zOVBKDsVqlYb09'
auth_token = '175123301-vLCEEA8vuKmPw4vSIFnrsSe0IGi2Qq2kHRMHfZdD'
auth_token_secret = 'a2rZpZZVuWpdtBB6X2pT4y1EFfqW7a0DwyFPmYF9hUC1R'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(auth_token, auth_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

def crawl_tweets():
    with open("C:/Users/Sumeet Singh/Documents/Code-Mixed/Train-MSA-EGY-2018-TweetIds.csv", 'r') as fr:
        with codecs.open("C:/Users/Sumeet Singh/Documents/Code-Mixed/EngSPATweets.csv", 'w', encoding='utf-8') as fw:
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

def crawl_unique_tweets():
    with open("C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\train_eng_spa_tweetIds.csv", 'r') as fr:
        with codecs.open("C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\EngSPATweets.csv", 'w', encoding='utf-8') as fw:
            tweet_ids = fr.read().split()
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

def parse_twitter(tsvfile):
        txt_lst = defaultdict(list)
        lbl_lst = defaultdict(list)
        with codecs.open(tsvfile, encoding='utf-8') as f:
            for line in f.readlines():
                all_info = line.replace('\n', '').split('\t')
                tweet_id = all_info[0]
                txt = all_info[4]
                lbl = all_info[5]
                txt_lst[tweet_id].append(txt)
                lbl_lst[tweet_id].append(lbl)
        return txt_lst, lbl_lst

'''
This was written to remove consecutive duplicate words in tweets that are in Conll format.
And also output a file with each containing a single tweet in each line. 
'''
def dedupAndReconstructTweets(originalTweetConll, dedupedTweetConll, reconstructedTweetTsv):
    txt_lst, lbl_lst = parse_twitter(originalTweetConll)
    with codecs.open(dedupedTweetConll, 'w', encoding='utf-8') as dupFp, \
        codecs.open(reconstructedTweetTsv, 'w', encoding='utf-8') as rectFp:
        for tweetId in txt_lst.keys():
            word_list = txt_lst[tweetId]
            label_list = lbl_lst[tweetId]
            prevWord = ""
            deduped_word_list = []
            for word,label in zip(word_list,label_list):
                if prevWord == word:
                    continue
                else:
                    dupFp.write(tweetId + "\t" + word + "\t" + label + "\n")
                    deduped_word_list.append(word)
                    prevWord = word
            rectFp.write(tweetId + "\t"+ " ".join(deduped_word_list) + "\n")

'''
Word level filter to identify an emoticon 
'''
def extract_emojis(word):
    for char in word:
        if char in emoji.UNICODE_EMOJI:
            word = "<emoticon>"
            break
    return word

'''
Wrote this function to apply the normalization filters at the word level 
'''
def applyFiltersToWord(word, filterCollection):
    for filter in filterCollection.filters:
        word = filter(word)
    return word

if __name__=='__main__':
    # crawl_tweets()
    # crawl_unique_tweets()

    # txt, lbl = parse_twitter("C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_train.tsv")
    # with codecs.open("C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\EngSPATweets.csv", 'w', encoding='utf-8') as fw:
    #     for tweetId, wordList in txt.items():
    #         sentence = " ".join(wordList)
    #         fw.write(sentence + "\n")

    originalConllFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_train.tsv"
    filteredConllFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_train_filtered.tsv"

    devConllFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_dev.tsv"
    filteredDevConllFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_dev_filtered.tsv"

    # apply filters to the conll file
    dumbFilters = dumbFilterCollection()
    with codecs.open(devConllFile, encoding='utf-8') as ddFp, \
        codecs.open(filteredDevConllFile, 'w', encoding='utf-8') as fFp:
        for line in ddFp.readlines():
            words = line.strip().split()
            filteredWord = extract_emojis(applyFiltersToWord(words[4],dumbFilters))
            words.append(filteredWord)
            fFp.write("\t".join(words) + '\n')

    '''
    Test file format was a little different from train and dev, 
    as in the word wasn't present in the same column. So repeating similar process as above
    '''
    testConllFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_test.conll"
    filteredTestFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\eng-spa-ner\\calcs_test_filtered.conll"
    with codecs.open(testConllFile, encoding='utf-8') as ddFp, \
        codecs.open(filteredTestFile, 'w', encoding='utf-8') as fFp:
        for line in ddFp.readlines():
            if line.isspace():
                fFp.write(line)
            else:
                word = line.strip()
                filteredWord = extract_emojis(applyFiltersToWord(word,dumbFilters))
                fFp.write(word + "\t" + filteredWord + '\n')

'''
# file with repeated string and NER within a tweet removed
    dedupedConllFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\calcs_train_deduped.tsv"

    # file containing a whole tweet per line
    reconstructedTsv = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\calcs_train_reconstructed.tsv"
    dedupAndReconstructTweets(originalConllFile, dedupedConllFile, reconstructedTsv)

    # normalized tweets per line joined with their tweet id
    filteredFile = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\calcs_train_filtered.tsv"
    filteredConll = "C:\\Users\\Sumeet Singh\\Documents\\Code-Mixed\\calcs_train_filtered_conll.tsv"
    parse_tweet_conll(dedupedConllFile, filteredFile,filteredConll)
'''
