import pandas as pd
from transformers import pipeline
from scraper import getComments


def main():

    # durl = "https://www.reddit.com/r/ask/comments/111591k/who_do_you_think_will_start_ww3/"

    # here you would probably check if the post id already exists in some DB so that you don't have to refetch comments.
    # if pushshift.io submission comments api starts working again, could probably make this all realtime.

    # df = getComments(url=durl)
    nlp = pipeline('summarization')
    df = pd.read_csv("111591k_comments.csv")
    gb = df.groupby("parent_id")

    for key, grp in gb:

        summary = nlp(grp.text.str.cat(), max_length=500)[0]["summary_text"]

        print(summary)
        break

if __name__ == "__main__":
    #     ldf = pd.read_csv('news_comments.csv')

    # pid = ldf.post_id.unique()[0]

    # df = ldf[ldf.post_id == pid]

    # txt = df.body[0:2].str.cat()

    # nlp = pipeline('summarization')
    # df.body[4] = ''

    # text = ' '.join(df.body[0:32])
    # summary1 = nlp(text, max_length=500)[0]["summary_text"]

    # text = ' '.join(df.body[33:60])
    # summary2 = nlp(text, max_length=500)[0]["summary_text"]

    # text = ' '.join(df.body[61:90])
    # summary3 = nlp(text, max_length=500)[0]["summary_text"]


    # summary = summary1 + ' ' + summary2 + ' ' + summary3

    # nsum = nlp(summary, max_length=500)[0]["summary_text"]

    # print ("Original Text")

    # print(summary)

    # print("Summarised")

    main()
