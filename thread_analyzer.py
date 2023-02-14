import pandas as pd
from transformers import pipeline
from scraper import getComments


def chunk(a):
    n = round(0.2 * len(a))
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def main():

    durl = "https://www.reddit.com/r/news/comments/111lv6d/there_were_more_toxic_chemicals_on_train_that/"

    # here you would probably check if the post id already exists in some DB
    # so that you don't have to refetch comments.
    # if pushshift.io submission comments api starts working again,
    # could probably make this all realtime.

    # df = getComments(url=durl)
    #
    df = pd.read_csv("111lv6d_comments.csv")

    smax = df.score.max()

    threshold = round(0.1 * smax)

    df = df[df.score >= threshold]

    if len(df.text) >= 200:
        df = df[:200]

    # this is to deal with giving the model too large of an input which makes things very slow
    chunked = list(chunk(df.text))

    nlp = pipeline('summarization')

    lst_summaries = []

    for grp in chunked:
        result = nlp(grp.str.cat(), max_length=500)[0]["summary_text"]
        lst_summaries.append(result)

    ntext = ' '.join(lst_summaries)

    thread_summary = nlp(ntext, max_length=500)[0]["summary_text"].replace(" .", ".")

    print(thread_summary)


if __name__ == "__main__":
    main()
