import os
import re
import sys
import nltk
import praw
import gradio as gr
import pandas as pd
import praw.exceptions
from transformers import pipeline


def index_chunk(a):
    n = round(0.3 * len(a))
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def sentence_chunk(a):
    sentences = []
    buffer = ""

    # the 512 token threshold is empirical

    for item in a:
        token_length_estimation = len(nltk.word_tokenize(buffer + item))

        if token_length_estimation > 512:
            sentences.append(buffer)
            buffer = ""

        buffer += item

    sentences.append(buffer)

    return sentences


def preprocessData(df):
    df["text"] = df["text"].apply(lambda x: re.sub(r"http\S+", "", x, flags=re.M))
    df["text"] = df["text"].apply(lambda x: re.sub(r"^>.+", "", x, flags=re.M))

    smax = df.score.max()

    threshold = round(0.05 * smax)

    df = df[df.score >= threshold]

    # empirically, having more than 200 comments doesn't change much but slows down the summarizer.
    if len(df.text) >= 200:
        df = df[:200]

    # chunking to handle giving the model too large of an input which crashes

    # chunked = list(index_chunk(df.text))
    chunked = sentence_chunk(df.text)

    return chunked


def getComments(url, debug=False):

    if debug and os.path.isfile('./debug_comments.csv'):
        df = pd.read_csv("./debug_comments.csv")
        return df

    client_id = os.environ['REDDIT_CLIENT_ID']
    client_secret = os.environ['REDDIT_CLIENT_SECRET']
    user_agent = os.environ['REDDIT_USER_AGENT']

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

    try:
        submission = reddit.submission(url=url)
    except praw.exceptions.InvalidURL:
        print("The URL is invalid. Make sure that you have included the submission id")

    submission.comments.replace_more(limit=0)

    cols = [
        "text",
        "score",
        "id",
        "parent_id",
        "submission_title",
        "submission_score",
        "submission_id"
    ]
    rows = []

    for comment in submission.comments.list():

        if comment.stickied:
            continue

        data = [
            comment.body,
            comment.score,
            comment.id,
            comment.parent_id,
            submission.title,
            submission.score,
            submission.id,
        ]

        rows.append(data)

    df = pd.DataFrame(data=rows, columns=cols)

    if debug:
        # save for debugging to avoid sending tons of requests to reddit

        df.to_csv('debug_comments.csv', index=False)

    return df


def summarizer(url: str) -> str:

    # pushshift.io submission comments api doesn't work so have to use praw
    df = getComments(url=url)
    chunked_df = preprocessData(df)

    submission_title = df.submission_title.unique()[0]

    lst_summaries = []

    nlp = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")

    for grp in chunked_df:
        # treating a group of comments as one block of text
        result = nlp(grp, max_length=500)[0]["summary_text"]
        lst_summaries.append(result)

    joined_summaries = ' '.join(lst_summaries).replace(" .", ".")

    total_summary = nlp(joined_summaries, max_length=500)[0]["summary_text"].replace(" .", ".")

    short_output = submission_title + '\n' + '\n' + total_summary

    long_output = submission_title + '\n' + '\n' + joined_summaries

    return short_output, long_output


if __name__ == "__main__":

    with gr.Blocks(css=".gradio-container {max-width: 900px !important; width: 100%}") as demo:
        submission_url = gr.Textbox(label='Post URL')

        sub_btn = gr.Button("Summarize")

        with gr.Row():
            short_summary = gr.Textbox(label='Short Summary')
            long_summary = gr.Textbox(label='Long Summary')

        sub_btn.click(fn=summarizer,
                      inputs=[submission_url],
                      outputs=[short_summary, long_summary])

    try:
        demo.launch()
    except KeyboardInterrupt:
        gr.close_all()
