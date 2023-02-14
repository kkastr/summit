import pandas as pd
import gradio as gr
from transformers import pipeline
from scraper import getComments


def chunk(a):
    n = round(0.3 * len(a))
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def main(url: str) -> str:

    # pushshift.io submission comments api doesn't work so have to use praw

    df = getComments(url=url)

    smax = df.score.max()

    threshold = round(0.1 * smax)

    df = df[df.score >= threshold]

    if len(df.text) >= 200:
        df = df[:200]

    # chunking to handle giving the model too large of an input which crashes
    chunked = list(chunk(df.text))

    nlp = pipeline('summarization')

    lst_summaries = []

    for grp in chunked:
        # treating a group of comments as one block of text
        result = nlp(grp.str.cat(), max_length=500)[0]["summary_text"]
        lst_summaries.append(result)

    ntext = ' '.join(lst_summaries)

    thread_summary = nlp(ntext, max_length=500)[0]["summary_text"].replace(" .", ".")

    return thread_summary


if __name__ == "__main__":

    demo = gr.Interface(fn=main, inputs="text", outputs="text")

    demo.launch()
