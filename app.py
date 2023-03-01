import pandas as pd
import gradio as gr
import re
from transformers import pipeline
from scraper import getComments


def chunk(a):
    n = round(0.3 * len(a))
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def preprocessText(df):
    df["text"] = df["text"].apply(lambda x: re.sub(r"http\S+", "", x, flags=re.M))
    df["text"] = df["text"].apply(lambda x: re.sub(r"^>.+", "", x, flags=re.M))
    return df


def summarizer(url: str, summary_length: str = "Short") -> str:

    # pushshift.io submission comments api doesn't work so have to use praw

    df = preprocessText(getComments(url=url))

    smax = df.score.max()

    threshold = round(0.05 * smax)

    df = df[df.score >= threshold]

    # empirically, having more than 200 comments doesn't change much but slows down the summarizer.
    if len(df.text) >= 200:
        df = df[:200]

    # chunking to handle giving the model too large of an input which crashes
    chunked = list(chunk(df.text))

    nlp = pipeline('summarization', model="./model/")

    lst_summaries = []

    for grp in chunked:
        # treating a group of comments as one block of text
        result = nlp(grp.str.cat(), max_length=500)[0]["summary_text"]
        lst_summaries.append(result)

    stext = ' '.join(lst_summaries).replace(" .", ".")

    if summary_length == "Short":
        thread_summary = nlp(stext, max_length=500)[0]["summary_text"].replace(" .", ".")
        return df.submission_title.unique()[0] + '\n' + '\n' + thread_summary
    else:
        return df.submission_title.unique()[0] + '\n' + '\n' + stext


if __name__ == "__main__":

    with gr.Blocks(css=".gradio-container {max-width: 900px; margin: auto;}") as demo:
        submission_url = gr.Textbox(label='Post URL')

        length_choice = gr.Radio(label='Summary Length', value="Short", choices=["Short", "Long"])

        sub_btn = gr.Button("Summarize")

        summary = gr.Textbox(label='Comment Summary')

        sub_btn.click(fn=summarizer, inputs=[submission_url, length_choice], outputs=summary)

    demo.launch(server_port=8080, enable_queue=False)
