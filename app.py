import os
import re
import sys
import toml
import praw
import gradio as gr
import pandas as pd
import praw.exceptions
from transformers import pipeline


def chunk(a):
    n = round(0.3 * len(a))
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


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
    chunked = list(chunk(df.text))

    return chunked


def getComments(url, debug=False):

    api_keys = toml.load('./api_params.toml')

    reddit = praw.Reddit(
        client_id=api_keys['client_id'] ,
        client_secret=api_keys['client_secret'] ,
        user_agent=api_keys['user_agent']
    )

    try:
        submission = reddit.submission(url=url)
        if debug and os.path.isfile(f'./{submission.id}_comments.csv'):
            df = pd.read_csv(f"./{submission.id}_comments.csv")
            return df
        else:
            pass
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

        df.to_csv(f'{submission.id}_comments.csv', index=False)

    return df


def summarizer(url: str, summary_length: str = "Short") -> str:

    # pushshift.io submission comments api doesn't work so have to use praw
    df = getComments(url=url)
    chunked_df = preprocessData(df)

    submission_title = df.submission_title.unique()[0]

    nlp = pipeline('summarization', model="model/")

    lst_summaries = []

    for grp in chunked_df:
        # treating a group of comments as one block of text
        result = nlp(grp.str.cat(), max_length=500)[0]["summary_text"]
        lst_summaries.append(result)

    stext = ' '.join(lst_summaries).replace(" .", ".")

    if summary_length == "Short":
        thread_summary = nlp(stext, max_length=500)[0]["summary_text"].replace(" .", ".")
        return submission_title + '\n' + '\n' + thread_summary
    else:
        return submission_title + '\n' + '\n' + stext


if __name__ == "__main__":
    if not os.path.isfile('./api_params.toml'):
        print("""
                Could not find api params config file in directory.
                Please create api_params.toml by following the instructions in the README.
              """)
        sys.exit(1)

    with gr.Blocks(css=".gradio-container {max-width: 900px; margin: auto;}") as demo:
        submission_url = gr.Textbox(label='Post URL')

        length_choice = gr.Radio(label='Summary Length', value="Short", choices=["Short", "Long"])

        sub_btn = gr.Button("Summarize")

        summary = gr.Textbox(label='Comment Summary')

        sub_btn.click(fn=summarizer, inputs=[submission_url, length_choice], outputs=summary)

    try:
        demo.launch()
    except KeyboardInterrupt:
        gr.close_all()
