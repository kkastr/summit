import os
import re
import sys
import nltk
import praw
import toml
import matplotlib
import gradio as gr
import pandas as pd
from tqdm import tqdm
import praw.exceptions
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline

matplotlib.use('Agg')


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

    # The df is sorted by comment score
    # Empirically, having more than ~100 comments doesn't change much but slows down the summarizer.
    # Slowdown is not present with load api but still seems good to limit low score comments.
    if len(df.text) >= 128:
        df = df[:128]

    # chunking to handle giving the model too large of an input which crashes
    chunked = sentence_chunk(df.text)

    return chunked


def getComments(url, debug=False):

    if debug and os.path.isfile('./debug_comments.csv'):
        df = pd.read_csv("./debug_comments.csv")
        return df

    reddit = praw.Reddit(
        client_id=api_keys['client_id'] ,
        client_secret=api_keys['client_secret'] ,
        user_agent=api_keys['user_agent']
    )

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

    submission_title = '## ' + df.submission_title.unique()[0]

    chunked_df = preprocessData(df)

    text = ' '.join(chunked_df)
    # transparent bg: background_color=None, mode='RGBA')
    wc_opts = dict(collocations=False, width=1280, height=720)
    wcloud = WordCloud(**wc_opts).generate(text)

    plt.imshow(wcloud, aspect='auto')
    plt.axis("off")
    plt.gca().set_position([0, 0, 1, 1])
    plt.autoscale(tight=True)
    fig = plt.gcf()
    fig.patch.set_alpha(0.0)
    fig.set_size_inches((12, 7))

    lst_summaries = []

    for grp in tqdm(chunked_df):
        # treating a group of comments as one block of text
        result = sum_api(grp)
        lst_summaries.append(result)

    long_output = ' '.join(lst_summaries).replace(" .", ".")

    short_output = sum_api(long_output).replace(" .", ".")

    sentiment = clf_api(short_output)

    return gr.update(visible=True), submission_title, short_output, long_output, sentiment, fig


if __name__ == "__main__":
    if not os.path.isfile('./api_params.toml'):
        print("""
                Could not find api params config file in directory.
                Please create api_params.toml by following the instructions in the README.
              """)
        sys.exit(1)

    api_keys = toml.load("api_params.toml")

    sum_model = "models/sshleifer/distilbart-cnn-12-6"
    clf_model = "models/finiteautomata/bertweet-base-sentiment-analysis"

    hf_token = (None if "hf_token" not in api_keys else api_keys["hf_token"])

    sum_api = gr.Interface.load(sum_model, api_key=hf_token)
    clf_api = gr.Interface.load(clf_model, api_key=hf_token)

    # 'https://www.reddit.com/r/wholesome/comments/uvcck3/man_realises_he_has_a_perfect_life/'
    # 'https://www.reddit.com/r/politics/comments/11caol9/

    sample_urls = ['https://www.reddit.com/r/wholesome/comments/10ehlxo/he_got_a_strong_message/']

    with gr.Blocks(css=".gradio-container {max-width: 900px !important; width: 100%}") as demo:
        submission_url = gr.Textbox(label='Post URL')

        sub_btn = gr.Button("Summarize")

        title = gr.Markdown("")

        with gr.Column(visible=False) as result_col:
            with gr.Row():
                with gr.Column():
                    short_summary = gr.Textbox(label='Short Summary')
                    summary_sentiment = gr.Label(label='Sentiment')

                thread_cloud = gr.Plot(label='Word Cloud')

            long_summary = gr.Textbox(label='Long Summary')

        out_lst = [result_col, title, short_summary, long_summary, summary_sentiment, thread_cloud]

        sub_btn.click(fn=summarizer, inputs=[submission_url], outputs=out_lst)

        examples = gr.Examples(examples=sample_urls,
                               fn=summarizer,
                               inputs=[submission_url],
                               outputs=out_lst,
                               cache_examples=True)

    try:
        demo.launch()
    except KeyboardInterrupt:
        gr.close_all()
