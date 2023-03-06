# Reddit Thread Summarizer (Gradio)

Leverage the power of the Transformers library and Gradio to automatically generate summaries of comment threads on Reddit! The code uses state-of-the-art NLP models, such as BART, to analyze and extract the most important information from lengthy comment threads on Reddit.

The main script `app.py` launches a gradio driven web UI which takes as input the URL of a Reddit thread and produces a short (and long) summary of the comments therein.

The model is deployed on huggingface spaces at [summit](huggingface.co/spaces/kkastr/summit).

## Usage

The following steps are needed to use to model locally.

First, obtain api credentials from a reddit account (as described [here](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps))

Next, install all the requirements for the code,

```bash
pip install -r requirements.txt
```

After the installation is complete, run the command below to download all the tokenizers and models needed to run the summarizer.

```bash
python download_model.py
```

Finally, you can launch the app from the terminal as shown below, creating a gradio instance at a port in localhost (default is `http://127.0.0.1:7860`)

```bash
python app.py
```

## TODO

- [x] Add sentence segmentation to improve performance.
- [ ] Add sentiment analysis to output
- [ ] Improve sentence segmentation
