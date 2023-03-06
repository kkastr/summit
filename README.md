# Reddit Thread Summarizer (Gradio)

Ever wanted to get the gist of a reddit thread without having to sift through memes, tldr worthy comments, and bad takes? Well, look no further!

Leverage the power of the Transformers library and Gradio to automatically generate summaries of comment threads on Reddit! The code uses state-of-the-art NLP models, such as BART, to analyze and extract the most important information from lengthy comment threads on Reddit.

This gradio app takes as input the URL of a Reddit thread and produces a short (and long) summary of the comments therein.

The model is deployed for online use [here](https://kkastr-summit.hf.space/).

## Usage

The following steps are only needed if you wish to run to model locally.

First, obtain api credentials from a reddit account ([instructions](https://github.com/reddit-archive/reddit/wiki/OAuth2-Quick-Start-Example#first-steps)). After you have the relevant api secrets, create a file named `api_params.toml` with the following contents,

```toml
client_id = "your-client-id"
client_secret = "your-client-secret"
user_agent = "your-user_agent"
```

Next, install all the requirements for the code,

```bash
pip install -r requirements.txt
```

After the installation is complete, run the command below to download all the tokenizers and models needed to run the summarizer.

```bash
python download_model.py
```

Finally, you can launch the app from the terminal as shown below, creating a gradio instance at a port in localhost.

```bash
python app.py
```

## TODO

- [x] Add sentence segmentation to improve performance.
- [ ] Add sentiment analysis to output
- [ ] Improve sentence segmentation
