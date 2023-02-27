import praw
import pandas as pd
from api_keys import client_id, client_secret, user_agent, username


def getComments(url):

    cols = [
        "text",
        "score",
        "id",
        "parent_id",
        "submission_title",
        "submission_score",
        "submission_id"
    ]

    reddit = praw.Reddit(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent, username=username
    )

    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)
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

    # save for testing to avoid sending tons of requests to reddit

    # df.to_csv(f'{submission.id}_comments.csv', index=False)

    return df


if __name__ == "__main__":
    pass
