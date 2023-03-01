import praw
import praw.exceptions as redditexception
import pandas as pd
import boto3


def getComments(url):

    ssm = boto3.client('ssm')
    cid = ssm.get_parameter(Name='client_id', WithDecryption=True)['Parameter']['Value']
    csecret = ssm.get_parameter(Name='client_secret', WithDecryption=True)['Parameter']['Value']
    user_agent = ssm.get_parameter(Name='user_agent', WithDecryption=True)['Parameter']['Value']

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
        client_id=cid , client_secret=csecret, user_agent=user_agent
    )

    try:
        submission = reddit.submission(url=url)
    except redditexception.InvalidURL:
        print("The URL is invalid. Make sure that you have included the submission id")

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
