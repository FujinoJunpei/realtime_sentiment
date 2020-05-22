import os


def save_dir(path):
    target_dir = f'realtime_sentiment/{path}'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    return target_dir
