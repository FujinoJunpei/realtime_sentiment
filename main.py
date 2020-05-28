import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from realtime_sentiment.src.streaming import get_df
from realtime_sentiment.src.preprocessing import preprocess_text, make_jsonl
from realtime_sentiment.src.output import output

def main():
    # start = time.time()
    df = get_df()
    if df is None:
        print("No new messages!")
        # elapsed_time = time.time() - start
        # print ("elapsed_time:{:.3f}".format(elapsed_time) + "[sec]")
    else:
        df = preprocess_text(df)
        make_jsonl(df)
        # predict()
        output()
        # elapsed_time = time.time() - start
        # print ("elapsed_time:{:.3f}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
    main()