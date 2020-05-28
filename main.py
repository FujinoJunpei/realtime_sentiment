import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from realtime_sentiment.src.streaming import get_df
from realtime_sentiment.src.preprocessing import preprocess_text, make_jsonl
from realtime_sentiment.src.output import output


df = get_df()
if df is None:
    print("No new messages!")
else:
    df = preprocess_text(df)
    make_jsonl(df)
    # predict()
    output()