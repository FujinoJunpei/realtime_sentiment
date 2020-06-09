import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from realtime_sentiment.src.streaming import get_df
from realtime_sentiment.src.preprocessing import preprocess_text, make_jsonl
from realtime_sentiment.src.output import output


df = get_df()
if isinstance(df, str):
    print("No new message!")
else:
    df = preprocess_text(df)
    # print(df)
    make_jsonl(df)

output()