from pyknp import Juman
from allennlp.data.tokenizers import Token, Tokenizer
from overrides import overrides

import re

import torch

@Tokenizer.register("juman_tokenizer")
class JumanTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

        self.juman = Juman()

    def _format_text(self, text):
        '''
        Jumanに入れる前のツイートの整形
        '''

        ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
        text=re.sub(r'[!-@]', "", text) # 半角記号,数字,英字
        text=re.sub(r'[︰-＠]', "", text) # 全角記号
        text=re.sub('\n', "", text) # 改行文字
        text=re.sub(' ', "", text) # 必須

        return text.decode('unicode-escape')

    def _split(self, text):
        text = self._format_text(text)
        
        if text != "":
            result = self.juman.analysis(text)
            tokens = [mrph.midasi for mrph in result.mrph_list()]
        else:
            tokens = None
        return tokens

    @overrides
    def tokenize(self, text):
        tokens = self._split(text)

        if tokens is None:
            return None

        return [Token(str(token)) for token in tokens]


