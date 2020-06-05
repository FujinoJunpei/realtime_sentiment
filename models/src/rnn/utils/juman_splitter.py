from pyknp import Juman
from allennlp.data.tokenizers import Token
import re
import emoji

from overrides import overrides
from typing import List

from allennlp.data.tokenizers import Token
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers.word_splitter import WordSplitter

@WordSplitter.register('juman')
class MeCabWordSplitter(WordSplitter):
    def __init__(self,
                 pos: bool = False,
                 tag: bool = False,
                 lemma: bool = False,
                 dicdir: str = '',
                 userdic: str = '',
                 other_options: str = '') -> None:
        self._pos = pos
        self._tag = tag
        self._lemma = lemma
        self._wakati = not (pos or tag or lemma)
        options = ''
        if self._wakati:
            options += '-Owakati'
        if dicdir:
            options += f' -d {dicdir}'
        if userdic:
            options += f' -u {userdic}'
        if other_options:
            options += f' {other_options}'
        self.juman = Juman()

    def _format_text(self, text):
        '''
        Jumanに入れる前のツイートの整形
        '''

        ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)
        text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
        text=re.sub(r'[!-~]', "", text) # 半角記号,数字,英字
        text=re.sub(r'[︰-＠]', "", text) # 全角記号
        text=re.sub('\n', "", text) # 改行文字
        text=re.sub(' ', "", text) # 必須

        return text.decode('unicode-escape')

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        tokens: List[Token] = []
        if self._wakati:
            elems = self.juman.analysis(self._format_text(sentence))
            tokens = [Token(text=elem) for elem in elems]
        else:
            elems = self.juman.analysis(self._format_text(sentence))[:-1]  # EOSを除く
            for elem in elems:
                text, fields = elem.split('\t')
                pos, tag_1, tag_2, tag_3, conjugation_type, conjugation, lemma, kana, utterance \
                    = fields.split(',')
                pos = pos if self._pos else None
                tag = ','.join((tag_1, tag_2, tag_3)) if self._tag else None
                lemma = lemma if self._lemma else None
                tokens.append(Token(text=text, lemma_=lemma, pos_=pos, tag_=tag))

        return tokens