import re

import nltk
from nltk.corpus import wordnet as wn

from preprocessor import Preprocessor
from spell_checker import SpellCorrector
from synonem_expander import SynonymExpander


class QueryRefiner:
    def __init__(self):
        self.expander = SynonymExpander()
        self.spell_corrector = SpellCorrector()
        self.preprocessor = Preprocessor()

    def expand_query(self, query_tokens, topn=3):
        expanded = list(query_tokens)
        for token in query_tokens:
            if token in self.word2vec_model.wv:
                similar = self.word2vec_model.wv.most_similar(token, topn=topn)
                expanded.extend([
                    word
                    for word, score in similar
                    if score > 0.7
                ])
        print('expaneded query: ', expanded)
        return expanded

    def refine(self, text: str) -> dict[str, object]:
        result = self.spell_corrector.correct(text)
        result = self.expander.expand(result)
        # print('>>>>>>>>')
        # print('enhanced query: ', result)
        # print('<<<<<<<<')
        return result
    
