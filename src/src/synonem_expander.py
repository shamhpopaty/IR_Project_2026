from nltk.corpus import wordnet as wn

class SynonymExpander:
    def __init__(self, max_synonyms=3):
        self.max_synonyms = max_synonyms

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().replace("_", " ").lower()
                if w != word:
                    synonyms.add(w)
        return list(synonyms)[:self.max_synonyms]

    def expand(self, query):
        words = query.lower().split()
        expanded_terms = []
        for w in words:
            expanded_terms.append(w)
            syns = self.get_synonyms(w)
            expanded_terms.extend(syns)
        return " ".join(expanded_terms)
