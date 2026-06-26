from symspellpy import SymSpell, Verbosity

class SpellCorrector:
    def __init__(self, dict_path="frequency_dictionary_en_82_765.txt"):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        self.sym_spell.load_dictionary(
            dict_path,
            term_index=0,
            count_index=1
        )

    def correct(self, query: str) -> str:
        words = query.split()
        corrected_words = []

        for w in words:
            suggestions = self.sym_spell.lookup(
                w,
                Verbosity.CLOSEST,
                max_edit_distance=2
            )

            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(w)

        return " ".join(corrected_words)
