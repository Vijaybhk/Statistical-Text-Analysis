import spacy
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans


class MWETokenizer(object):
    """Use spaCy tokenizer to tokenize text, with MWEs tokenized as a single unit."""

    def __init__(self, terms):
        """Create MWE tokenizer.

        Parameters
        ----------
        terms: sequence of MWEs to be tokenized as units
        """

        self.nlp = spacy.load('en_core_web_sm', exclude=[
            'tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler'])
        self.phraser = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        self.phraser.add(
            'TERM', [self.nlp.tokenizer(t.strip()) for t in terms])

    def tokenize(self, text, sep='_'):
        """Tokenize text.

        Parameters
        ----------
        text: string to be tokenized
        sep: character used to separate words in MWEs

        Returns
        -------
        List of tokens
        """

        doc = self.nlp.tokenizer(text)
        with doc.retokenize() as r:
            for span in filter_spans(self.phraser(doc, as_spans=True)):
                r.merge(span)
        return [t.norm_.replace(' ', sep) for t in doc if not t.is_space and not t.is_punct]
