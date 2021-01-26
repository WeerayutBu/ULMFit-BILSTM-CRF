class CoNLLDataset(object):
    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None, use_crf=True):

        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.use_crf = use_crf
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename, encoding='utf-8') as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0],ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        if self.use_crf:
                            tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]


    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length