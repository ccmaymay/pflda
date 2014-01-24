#!/usr/bin/env python


import lda
import sys

DATASETS = dict(
    diff3=('alt.atheism', 'rec.sport.baseball', 'sci.space'),
    sim3=('comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x'),
    rel3=('talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'),
    tng=('alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
        'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
        'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
        'sci.space', 'soc.religion.christian', 'talk.politics.guns',
        'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'),
)

kwargs = dict()
for token in sys.argv[2:]:
    eq_pos = token.find('=')
    if token.startswith('--') and eq_pos >= 0:
        k = token[len('--'):eq_pos]
        v = token[(eq_pos+1):len(token)]
        kwargs[k] = v

lda.run_lda('../../data/txt/tng', DATASETS[sys.argv[1]], **kwargs)
