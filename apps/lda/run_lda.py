#!/usr/bin/env python


import lda
import sys
DIFF3 = ('alt.atheism', 'rec.sport.baseball', 'sci.space')
SIM3 = ('comp.graphics', 'comp.os.ms-windows.misc', 'comp.windows.x')
REL3 = ('talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc')
TNG = ('alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
    'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
    'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
    'sci.space', 'soc.religion.christian', 'talk.politics.guns',
    'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc')
lda.run_lda('../../data/txt/tng', globals()[sys.argv[1]], int(sys.argv[2]))
