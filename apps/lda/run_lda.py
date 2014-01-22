#!/usr/bin/env python


import lda
lda.run_lda('../../data/txt/tng', *('alt.atheism', 'rec.sport.baseball', 'sci.space'))
