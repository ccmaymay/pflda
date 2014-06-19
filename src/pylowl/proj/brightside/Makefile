# This is not the makefile you're looking for.

model.pdf: model.tex model.bib
	pdflatex model && bibtex model && pdflatex model && pdflatex model

.PHONY: clean
clean:
	rm -f model.aux model.bbl model.blg model.log model.pdf
