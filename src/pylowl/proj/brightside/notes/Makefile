notes.pdf: notes.tex notes.bib
	pdflatex notes && bibtex notes && pdflatex notes && pdflatex notes

.PHONY: clean
clean:
	rm -f notes.aux notes.bbl notes.blg notes.log notes.pdf
