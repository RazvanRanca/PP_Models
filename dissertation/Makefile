LATEX=pdflatex
BIBTEX=bibtex
DVIPS=dvips
PS2PDF=ps2pdf

all: acs-dissertation.pdf
acs-dissertation.pdf: acs-dissertation.ps
	$(PS2PDF) -dEmbedAllFonts=true acs-dissertation.ps test.pdf
	ps2pdf13 -dPDFSETTINGS=/prepress test.pdf acs-dissertation.pdf
	rm -f test.pdf

acs-dissertation.ps: acs-dissertation.dvi
	$(DVIPS) -Pdownload35 -ta4 acs-dissertation.dvi

acs-dissertation.dvi: *.tex dissertation.bib
	$(LATEX) acs-dissertation
	$(BIBTEX) acs-dissertation
	$(LATEX) acs-dissertation
	$(LATEX) acs-dissertation

clean:
	$(RM) -f acs-dissertation.pdf acs-dissertation.ps acs-dissertation.dvi 
	$(RM) -f *.log *.aux *.toc *.bbl *.lot *.lof

