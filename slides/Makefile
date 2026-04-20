# Makefile for LaTeX Beamer presentation
# Default: pdflatex (works out of the box)
# Use 'make xelatex' if you have xelatex properly configured

MAIN = slides
TEX = pdflatex
FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean view xelatex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(FLAGS) $(MAIN).tex
	$(TEX) $(FLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.pdf *.fls *.fdb_latexmk *.synctex.gz

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "Open $(MAIN).pdf manually"

# Use xelatex (requires fontspec uncommented in slides.tex)
xelatex:
	xelatex $(FLAGS) $(MAIN).tex
	xelatex $(FLAGS) $(MAIN).tex
