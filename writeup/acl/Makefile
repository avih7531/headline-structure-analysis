PAPER = final_paper

.PHONY: all clean

# Tectonic auto-fetches missing TeX packages and runs bibtex automatically.
# Falls back to the standard pdflatex/bibtex chain if tectonic is unavailable.
all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex custom.bib acl.sty acl_natbib.bst
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic --reruns 2 $(PAPER).tex; \
	else \
		pdflatex -interaction=nonstopmode $(PAPER).tex; \
		bibtex $(PAPER); \
		pdflatex -interaction=nonstopmode $(PAPER).tex; \
		pdflatex -interaction=nonstopmode $(PAPER).tex; \
	fi

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).toc
