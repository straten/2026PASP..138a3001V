
all: paper.pdf

FIGS := figures/dsb.pdf figures/usb.pdf figures/lsb.pdf figures/nyquist1.pdf figures/nyquist2.pdf \
	figures/signal_path.pdf figures/polarization_cases.pdf \
	figures/polarization_ellipse.pdf figures/polarization_sphere.pdf \
	figures/polarization_sphere_circular.pdf

paper.pdf: paper.tex local.bib $(FIGS) 
	pdflatex paper
	BIBINPUTS=psrrefs:.: bibtex paper
	pdflatex paper
	pdflatex paper
	pdflatex paper

MIXING = plot_scripts/mixing.py

usb-raw.pdf: $(MIXING)
	python3 $(MIXING) upper $@ 

lsb-raw.pdf: $(MIXING)
	python3 $(MIXING) lower $@

dsb-raw.pdf: $(MIXING)
	python3 $(MIXING) dual $@

nyquist1-raw.pdf: $(MIXING)
	python3 $(MIXING) nyquist1 $@

nyquist2-raw.pdf: $(MIXING)
	python3 $(MIXING) nyquist2 $@

signal_path-raw.pdf: plot_scripts/signal_path.py circuit.png
	python3 plot_scripts/signal_path.py $@

%-raw.pdf: plot_scripts/%.py
	python3 $< $@

circuit.png: circuit.ps
	gs -sDEVICE=png16m -dSAFER -dBATCH -dNOPAUSE -r1200 -sOutputFile=circuit.png circuit.ps

circuit.ps: circuit.tex
	latex circuit
	dvips circuit

figures/%.pdf: %-raw.pdf figures
	pdfcrop $< $@


figures:
	mkdir -p figures

ARXIV_FILES := paper.tex paper.bbl $(FIGS) Orcid-ID.png aastex701.cls aasjournalv7.bst latexmkrc
arxiv: arxiv.tgz
arxiv.tgz : $(ARXIV_FILES)
	tar zcvf arxiv.tgz $(ARXIV_FILES) 

publish:
	mkdir -p /tmp/gh-pages
	mv paper.pdf /tmp/gh-pages
	cp README.md /tmp/gh-pages
	git checkout gh-pages
	mv /tmp/gh-pages/README.md index.md
	mv /tmp/gh-pages/paper.pdf 2026PASP..138a3001V.pdf
	git add index.md
	git add 2026PASP..138a3001V.pdf
	git commit -m "published using 'make publish'" .
	git push --set-upstream origin gh-pages
	git checkout main

clean:
	rm -rf `cat .gitignore`

