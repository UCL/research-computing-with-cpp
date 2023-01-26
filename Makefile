PANDOC=pandoc

ROOT="/research-computing-with-cpp/"

PANDOCARGS=-t revealjs -s -V theme=night --css=http://lab.hakim.se/reveal-js/css/theme/night.css \
           --css=$(ROOT)/css/ucl_reveal.css --css=$(ROOT)/site-styles/reveal.css \
           -V width:'"100%"' -V height:'"100%"' -V margin:0 -V minScale:0.1 -V maxScale:1.5 \
           --default-image-extension=png --highlight-style=zenburn --mathjax -V revealjs-url=http://lab.hakim.se/reveal-js

MDS=$(filter-out _site%, $(wildcard 01research/*.md 02cpp1/index.md 02cpp1/sec01*.md 02cpp1/sec02*.md 02cpp1/sec03EssentialReading-reveal.md 03cpp2/*.md 04cpp3/*.md ))

TEMPLATED=$(MDS:.md=.tmd)

RELATIVE=$(MDS:.md=.rmd)

SLIDES=$(MDS:.md=-reveal.html)

EXES=$(shell find build -type f \( -perm -u=x -o -perm -g=x -o -perm -o=x \) -name *.x)

PY_FIGURE_SOURCES= $(shell find 06MPI/figures -name *.py)

PY_FIGURES=$(PY_FIGURE_SOURCES:.py=.png)

vpath %.x build

OUTS=$(subst build/,,$(EXES:.x=.out))

default: _site

.DELETE_ON_ERROR:

%.out: %.x Makefile
	$< > $@

%-reveal.html: %.rmd Makefile
	cat $< | $(PANDOC) $(PANDOCARGS) -o $@

%.png: %.py Makefile
	python $< $@

%.png: %.nto Makefile
	neato $< -T png -o $@

%.png: %.dot Makefile
	dot $< -T png -o $@

%.png: %.uml Makefile
	java -Djava.awt.headless=true -jar plantuml.jar -p < $< > $@

notes.pdf: combined.md Makefile $(PY_FIGURES)
	$(PANDOC) -Vdocumentclass=report --toc --from markdown combined.md -o notes.pdf

%.tmd: %.md liquify.rb _plugins/idio.rb Makefile
	bundle exec ruby liquify.rb $< > $@

%.rmd: %.md liquify.rb _plugins/idio.rb Makefile
	bundle exec ruby liquify.rb $< slides > $@

combined.md: $(TEMPLATED) cover.md
	cat cover.md `echo $^ | sed s/cover.md// ` > $@

notes.tex: combined.md Makefile $(OUTS)
	$(PANDOC) --from markdown combined.md -o notes.tex

master.zip: Makefile
	rm -f master.zip

ready: $(OUTS) notes.pdf $(SLIDES) notes.tex $(PY_FIGURES)

.PHONY: ready

_site: ready _plugins/idio.rb
	jekyll build --verbose

preview: ready
	jekyll serve --verbose

clean:
	rm -rf build
	rm -f master.zip
	rm -f notes.pdf
	rm -rf _site
	rm -f favicon*
	rm -f combined*
	rm -rf _includes
	rm -rf _layouts
	rm -rf js
	rm -rf images
	rm -f */*.tmd
	rm -f */*.rmd
	rm -f */*.slide.html
	rm -f */*-reveal.html
	rm -rf css
