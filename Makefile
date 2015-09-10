PANDOC=pandoc

ROOT=""

PANDOCARGS=-t revealjs -s -V theme=night --css=http://lab.hakim.se/reveal-js/css/theme/night.css \
					 --css=$(ROOT)/css/ucl_reveal.css --css=$(ROOT)/site-styles/reveal.css \
           --default-image-extension=png --highlight-style=zenburn --mathjax -V revealjs-url=http://lab.hakim.se/reveal-js

MDS=$(wildcard session*/*.md )

TEMPLATED=$(MDS:.md=.tmd)

SLIDES=$(MDS:.md=.slide.html)

EXES=$(shell find build -name *.x)

vpath %.x build

OUTS=$(subst build/,,$(EXES:.x=.out))

default: _site

.DELETE_ON_ERROR:

%.out: %.x Makefile
	$< > $@

%.slide.html: %.md Makefile
	cat $^ | $(PANDOC) $(PANDOCARGS) -o $@

%.png: %.py Makefile
	python $< $@

%.png: %.nto Makefile
	neato $< -T png -o $@

%.png: %.dot Makefile
	dot $< -T png -o $@

%.png: %.uml Makefile
   java -Djava.awt.headless=true -jar plantuml.jar -p < $< > $@

notes.pdf: combined.md Makefile
	$(PANDOC) --from markdown combined.md -o notes.pdf

%.tmd: %.md liquify.rb _plugins/idio.rb Makefile
	ruby liquify.rb $< > $@

combined.md: $(TEMPLATED)
	cat $^ > $@

notes.tex: combined.md Makefile
	$(PANDOC) --from markdown combined.md -o notes.tex

master.zip: Makefile
	rm -f master.zip
	wget https://github.com/UCL-RITS/indigo-jekyll/archive/master.zip

ready: indigo $(OUTS)

indigo-jekyll-master: Makefile master.zip
	rm -rf indigo-jekyll-master
	unzip master.zip
	touch indigo-jekyll-master

indigo: indigo-jekyll-master Makefile
	cp -r indigo-jekyll-master/indigo/images .
	cp -r indigo-jekyll-master/indigo/js .
	cp -r indigo-jekyll-master/indigo/css .
	cp -r indigo-jekyll-master/indigo/_includes .
	cp -r indigo-jekyll-master/indigo/_layouts .
	cp -r indigo-jekyll-master/indigo/favicon* .
	touch indigo

.PHONY: ready

_site: ready _plugins/idio.rb
	jekyll build --verbose

preview: ready
	jekyll serve --verbose

clean:
	rm -rf build
	rm -rf indigo
	rm -rf indigo-jekyll-master
	rm -f master.zip
	rm -f notes.pdf
	rm -rf _site
	rm -f favicon *
	rm -f combined*
	rm -rf _includes
	rm -rf _layouts
	rm -rf js
	rm -rf images
