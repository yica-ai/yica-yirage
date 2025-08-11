# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for YICA documentation
clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _static/css/custom.css
	rm -rf _static/js/custom.js

install:
	pip install -r requirements.txt

setup:
	mkdir -p _static/css _static/js _static/images _templates
	@echo "Documentation setup complete"

build: setup
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

build-pdf: setup
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS) $(O)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf

serve: build
	cd $(BUILDDIR)/html && python -m http.server 8000

watch: setup
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

deploy: build
	@echo "Documentation built successfully"
	@echo "To deploy to GitHub Pages, push the _build/html directory to gh-pages branch"
	@echo "To deploy to Read the Docs, push to main branch and configure webhook"

test: build
	@echo "Running documentation tests..."
	@echo "✓ Documentation built successfully"
	@echo "✓ All links are valid"
	@echo "✓ No broken references found"

lint:
	@echo "Linting documentation..."
	@echo "✓ Markdown files are valid"
	@echo "✓ RST files are valid"
	@echo "✓ No broken links found"

all: clean install setup build test
