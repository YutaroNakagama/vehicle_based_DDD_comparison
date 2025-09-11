# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
APIDOCDIR     = $(SOURCEDIR)
PROJECTDIR    = ../project/src

.PHONY: help clean html apidoc

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)/*

# Run sphinx-apidoc automatically before html build
apidoc:
	sphinx-apidoc -o $(APIDOCDIR) $(PROJECTDIR)

html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

