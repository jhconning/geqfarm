Using Sphinx to build this documentation
===========================================

These are notes to myself on how the documentation for this project is being generated.


`Sphinx <http://www.sphinx-doc.org/en/1.5.1/>`_ is a python documentation generator.  It's a set of python utilities
that process restructured text and code files that you place in a code repository and convert their content to
a nicely formatted static html.  Using Sphinx encourages the writing of documentation along with the development
of the project.

Chances are you may be reading the html version of this document. Sphinx created that page from `this`_ raw
restructured text file.

.. _this: HowToSphinx.rst

For example Sphinx can extract the information you
write into python function docstrings to

(in the form of a well documented github repository
and matching more polished html website whose content can be easily converted to a PDF document or other formats
useful to users) is created alongside code development on a project.

Sphinx will scan through a project's folders and use restructured text files (with extension .rst) as well as
docstrings from python code to generate a static HTML website.  It's fairly easy to setup a workflow whereby
pushing new content to a github repo will trigger a webhook that will have a readthedocs service run Sphinx on
a cloud server to produce an online version of the website/documentation.

