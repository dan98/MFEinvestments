(TeX-add-style-hook
 "solution"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("inputenc" "utf8") ("geometry" "margin=1.5in")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "fontenc"
    "inputenc"
    "concmath"
    "eulervm"
    "amsmath"
    "amsthm"
    "amssymb"
    "mathtools"
    "multicol"
    "marginnote"
    "pgfplots"
    "float"
    "hyperref"
    "bbm"
    "booktabs"
    "listings"
    "xcolor"
    "wasysym"
    "geometry"
    "enumerate")
   (TeX-add-symbols
    "N"
    "Z"
    "R"
    "C"
    "Pbb"
    "Fcal"
    "Lcal"
    "Acal"
    "Ecal"
    "Ebb"
    "Qbb")
   (LaTeX-add-labels
    "eq:twoportfolios"
    "eq:covariances"
    "eq:variances"
    "fig:ex3")
   (LaTeX-add-environments
    '("definition" LaTeX-env-args ["argument"] 1)
    '("corollary" LaTeX-env-args ["argument"] 1)
    '("proposition" LaTeX-env-args ["argument"] 1)
    '("reflection" LaTeX-env-args ["argument"] 1)
    '("exercise" LaTeX-env-args ["argument"] 1)
    '("lemma" LaTeX-env-args ["argument"] 1)
    '("theorem" LaTeX-env-args ["argument"] 1))
   (LaTeX-add-index-entries
    "\\usepackage")
   (LaTeX-add-listings-lstdefinestyles
    "mystyle")
   (LaTeX-add-xcolor-definecolors
    "codegreen"
    "codegray"
    "codepurple"
    "backcolour"))
 :latex)

