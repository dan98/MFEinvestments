(TeX-add-style-hook
 "solution"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("geometry" "margin=1.5in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "fontenc"
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
    "Acal"
    "Ecal"
    "Ebb"
    "Qbb")
   (LaTeX-add-environments
    '("definition" LaTeX-env-args ["argument"] 1)
    '("corollary" LaTeX-env-args ["argument"] 1)
    '("proposition" LaTeX-env-args ["argument"] 1)
    '("reflection" LaTeX-env-args ["argument"] 1)
    '("exercise" LaTeX-env-args ["argument"] 1)
    '("lemma" LaTeX-env-args ["argument"] 1)
    '("theorem" LaTeX-env-args ["argument"] 1))
   (LaTeX-add-index-entries
    "\\usepackage"))
 :latex)

