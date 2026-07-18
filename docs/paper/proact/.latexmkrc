# Keep paper/ tidy: all LaTeX intermediates (.aux .log .bbl .fls .synctex ...)
# go into build/, and only the final PDF is copied back up next to the .tex.
# Works for `latexmk` on the command line AND for editor plugins that call
# latexmk (e.g. VS Code LaTeX Workshop), because latexmk auto-reads this file.
# NOTE: this file is Perl -- comments use '#', not '%'.
$pdf_mode    = 1;            # build PDF with pdflatex
$out_dir     = 'build';      # all intermediates + PDF land here first
$bibtex_use  = 2;            # run bibtex when references change
# after a successful build, copy the PDF up into paper/
$success_cmd = 'cp build/%R.pdf %R.pdf';
