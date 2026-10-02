#!/usr/bin/env python3
"""Compile the generated TikZ sources to standalone vector PDFs."""
from pathlib import Path
import os,subprocess,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
PRE=r'''\documentclass[tikz,border=0pt]{standalone}
\usepackage{fontspec}
\setmainfont[ItalicFont=Arial,BoldItalicFont={Arial Bold}]{Arial}
\setsansfont[ItalicFont=Arial,BoldItalicFont={Arial Bold}]{Arial}
\usepackage{newtxmath}
\usepackage{pgfplots,graphicx,amsmath,siunitx}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{positioning,arrows.meta,calc,fit,backgrounds}
\definecolor{tblue}{RGB}{43,108,176}
\definecolor{tgreen}{RGB}{47,133,90}
\definecolor{torange}{RGB}{221,107,32}
\definecolor{tred}{RGB}{197,48,48}
\definecolor{tgray}{RGB}{160,174,192}
\definecolor{tpurple}{RGB}{107,70,193}
\begin{document}
'''
env=dict(os.environ,SOURCE_DATE_EPOCH='0',FORCE_SOURCE_DATE='1',TZ='UTC')
def compile_one(p):
 name='_render_'+p.stem;wrapper=ROOT/(name+'.tex')
 wrapper.write_text(PRE+'\\input{figs/'+p.name+'}\n\\end{document}\n')
 run=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',wrapper.name],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 if run.returncode:raise RuntimeError(p.name+'\n'+run.stdout.decode(errors='replace')[-1800:])
 shutil.move(ROOT/(name+'.pdf'),p.with_suffix('.pdf'))
 for suffix in ['.tex','.aux','.log']:(ROOT/(name+suffix)).unlink(missing_ok=True)
 return p.stem
if __name__=='__main__':
 sources=[ROOT/'figs/fig_mechanism_arch.tex']
 with ThreadPoolExecutor(max_workers=3) as pool:
  for name in pool.map(compile_one,sources):print(name,flush=True)
