#!/usr/bin/env python3
"""Validate generated PDF and package all manuscript dependencies."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];paper=ROOT/'paper/jsa'
sys.path.insert(0,str(ROOT/'.experiments/pdf_tools'))
import pymupdf
pdf=ROOT/'artifacts/jsa-main.pdf'
with pymupdf.open(pdf) as doc:
 text='\n'.join(page.get_text() for page in doc)
 assert len(doc)>0 and 'References' in text,'Incomplete manuscript PDF'
 assert 'SCIC-v2' in text,'New evaluation missing from manuscript PDF'
 report=dict(pages=len(doc),sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),references_present=True,v2_present=True)
(ROOT/'paper/data/jsa_v2/provenance/pdf_validation.json').write_text(json.dumps(report,indent=2)+'\n')
files=[*paper.glob('*.tex'),paper/'references.bib',paper/'jsa_main_zh.md',paper/'README-OVERLEAF.md']
files += [p for folder in ('figures','thumbnails') for p in (paper/folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.pdf','.png','.jpeg','.jpg')]
archive=ROOT/'paper/jsa-overleaf.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for path in sorted(set(files)):z.write(path,path.relative_to(paper))
shutil.copy2(archive,ROOT/'artifacts/jsa-overleaf.zip')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert {'v2_method.tex','v2_evaluation.tex','main.tex','jsa_main_zh.md'}<=set(z.namelist())
 print('PDF pages',report['pages'],'Overleaf files',len(z.namelist()))
(ROOT/'paper/data/jsa_v2/provenance/completion.json').write_text(json.dumps(dict(
    pdf=report,overleaf_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    status='completed',archive_valid=True),indent=2)+'\n')
