""" 
This code will use PyPDFParser package
https://py-pdf-parser.readthedocs.io/en/latest/overview.html 

"""

from py_pdf_parser.loaders import load_file
from py_pdf_parser.visualise import visualise

document = load_file("simple_memo.pdf")

visualise(document)
