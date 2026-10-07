"""Execute the report notebook in a fresh Jupyter kernel."""
import asyncio
import os
import sys
import nbformat
from nbclient import NotebookClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT

def main():
    if sys.platform == 'win32':
        # ZeroMQ requires add_reader support, provided by the selector loop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import tempfile
    os.environ['MPLCONFIGDIR']=str(Path(tempfile.gettempdir())/'freight_assessment_matplotlib')
    path=ROOT/'notebooks/assessment.ipynb'
    notebook=nbformat.read(path,as_version=4)
    client=NotebookClient(notebook,timeout=180,startup_timeout=40,kernel_name='python3',
                          resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook,path)
    counts=[c.execution_count for c in notebook.cells if c.cell_type=='code']
    print('Fresh kernel completed; executed code cells:',len(counts),flush=True)

if __name__=='__main__':main()
