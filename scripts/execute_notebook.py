import nbformat
from nbclient import NotebookClient
from pathlib import Path
root=Path(__file__).resolve().parents[1];file=root/'notebooks/clip-lab.ipynb';nb=nbformat.read(file,as_version=4)
client=NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(root)}})
try:client.execute()
finally:nbformat.write(nb,file)
assert not [o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
print('Executed all code cells successfully.')
