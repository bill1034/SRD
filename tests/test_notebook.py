import ast
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]

def test_notebook_contract_and_no_hidden_source_dependency():
    nb=nbformat.read(ROOT/'notebooks'/'SRD.ipynb',as_version=4)
    nbformat.validate(nb)
    assert nb.cells[0].cell_type == nb.cells[1].cell_type == 'code'
    imports=ast.parse(nb.cells[0].source).body
    assert imports and all(isinstance(n,(ast.Import,ast.ImportFrom)) for n in imports)
    assert not any(isinstance(n,ast.ImportFrom) and n.module in ('source','model','pipeline','reasoning','data','config') for n in imports)
    control=ast.parse(nb.cells[1].source).body
    assert len(control)==1 and isinstance(control[0],ast.Assign)
    cfg=ast.literal_eval(control[0].value)
    assert cfg['image_size']==1024 and cfg['stage_a_epochs']==30 and cfg['stage_b_epochs']==20
    assert cfg['mode']=='train' and cfg['checkpoint_path']==''
    for cell in nb.cells[2:]:
        if cell.cell_type=='code':
            tree=ast.parse(cell.source)
            assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree))
            assert cell.outputs==[] and cell.execution_count is None
    assert sum('RESULT = run_model(CFG)' in c.source for c in nb.cells)==1
