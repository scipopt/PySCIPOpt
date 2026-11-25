import re
from pathlib import Path

import pytest

from pyscipopt import BaseModel, Eventhdlr, ExactModel, ExactSolution, ExactVariable, Model, Variable

SRC = Path(__file__).resolve().parents[1] / "src" / "pyscipopt"
BASEMODEL_PXI = SRC / "basemodel.pxi"
MODEL_PXI = SRC / "model.pxi"

FP_TYPES = ("SCIP_Real", "Model", "Expr", "ExprCons", "Row", "Column", "NLRow")
FP_CALLS = ("float(", "dtype=float", "SCIPinfinity(")

FP_ALLOWED = frozenset({
    "create", "getBranchScoreMultiple", "setParam",
})

MEMBER_RE = re.compile(r"^    (?:def|cdef|cpdef)\b.*?(\w+)\s*\(")


def uses_floating_point(body):
    return any(re.search(rf"\b{name}\b", body) for name in FP_TYPES) or any(call in body for call in FP_CALLS)


def class_members(path, header):
    if not path.exists():
        pytest.skip(f"{path.name} is not available in this checkout")
    lines = path.read_text().splitlines()
    start = lines.index(header)
    end = start + 1
    while end < len(lines) and (not lines[end] or lines[end][0] in " \t#"):
        end += 1
    members = {}
    name = None
    for line in lines[start + 1:end]:
        match = MEMBER_RE.match(line)
        if match:
            name = match.group(1)
            members.setdefault(name, [])
        if name:
            members[name].append(line)
    stripped = {}
    for name, body in members.items():
        text = re.sub(r'"""[\s\S]*?"""', "", "\n".join(body))
        stripped[name] = "\n".join(line.split("#")[0] for line in text.splitlines())
    return stripped


def test_base_model_calls_only_shared_methods():
    base = class_members(BASEMODEL_PXI, "cdef class BaseModel:")
    model_only = set(class_members(MODEL_PXI, "cdef class Model(BaseModel):")) - set(base)
    offenders = {}
    for name, body in base.items():
        calls = set(re.findall(r"self\.(\w+)\(", body))
        bad = sorted(calls & model_only)
        if bad:
            offenders[name] = bad
    assert offenders == {}


def test_base_model_has_no_fp_markers():
    base = class_members(BASEMODEL_PXI, "cdef class BaseModel:")
    hits = {name for name, body in base.items() if uses_floating_point(body)}
    assert hits - FP_ALLOWED == set()


def test_model_subclass_and_mro():
    assert Model.__mro__[:2] == (Model, BaseModel)
    with pytest.raises(TypeError):
        BaseModel()

    class MyModel(Model):
        def __init__(self):
            super().__init__("sub")

    m = MyModel()
    x = m.addVar("x", lb=1)
    m.setObjective(x)
    m.hideOutput()
    m.optimize()
    assert m.getStatus() == "optimal"
    assert m.getVal(x) == 1


def test_from_ptr_returns_model():
    m = Model()
    assert type(Model.from_ptr(m.to_ptr(False), False)) is Model


def test_plugin_sees_the_same_model_object():
    seen = []

    class Watcher(Eventhdlr):
        def eventinit(self):
            seen.append(self.model)

    m = Model()
    m.includeEventhdlr(Watcher(), "watcher", "")
    m.addVar("x")
    m.hideOutput()
    m.optimize()
    assert seen and seen[0] is m and isinstance(seen[0], BaseModel)


def test_exact_model_skeleton(tmp_path):
    assert ExactModel.__mro__[:2] == (ExactModel, BaseModel)
    assert {"addRow", "allowNegSlack", "branchLP"} <= set(vars(ExactModel))
    with pytest.warns(DeprecationWarning):
        Model().isExact()
    with pytest.raises(ValueError):
        Model().enableExactSolving(True)
    try:
        m = ExactModel()
    except ValueError:
        pytest.skip("SCIP was built without exact solving support")
    assert not isinstance(m, Model)
    assert type(BaseModel.from_ptr(m.to_ptr(False), False)) is ExactModel
    assert not hasattr(m, "getVal")
    with pytest.raises(AttributeError, match="Model.includeConshdlr has this attribute"):
        m.includeConshdlr
    with pytest.raises(AttributeError, match="no attribute 'noSuchMethod'$"):
        m.noSuchMethod
    assert m.getNVars() == 0
    assert type(Model().addVar()) is Variable
    path = tmp_path / "exact.lp"
    path.write_text("Minimize\n obj: x + 3 y\nSubject To\n c1: 2 x + 3 y >= 1\nBounds\n 0 <= x <= 10\n 0 <= y <= 10\nGeneral\n x\nEnd\n")
    m.hideOutput()
    m.readProblem(str(path))
    m.optimize()
    assert all(type(v) is ExactVariable for v in m.getVars())
    sol = m.getBestSol()
    assert type(sol) is ExactSolution
    x = m.getVars()[0]
    assert x.name == "x" and x.vtype() == "INTEGER"
    with pytest.raises(NotImplementedError):
        x.getLbLocal()
    with pytest.raises(NotImplementedError):
        sol[x]
