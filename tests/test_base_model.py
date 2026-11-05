import re
from pathlib import Path

import pytest

from pyscipopt import BaseModel, Eventhdlr, Model

SCIP_PXI = Path(__file__).resolve().parents[1] / "src" / "pyscipopt" / "scip.pxi"
MODEL_PXI = SCIP_PXI.with_name("model.pxi")

BASE_MODEL_METHODS = frozenset({
    "_freescip", "_getStageNames", "activateBenders", "addBendersSubproblem", "addConss",
    "addMatrixCons", "addNNodes", "addSol", "addVarLocks",
    "addVarLocksType", "allColsInLP", "applyCutsProbing", "attachEventHandlerCallback", "backtrackProbing",
    "branchVar", "catchEvent", "catchRowEvent", "catchVarEvent", "checkBendersSubproblemOptimality",
    "checkSol", "chgVarBranchPriority", "chgVarType", "computeBestSolSubproblems", "constructLP",
    "count", "createChild", "createProbBasic", "cutoffNode",
    "data", "deactivatePricer", "delCons", "delConsLocal", "delVar",
    "disableDebugSol", "disablePropagation", "dropEvent", "dropRowEvent", "dropVarEvent",
    "enableDebugSol", "enableReoptimization", "endProbing", "endStrongbranch", "free",
    "freeBendersSubproblems", "freeProb", "freeReoptSolve", "freeSol", "freeTransform",
    "from_ptr", "generateIIS", "getBendersAuxiliaryVar", "getBendersSubproblem", "getBendersVar",
    "getBestChild", "getBestLeaf", "getBestNode", "getBestSibling", "getBestSol",
    "getBestboundNode", "getBranchScoreMultiple", "getChildren", "getCondition",
    "getConsNVars", "getConss", "getCountedSols", "getCurrentNode", "getDepth",
    "getDeterministicTime", "getFocusNode", "getGap", "getHeurTiming", "getIIS",
    "getLPBasisInd", "getLeaves", "getMajorVersion", "getMaxDepth", "getMaxTotalDepth",
    "getMemExternEstim", "getMemTotal", "getMemUsed", "getMinorVersion", "getNBacktracks",
    "getNBestSolsFound", "getNBinVars", "getNChildren", "getNConss", "getNContVars",
    "getNCountedSols", "getNCuts", "getNCutsApplied", "getNFeasibleLeaves", "getNImplVars",
    "getNInfeasibleLeaves", "getNIntVars", "getNLPBranchCands", "getNLPCols", "getNLPIterations",
    "getNLPRows", "getNLPs", "getNLeaves", "getNLimSolsFound", "getNNodeLPIterations",
    "getNNodes", "getNNodesLeft", "getNObjlimLeaves", "getNReaders", "getNReoptRuns",
    "getNRuns", "getNSepaRounds", "getNSiblings", "getNSols", "getNSolsFound",
    "getNStrongbranchLPIterations", "getNTotalNodes", "getNVars", "getObjVal", "getObjectiveSense",
    "getOpenNodes", "getParam", "getParams", "getPlungeDepth", "getPresolvingTime",
    "getPrimalDualIntegral", "getPrioChild", "getPrioSibling", "getProbName", "getProbingDepth",
    "getPseudoBranchCands", "getReadingTime", "getSiblings", "getSolTime", "getSols",
    "getSolvingTime", "getStage", "getStageName", "getStatus", "getTechVersion",
    "getTotalTime", "getTransformedCons", "getTransformedVar", "getTreesizeEstimation", "getVarPseudocost",
    "getVarPseudocostScore", "getVarStrongbranchNode", "getVars", "hasPrimalRay", "hideOutput",
    "inProbing", "inRepropagation", "includeBenders", "includeBendersDefaultCuts", "includeBenderscut",
    "includeBranchrule", "includeConshdlr", "includeCutsel", "includeDefaultPlugins", "includeEventhdlr",
    "includeHeur", "includeIISfinder", "includeNodesel", "includePresol", "includePricer",
    "includeProp", "includeReader", "includeRelax", "includeSepa", "initBendersDefault",
    "interruptSolve", "isLPSolBasic", "isObjChangedProbing", "isObjIntegral", "isReoptEnabled",
    "lpiGetIterations", "markDoNotAggrVar", "markDoNotMultaggrVar", "newProbingNode", "optimize",
    "optimizeNogil", "presolve", "printCons", "printExternalCodeVersions", "printProblem",
    "printStatistics", "printStatisticsJson", "printVersion", "propagateProbing", "readParams",
    "readProblem", "redirectOutput", "relax", "repropagateNode", "resetParam",
    "resetParams", "restartSolve", "separateSol", "setBendersSubproblemIsConvex", "setBoolParam",
    "setCharParam", "setCheck", "setEmphasis", "setEnforced", "setHeurTiming",
    "setHeuristics", "setInitial", "setIntParam", "setLogfile", "setLongintParam",
    "setMaximize", "setMinimize", "setModifiable", "setObjIntegral", "setParam",
    "setParams", "setParamsCountsols", "setPresolve", "setProbName", "setRealParam",
    "setRemovable", "setSeparating", "setStringParam", "setupBendersSubproblem", "solveBendersSubproblem",
    "solveConcurrent", "solveProbingLP", "solveProbingLPWithPricing", "startProbing", "startStrongbranch",
    "to_ptr", "translateSubSol", "updateBendersLowerbounds", "updateVarPseudocost", "version",
    "writeName", "writeParams", "writeProblem", "writeStatistics", "writeStatisticsJson",
})

SUBCLASS_CONTRACT = frozenset({"addCons", "getSolObjVal"})

FP_MARKERS = re.compile(
    r"\bSCIP_Real\b|\bfloat\(|dtype=float|\bModel\b|\bExprCons\b|\bExpr\b|\bRow\b|\bColumn\b|\bNLRow\b|SCIPinfinity\("
)

FP_ALLOWED = frozenset({
    "addMatrixCons", "catchRowEvent", "create", "dropRowEvent", "freeBendersSubproblems",
    "getBranchScoreMultiple", "getCondition", "setParam", "solveBendersSubproblem",
})

MEMBER_RE = re.compile(r"^    (?:def|cdef|cpdef)\b.*?(\w+)\s*\(")


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


def test_base_model_surface_is_pinned():
    actual = {name for name in vars(BaseModel) if not name.startswith("__")}
    assert actual == BASE_MODEL_METHODS


def test_subclass_contract_is_implemented_by_model():
    assert SUBCLASS_CONTRACT <= set(vars(Model))


def test_base_model_calls_only_shared_methods():
    base = class_members(SCIP_PXI, "cdef class BaseModel:")
    model_only = set(class_members(MODEL_PXI, "cdef class Model(BaseModel):")) - set(base)
    offenders = {}
    for name, body in base.items():
        calls = set(re.findall(r"self\.(\w+)\(", body))
        bad = sorted((calls & model_only) - SUBCLASS_CONTRACT)
        if bad:
            offenders[name] = bad
    assert offenders == {}


def test_base_model_has_no_fp_markers():
    base = class_members(SCIP_PXI, "cdef class BaseModel:")
    hits = {name for name, body in base.items() if FP_MARKERS.search(body)}
    assert hits - FP_ALLOWED == set()


def test_model_subclass_and_mro():
    assert Model.__mro__[:2] == (Model, BaseModel)

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
