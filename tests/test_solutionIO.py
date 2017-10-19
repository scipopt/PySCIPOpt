from pyscipopt import Model
import pytest
from pytest import approx
import re

def create_model_with_one_optimum():
    m = Model()
    x = m.addVar("x", vtype = 'C', obj = 1.0)
    y = m.addVar("y", vtype = 'C', obj = 2.0)
    c = m.addCons(x + 2 * y >= 1.0)
    m.data = [x,y], [c]
    return m

def parse_solutionfile(filename):
    with open(str(filename), "r") as f:
        lines = f.readlines()

    solobj = lines[0].split(":")[1].strip()
    pattern = re.compile(r"^(?P<name>.*)\s+(?P<value>\d+(?:\.\d+)?)\s+\(obj:(?P<objectiv>\d+(?:\.\d+)?)\)$")
    variables = {}
    for line in lines[1:]:
        match = pattern.match(line)
        if match is None:
            raise Exception("unexpected or mallformated line in solution file:\n%s" % line)
        d = match.groupdict()
        variables[d['name'].strip()] = float(d['value']), float(d['objectiv'])
    return solobj, variables

def extract_solutionvalues(model, solution):
    solobj = model.getSolObjVal(solution)

    variables = {}
    for x in model.data[0]:
        val = model.getSolVal(solution, x)
        obj = x.getObj()
        variables[x.name] = float(val), float(obj)

    return solobj, variables

def validate_solutionfile(filename, model, solution):
    current = parse_solutionfile(filename)
    expected = extract_solutionvalues(model, solution)
    assert_equal_solutiondata(current, expected)

def assert_equal_solutions(current_model, current_sol, expected_model, expected_sol):
    current = extract_solutionvalues(current_model, current_sol)
    expected = extract_solutionvalues(expected_model, expected_sol)
    assert_equal_solutiondata(current, expected)

def assert_equal_solutiondata(current, expected):
    solobj, variables = current
    exp_solobj, exp_variables = expected

    assert approx(float(solobj)) == float(exp_solobj)

    for name, (val, obj) in exp_variables.items():
        if name in variables:
            exp_val, exp_obj = variables[name]
            assert val == approx(exp_val), \
                "variable '%s' has wrong value: expected %s got %s" % (name, val, exp_val)
            assert obj == approx(exp_obj), \
                "variable '%s' has wrong objectiv: expected %s got %s" % (name, obj, exp_obj)
        else:
            assert val == approx(0), \
                "variable '%s' was not in solutionfile even so its value is '%s' and not zero" % (name, val)


def test_writeBestSol(tmpdir):
    model = create_model_with_one_optimum()
    model.optimize()
    assert model.getStatus() == "optimal", "model could not be optimized"

    solfile = tmpdir.join("x.sol")
    model.writeBestSol(str(solfile))
    assert solfile.exists(), "no solution file was written"

    sol = model.getBestSol()
    validate_solutionfile(solfile, model, sol)


def test_writeAllSol(tmpdir):
    model = create_model_with_one_optimum()
    model.optimize()
    assert model.getStatus() == "optimal", "model could not be optimized"

    for idx, sol in enumerate(model.getSols()):
        solfile = tmpdir.join("x_%d.sol" % idx)
        model.writeSol(sol, str(solfile))
        assert solfile.exists() , "no solution file was written"
        validate_solutionfile(solfile, model, sol)

def test_readSolFile(tmpdir):
    model = create_model_with_one_optimum()
    model.optimize()
    assert model.getStatus() == "optimal", "model could not be optimized"

    solfile = tmpdir.join("x.sol")
    model.writeBestSol(str(solfile))
    assert solfile.exists(), "no solution file was written"

    exp_sol = model.getBestSol()

    model2 = create_model_with_one_optimum()
    sol = model2.readSolFile(str(solfile))

    assert_equal_solutions(model2, sol, model, exp_sol)


def test_useReadSol(tmpdir):
    model = create_model_with_one_optimum()
    model.optimize()
    assert model.getStatus() == "optimal", "model could not be optimized"

    solfile = tmpdir.join("x.sol")
    model.writeBestSol(str(solfile))
    assert solfile.exists(), "no solution file was written"

    statfile = tmpdir.join("x.stats")
    model.writeStatistics(str(statfile))
    assert statfile.exists(), "no statistics file was written"

    # second model instance
    model2 = create_model_with_one_optimum()
    assert len(model2.getSols()) == 0

    sol = model2.readSolFile(str(solfile))
    model2.addSol(sol)
    assert len(model2.getSols()) == 1

    model2.optimize()

    statfile2 = tmpdir.join("x2.stats")
    model2.writeStatistics(str(statfile2))
    assert statfile2.exists(), "no statistics file was written"

    pattern = re.compile(r"Solutions\sfound\s+:\s+(\d+)\s+\((\d+)\s+improvements\)")
    with open(str(statfile), "r") as f1:
        with open(str(statfile2), "r") as f2:
            match1 = pattern.search(f1.read())
            match2 = pattern.search(f2.read())
            assert match1 is not None and match2 is not None, "statistics files are incomplete"
            assert match1.group(1,2) != match2.group(1,2) , \
                "the number of solutions and improvements should be differing"

#TODO: implement these

def test_readSolWithWrongModel():
    pass

def test_readSolWithInvalidVarnames():
    pass

def test_readSolAtWrongStage():
    pass
