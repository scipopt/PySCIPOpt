import pytest

from pyscipopt import Model

def test_constraint_option_setting():
    m = Model()
    x = m.addVar()
    c = m.addCons(x >= 3)

    for option in [True, False]:
        m.setCheck(c, option)
        m.setEnforced(c, option)
        m.setRemovable(c, option)
        m.setInitial(c, option)

        assert c.isChecked() == option
        assert c.isEnforced() == option
        assert c.isRemovable() == option
        assert c.isInitial() == option
