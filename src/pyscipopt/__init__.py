__version__ = '3.3.0'

# required for Python 3.8 on Windows
import os
if hasattr(os, 'add_dll_directory'):
    if os.getenv('SCIPOPTDIR'):
        os.add_dll_directory(os.path.join(os.getenv('SCIPOPTDIR').strip('"'), 'bin'))

# export user-relevant objects:
from pyscipopt.Multidict import multidict
from pyscipopt.scip      import Model
from pyscipopt.scip      import Benders
from pyscipopt.scip      import Benderscut
from pyscipopt.scip      import Branchrule
from pyscipopt.scip      import Nodesel
from pyscipopt.scip      import Conshdlr
from pyscipopt.scip      import Eventhdlr
from pyscipopt.scip      import Heur
from pyscipopt.scip      import Presol
from pyscipopt.scip      import Pricer
from pyscipopt.scip      import Prop
from pyscipopt.scip      import Sepa
from pyscipopt.scip      import LP
from pyscipopt.scip      import Expr
from pyscipopt.scip      import quicksum
from pyscipopt.scip      import quickprod
from pyscipopt.scip      import exp
from pyscipopt.scip      import log
from pyscipopt.scip      import sqrt
from pyscipopt.scip      import PY_SCIP_RESULT       as SCIP_RESULT
from pyscipopt.scip      import PY_SCIP_PARAMSETTING as SCIP_PARAMSETTING
from pyscipopt.scip      import PY_SCIP_PARAMEMPHASIS as SCIP_PARAMEMPHASIS
from pyscipopt.scip      import PY_SCIP_STATUS       as SCIP_STATUS
from pyscipopt.scip      import PY_SCIP_STAGE        as SCIP_STAGE
from pyscipopt.scip      import PY_SCIP_PROPTIMING   as SCIP_PROPTIMING
from pyscipopt.scip      import PY_SCIP_PRESOLTIMING as SCIP_PRESOLTIMING
from pyscipopt.scip      import PY_SCIP_HEURTIMING   as SCIP_HEURTIMING
from pyscipopt.scip      import PY_SCIP_EVENTTYPE    as SCIP_EVENTTYPE
from pyscipopt.scip      import PY_SCIP_LPSOLSTAT    as SCIP_LPSOLSTAT
from pyscipopt.scip      import PY_SCIP_BRANCHDIR    as SCIP_BRANCHDIR
from pyscipopt.scip      import PY_SCIP_BENDERSENFOTYPE as SCIP_BENDERSENFOTYPE
from pyscipopt.scip      import PY_SCIP_ROWORIGINTYPE as SCIP_ROWORIGINTYPE
