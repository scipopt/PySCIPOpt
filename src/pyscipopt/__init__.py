from ._version import __version__

# required for Python 3.8 on Windows
import os
if hasattr(os, 'add_dll_directory'):
    if os.getenv('SCIPOPTDIR'):
        os.add_dll_directory(os.path.join(os.environ['SCIPOPTDIR'].strip('"'), 'bin'))

# export user-relevant objects:
from pyscipopt.Multidict import multidict as multidict
from pyscipopt.scip      import Model as Model
from pyscipopt.scip      import Variable as Variable
from pyscipopt.scip      import MatrixVariable as MatrixVariable
from pyscipopt.scip      import Constraint as Constraint
from pyscipopt.scip      import MatrixConstraint as MatrixConstraint
from pyscipopt.scip      import Benders as Benders
from pyscipopt.scip      import Benderscut as Benderscut
from pyscipopt.scip      import Branchrule as Branchrule
from pyscipopt.scip      import Nodesel as Nodesel
from pyscipopt.scip      import Conshdlr as Conshdlr
from pyscipopt.scip      import Eventhdlr as Eventhdlr
from pyscipopt.scip      import Heur as Heur
from pyscipopt.scip      import Presol as Presol
from pyscipopt.scip      import Pricer as Pricer
from pyscipopt.scip      import Prop as Prop
from pyscipopt.scip      import Reader as Reader
from pyscipopt.scip      import Sepa as Sepa
from pyscipopt.scip      import LP as LP
from pyscipopt.scip      import IISfinder as IISfinder 
from pyscipopt.scip      import PY_SCIP_LPPARAM as SCIP_LPPARAM
from pyscipopt.scip      import readStatistics as readStatistics
from pyscipopt.scip      import Expr as Expr
from pyscipopt.scip      import MatrixExpr as MatrixExpr
from pyscipopt.scip      import MatrixExprCons as MatrixExprCons
from pyscipopt.scip      import ExprCons as ExprCons
from pyscipopt.scip      import quicksum as quicksum
from pyscipopt.scip      import quickprod as quickprod
from pyscipopt.scip      import exp as exp
from pyscipopt.scip      import log as log
from pyscipopt.scip      import sqrt as sqrt
from pyscipopt.scip      import sin as sin
from pyscipopt.scip      import cos as cos
from pyscipopt.scip      import PY_SCIP_RESULT          as SCIP_RESULT
from pyscipopt.scip      import PY_SCIP_PARAMSETTING    as SCIP_PARAMSETTING
from pyscipopt.scip      import PY_SCIP_PARAMEMPHASIS   as SCIP_PARAMEMPHASIS
from pyscipopt.scip      import PY_SCIP_STATUS          as SCIP_STATUS
from pyscipopt.scip      import PY_SCIP_STAGE           as SCIP_STAGE
from pyscipopt.scip      import PY_SCIP_NODETYPE        as SCIP_NODETYPE
from pyscipopt.scip      import PY_SCIP_PROPTIMING      as SCIP_PROPTIMING
from pyscipopt.scip      import PY_SCIP_PRESOLTIMING    as SCIP_PRESOLTIMING
from pyscipopt.scip      import PY_SCIP_HEURTIMING      as SCIP_HEURTIMING
from pyscipopt.scip      import PY_SCIP_EVENTTYPE       as SCIP_EVENTTYPE
from pyscipopt.scip      import PY_SCIP_LOCKTYPE        as SCIP_LOCKTYPE
from pyscipopt.scip      import PY_SCIP_LPSOLSTAT       as SCIP_LPSOLSTAT
from pyscipopt.scip      import PY_SCIP_BRANCHDIR       as SCIP_BRANCHDIR
from pyscipopt.scip      import PY_SCIP_BENDERSENFOTYPE as SCIP_BENDERSENFOTYPE
from pyscipopt.scip      import PY_SCIP_ROWORIGINTYPE as SCIP_ROWORIGINTYPE
from pyscipopt.scip      import PY_SCIP_SOLORIGIN as SCIP_SOLORIGIN
from pyscipopt.scip      import PY_SCIP_NODETYPE as SCIP_NODETYPE
from pyscipopt.scip      import PY_SCIP_IMPLINTTYPE as SCIP_IMPLINTTYPE
