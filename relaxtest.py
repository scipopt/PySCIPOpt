#! usr/bin/env python3
from pyscipopt      import Model, SCIP_PARAMSETTING, Expr, Relax, Term, ExprCons
from SONCrelaxator  import SoncRelax
from POEM.python    import Polynomial, build_lagrangian, constrained_opt

import re
import numpy as np

def example():
    m = Model()
    #m.hideOutput()
    m.setPresolve(SCIP_PARAMSETTING.OFF)
    m.setHeuristics(SCIP_PARAMSETTING.OFF)
    m.disablePropagation()
    #m.setIntParam('lp/disablecutoff', 1)
    #m.setBoolParam('nlp/disable', True)
    #m.setIntParam('lp/solvefreq', -1)

    #add Variables
    x0 = m.addVar(vtype = "C", name = "x0")
    x1 = m.addVar(vtype = "C", name = "x1")
    x2 = m.addVar(vtype = "C", name = "x2")
    #x3 = m.addVar(vtype = "C", name = "x3")
    #x4 = m.addVar(vtype = "C", name = "x4")
    #x5 = m.addVar(vtype = "C", name = "x5")


    #TODO: find SONC polynomial with linear term, to split into objective - constraint
    #TODO: find examples with more variables
    """
    #shows that relaxator is really used, time ~17 sec #4.15sec
    #p = 7.93823555685 + 1.80070743765*x0**6 + 4.40432092848*x1**6 + 0.950088417526*x0**2*x1**2 + 2.2408931992*x0**1*x1**2 + 1.86755799015*x0**1*x1**4 - 0.977277879876*x0**2*x1**1 - 0.151357208298*x0**3*x1**1  - 0.103218851794*x0**4*x1**1
    #m.addCons(p<= 0)
    #m.addCons(x1-x0>=0)
    m.addCons(x0**2*x1**2+x0*x1+x0**4+x1**4-x0<=10)
    m.setObjective(-2*x0+x1)
    """
    """
    #SONC leads to 'cutoff' in SCIP (lowerbound = 5.21), hence infeasible, SCIP alone does not terminate (spatial branch and bound), time ~ 11 sec
    m.addCons(7.10008969*x1**2 + 8.53135095*x0**6 + 1.9507754*x0**4 - 0.50965218*x0*x1 - 0.4380743*x0**3  <= 0)
    m.addCons(0.77749036*x0**2 - 1.61389785*x0**3*x1 - 0.21274028*x0**2*x1 <= 0)
    m.setObjective(5.24276483 - 1.25279536*x1)
    """
    """
    #SONC get some cutoff even though lowerbound < solution?? (leads to infeasibility even though SCIP without relaxator finds solution), time ~ 30 sec #6.28sec
    m.addCons(4.27619696*x1**2 + 8.87643816*x0**6 - 0.21274028*x0**4 + 0.3869025*x0*x1  <= 0)
    m.addCons(- 0.51080514*x0**3  + 0.06651722*x0**2*x1 <= 0)
    m.addCons(-0.02818223*x0**2 + 0.42833187*x0**3*x1 <= 0)
    m.setObjective(6.89037448 -0.89546656*x0 - 1.18063218*x1)
    """
    """
    #reasonable solution, time ~ 18 sec #5.34sec (LP first 0.00)
    m.addCons(1.16786015*x1**2 + 5.22927676*x0**2*x1**2 + 0.71805392*x0**4 <= 0)
    m.addCons( - 0.85409574*x0**2*x1 -2.55298982*x0*x1 <= 0)
    m.setObjective(1.55352131 +0.3130677*x1)
    """
    """
    #unbounded x1 since occurs only with exponent 1, hence SONC does not work, time ~ 0.1sec
    m.addCons(x0 <= 3)
    m.addCons(x1 <= 4)
    m.addCons(8*x0**3 - 2*x0**4 - 8*x0**2 + x1 >= 2)
    m.addCons(32*x0**3 - 4*x0**4 - 88*x0**2 + 96*x0 + x1 >= 36)
    m.setObjective(-x0-x1)
    """
    """
    #Problem only containing linear and quadratic constraints, using sonc anyway shows that relaxator is really used (if do not return if polycons==False), time ~ 21secs #3.91sec
    m.addCons(0.25*x1 - 0.0625*x1**2 - 0.0625*x0**2 + 0.5*x0 <= 1)
    m.addCons(0.0714285714285714*x1**2 + 0.0714285714285714*x0**2 - 0.428571428571429*x1 - 0.428571428571429*x0 <= -1)
    #m.addCons((1<=x0)<=5.5)
    #m.addCons((1<=x1)<=5.5)
    m.chgVarLb(x0,1)
    m.chgVarLb(x1,1)
    m.chgVarUb(x0,5.5)
    m.chgVarUb(x1,5.5)
    m.setObjective(x1)
    """
    """
    #terminates, gives reasonable lower bound, time ~ 10sec #3.81sec (LP first 0.00)
    m.addCons(0<=0.5+x0**2*x1**1-x0**6*x1**4-x0**3*x1**3)
    m.addCons(-x0<=1-x0**4-x0**2*x1**4)
    m.setObjective(3+x0)
    """
    """
    #(a) Test Polynomials modified, terminates, good lower bound (exact to 5sf.), time ~ 24sec #5.72sec
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    m.addCons(x0**2*x1 + 3*x1 >=0)
    m.setObjective(-x0)
    """
    """
    #(b) Test Polynomials modified, reasonable lower bound, time ~ 7sec #4.92sec
    m.addCons(-2*x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x1**4+42>=0)
    m.addCons(-x0**2*x1 - 3*x1 +0.5*x0 >=0)
    m.setObjective(-0.5*x0)
    """
    """
    #(c) Test Polynomials modified, terminates, time ~ 25sec #faster if using orthant=[(0,1)] in POEM #6.30sec
    m.addCons(-x0**4+3*x0-2*x1**2+1>=0)
    m.addCons(-x0**4-x1**4+42>=0)
    #m.addCons(x0**2*x1 >=0) #if added, time ~ 30sec
    m.setObjective(3*x1-x0)
    """
    """
    #Example 4.8 (6)i. in Lower Bounds for a polynomial (Ghasemi, Marshall), lower bound: -1.6, sol: 0.0, time ~ 15sec #4.53sec (LP first 0.00)
    m.addCons(1-2*x1-6*x0**2-x0**4>=0)
    m.addCons(-x0**3-x1**4>=0)
    m.setObjective(x0+x1)
    """
    """
    #Example 4.8 (8), good bound (-7.01e-05, sol: -9.06e-07), time ~ 14sec #only one sol (instead of 3) probably due to rounding error #2.99sec
    m.addCons(2*x0**2+x1>=0)
    m.addCons(x1-x0**4*x1+x1**5-x0**6-x1**6>=0)
    m.addCons(x1-5*x0**2+x0**4*x1-x0**6-x1**6>=0)
    m.setObjective(-x1-1)
    """
    """
    #terminates, reasonable lower bound, time ~ 8sec #3.04sec (LP first 0.00)
    m.addCons(1-x0**4-x1**4>=0)
    m.addCons(x0**3+2>=0)
    m.setObjective(x0-2)
    """
    """
    #ex14_1_1, unbounded point x3, 4 variables, time ~ 0.1sec
    m.addCons(-x3+x0 <= 0)
    m.addCons(-x3+x0 >= 0)
    m.addCons(x3**2 + x0**2 +x2**2 <= 10)
    m.addCons(2*x2**2 + 4*x1*x2 - 42*x1 + 4*x1**3 - x3 <= 14)
    m.addCons(-2*x2**2 - 4*x1*x2 + 42*x1 - 4*x1**3 - x3 <= -14)
    m.addCons(2*x1**2 + 4*x1*x2 - 26*x2 + 4*x2**3 - x3 <= 22)
    m.addCons(-2*x1**2 - 4*x1*x2 + 6*x2 - 4*x2**3 - x3 <= -54)
    m.addCons((-5<=x1)<=5)
    m.addCons((-5<=x2)<=5)
    m.setObjective(x0)
    """
    """
    #Example 3.2 Ghasemi, Marshall, extended, time ~ 30sec #7.38sec (LP first 0.00)
    m.addCons(x0**2-2*x0*x1+x1**2>=0)
    m.addCons(-x0**4+x1-x0**3-x1**4+20+x0**2*x1>=0)
    m.setObjective(x0+x1)
    """
    """
    #Example instances/MINLP/circle.cip changed #3.62sec
    m.addCons(-(x0+2.545724188)**2 + (x1-9.983058643)**2 <= -x2**2)
    m.addCons((x0-8.589400372)**2 - (x1-6.208600402)**2  <= x2**2)
    m.addCons((x0-5.953378204)**2 + 4*(x1-9.920197351)**2 <= x2**2)
    m.addCons((x0-3.710241136)**2 - (x1-7.860254203)**2 <= -x2**2)
    m.addCons(-8.88178419700125e-16 + (x0-3.629909053)**2 - (x1-2.176232347)**2  <= -x2**2)
    #m.addCons((x0-3.016475803)**2 + (x1-6.757468831)**2 <= x2**2)
    #m.addCons(8.88178419700125e-16+ (x0-4.148474536)**2 + (x1-2.435660776)**2 <= x2**2)
    #m.addCons((x0-8.706433123)**2 + (x1-3.250724797)**2 <= x2**2)
    #m.addCons((x0-1.604023507)**2 + (x1-7.020357481)**2 <= x2**2)
    #m.addCons((x0-5.501896021)**2 + (x1-4.918207429)**2 <= x2**2)
    m.setObjective(x2)
    """
    """
    m.addCons(10.755075611691005 + 7.063822083711873*x4**2 + 4.961875620708473*x3**2*x4**2 + 10.221991899856631*x2**4 + 15.76242393960385*x1**2*x4**2 + 0.40016657123676586*x1**4 + 11.129187001117488*x0**2*x3**2 + 5.034052830118634*x0**4 - 0.7484618472380008*x2**2 + 0.21440675171257084*x1**2*x2**2 + 0.020626390018846214*x0**2 + 0.6813036508255289*x0**2*x2**2 - 0.5979961657253435*x0**2*x1**2 + 0.5682771044569751*x0**1*x4**1 - 0.18971143811048619*x0**1*x3**1*x4**1 - 0.061096149668652164*x1**1*x3**1*x4**1 + 0.3320310653535568*x0**1*x2**1*x3**1 - 1.9406747544749365*x0**1*x2**1*x4**1 - 0.7525207920686302*x0**1*x1**1 + 1.0434961844464452*x0**1*x1**1*x2**2 - 0.37460183560912585*x0**1*x1**1*x3**1*x4**1 + 0.5651697025660114*x1**1*x2**1*x4**1 + 0.5554162457504477*x0**1*x2**1 + 0.9496691100842614*x0**2*x1**1*x2**1 + 1.2916407373152514*x0**1 + 1.445097436469295*x0**1*x1**1*x2**1*x3**1 + 1.3328384398532207*x0**1*x2**1*x3**1*x4**1 + 0.4564044489006344*x1**1*x2**1 + 1.4189228838010666*x0**1*x1**1*x2**1 + 0.8040100398985036*x0**1*x1**1*x4**1 - 1.2569810836440913*x0**1*x1**1*x2**1*x4**1<=22)
    m.addCons(0.19775869019553727 + 0.3801208227588542*x4**2 + 0.11395588550403411*x3**2 + 0.38669778347940703*x2**2 + 1.2595283166164397*x1**2<=48)
    m.addCons(0.7684712106349337 + 0.324200060305769*x4**2 + 0.2296587888788862*x3**2 + 1.4686586565748996*x1**2 + 1.2678986270957202*x0**2<=36)
    m.addCons(1.0658789874173467 + 2.286743480583174*x4**2 + 1.4945159131688759*x3**2 + 1.6748012990867733*x2**2 + 0.04951887483046802*x1**2 + 3.6936202427614884*x0**2 + 1.3860574977441495*x0**1 - 2.4013171279993952*x2**1*x3**1 + 0.8694056264642895*x4**1 + 0.9638926214374866*x3**1*x4**1 - 0.5180359365652341*x1**1*x3**1 + 1.0346036809110681*x1**1*x4**1 + 0.7811421659242989*x2**1*x4**1 + 0.22546420630904268*x3**1<=10)
    m.addCons(5.8985083815221495 + 0.649834209423522*x4**4 + 13.749962825297802*x3**4 + 2.042667968154976*x2**4 + 12.561947935709977*x1**4 + 5.3423162502508035*x0**2 + 3.0855684081830734*x0**2*x4**2 + 4.245272843454879*x0**2*x3**2 + 7.49502658231706*x0**2*x2**2 + 1.1188085225361766*x0**2*x1**2 + 2.140378371501857*x4**2 + 1.1993495460966774*x3**2 - 0.09287519229865791*x3**2*x4**2 - 1.0860253762142618*x2**2 - 0.8676940736567079*x2**2*x4**2 - 2.268977316855398*x2**2*x3**2 - 0.11210713406598985*x1**2 + 1.4627262630797666*x1**2*x4**2 - 0.7065257221032553*x1**2*x3**2 + 0.6626846989016041*x1**2*x2**2 + 0.49191327477713453*x1**1*x2**1*x4**1 - 0.6291227722266814*x2**1*x4**1 - 0.8149912255562152*x0**1*x1**1*x3**1*x4**1 + 1.1222158691846984*x2**1*x3**1*x4**1 - 1.2079501007663722*x1**1*x2**1*x3**1 - 0.42809788075746485*x0**1*x1**1*x2**1*x4**1 - 0.7939834125140984*x0**1*x2**1*x3**1*x4**1 - 0.7729837863104726*x0**1*x1**1*x4**1 + 0.32572548225308484*x1**1*x3**1*x4**1 - 0.027605717103113037*x0**1*x1**1*x2**1*x3**1 - 2.1621065272757733*x1**1*x2**1*x3**1*x4**1<=36)
    m.setObjective(0.0 - 1.0*x1**1.0 + 1.0*x2**1.0 + 1.0*x3**1.0 - 1.0*x4**1.0)
    """
    """
    m.addCons(1.3664974114918205 + 0.9221111862427721*x5**2 + 0.9082578790732385*x4**2 + 1.9537348320612784*x3**2 + 1.9191613578792153*x2**2 + 1.1837475625189882*x1**2 + 1.4844919657475493*x0**2 + 1.0760120635086576*x5**1<=31)
    m.addCons(1.0782404272815291 + 1.8950280074303962*x5**4 + 0.23538166156418824*x3**4 + 0.6540572898876087*x2**2 + 2.371281473874693*x1**2*x5**2 + 0.4807956455997765*x0**2*x1**2 + 1.602527385772485*x3**2 + 1.86964275196167*x1**1*x5**2 + 1.4400412484151541*x3**1*x5**1<=19)
    m.addCons(1.7445717057204921 + 0.797680496679996*x4**4 + 0.8519789295292074*x2**2 + 0.39143610404881574*x1**2 + 3.1369845408888315*x1**2*x2**2 + 1.9045700065138427*x0**2*x4**2 + 2.2230199409917213*x0**2*x1**2 - 0.14699117328644629*x0**1*x1**1*x4**2 - 0.8160783419943797*x0**1*x1**1*x2**1*x4**1 - 1.3352903884325746*x1**1*x2**1*x4**1<=26)
    m.setObjective(0.0 - 1.0*x0**1.0 - 1.0*x1**1.0 + 1.0*x3**1.0 + 1.0*x5**1.0)
    """
    """
    m.addCons(7.371389774337124 + 0.8377689605847962*x4**4 + 0.9438158769233826*x3**4 + 6.043950814717542*x2**4 + 10.986511858765803*x1**4 + 1.1439384448459846*x0**4 + 1.4028225204785072*x4**2 + 0.15222786760564846*x3**2*x4**2 + 0.8755199544644061*x2**2 + 2.0818944929182512*x2**2*x4**2 + 0.2405760127797441*x2**2*x3**2 - 0.19349589595150815*x1**2*x4**2 - 1.7743403463032474*x1**2*x3**2 + 1.0558473346366768*x1**2*x2**2 - 0.8017222812290635*x0**2 - 1.2406652963532026*x0**2*x4**2 + 0.9775873358182247*x0**2*x3**2 - 1.5194649237993785*x0**2*x2**2 - 0.36708914248408053*x0**2*x1**2 + 0.5890150511676189*x0**1*x1**1*x4**1 - 1.1677473767621216*x0**1*x1**1*x2**1 + 0.9184884824963808*x1**1*x2**1*x3**1 + 0.5669623064368925*x0**1*x1**1*x3**1*x4**1 - 0.6834862788376451*x1**1*x2**1*x4**1 - 1.6572959866890093*x0**1*x2**1*x3**1 - 0.7604070595678674*x0**1*x2**1*x4**1 - 0.1854137016319008*x1**1*x2**1*x3**1*x4**1 - 0.47315291857325814*x2**1*x3**1*x4**1 + 0.8798227578179353*x0**1*x2**1*x3**1*x4**1 + 1.5926534630762015*x0**1*x1**1*x2**1*x4**1 - 1.356625953940452*x0**1*x1**1*x2**1*x3**1<=17)
    m.addCons(0.6011375383597662 + 0.7427133235230527*x4**2 + 1.509391012118035*x2**2 + 0.31920771628646943*x1**2 + 1.727252871968567*x0**2<=16)
    m.setObjective(0.0 + 1.0*x1**1.0 - 1.0*x3**1.0 + 1.0*x4**1.0)
    """
    return m

if __name__=="__main__":
    m = example()
    relaxator = SoncRelax()
    #which frequency is reasionable/sufficient? maybe also change priority param?
    m.includeRelax(relaxator,"SONCRelaxator", "Relaxator using SONC decompositions", freq=1) #priority=-1
    m.optimize()
    var = m.getVars()
    for v in var:
        print(v,"=", m.getVal(v))
    print("objective = ", m.getObjVal())

    n = example()
    n.disablePropagation()
    n.optimize()
    va = n.getVars()
    for v in va:
        print(v, "=", n.getVal(v))
    print("objective = ", n.getObjVal())
