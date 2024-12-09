##############
SCIP Log Files
##############

For the following let us assume that we have called ``optimize()`` on a SCIP Model.
When running, SCIP outputs a constant stream of information on the current state of the
optimization process.

.. contents:: Contents

How to Read SCIP Output
=======================

Let's consider the example complete output below:

.. code-block:: RST

    presolving:
    (round 1, fast)       136 del vars, 0 del conss, 2 add conss, 0 chg bounds, 0 chg sides, 0 chg coeffs, 0 upgd conss, 0 impls, 0 clqs
    (round 2, fast)       136 del vars, 1 del conss, 2 add conss, 0 chg bounds, 132 chg sides, 0 chg coeffs, 0 upgd conss, 0 impls, 0 clqs
    (round 3, exhaustive) 136 del vars, 2 del conss, 2 add conss, 0 chg bounds, 133 chg sides, 0 chg coeffs, 0 upgd conss, 0 impls, 0 clqs
    (round 4, exhaustive) 136 del vars, 2 del conss, 2 add conss, 0 chg bounds, 133 chg sides, 0 chg coeffs, 131 upgd conss, 0 impls, 0 clqs
       (0.0s) probing cycle finished: starting next cycle
       (0.0s) symmetry computation started: requiring (bin +, int +, cont +), (fixed: bin -, int -, cont -)
       (0.0s) no symmetry present (symcode time: 0.00)
    presolving (5 rounds: 5 fast, 3 medium, 3 exhaustive):
     136 deleted vars, 2 deleted constraints, 2 added constraints, 0 tightened bounds, 0 added holes, 133 changed sides, 0 changed coefficients
     231 implications, 0 cliques
    presolved problem has 232 variables (231 bin, 0 int, 1 impl, 0 cont) and 137 constraints
         53 constraints of type <knapsack>
          6 constraints of type <linear>
         78 constraints of type <logicor>
    transformed objective value is always integral (scale: 1)
    Presolving Time: 0.01

     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      0.0s|     1 |     0 |   409 |     - |  5350k |   0 | 232 | 156 | 137 |   0 |  0 |  19 |   0 | 7.649866e+03 |      --      |    Inf | unknown
    o 0.0s|     1 |     0 |  1064 |     - |feaspump|   0 | 232 | 156 | 137 |   0 |  0 |  19 |   0 | 7.650000e+03 | 8.267000e+03 |   8.07%| unknown
      0.0s|     1 |     0 |  1064 |     - |  5368k |   0 | 232 | 156 | 137 |   0 |  0 |  19 |   0 | 7.650000e+03 | 8.267000e+03 |   8.07%| unknown
      0.0s|     1 |     0 |  1064 |     - |  5422k |   0 | 232 | 156 | 137 |   0 |  0 |  19 |   0 | 7.650000e+03 | 8.267000e+03 |   8.07%| unknown
      0.0s|     1 |     0 |  1067 |     - |  5422k |   0 | 232 | 156 | 137 |   0 |  0 |  19 |   0 | 7.650000e+03 | 8.267000e+03 |   8.07%| unknown
      0.1s|     1 |     0 |  1132 |     - |  9912k |   0 | 232 | 156 | 138 |   1 |  1 |  19 |   0 | 7.659730e+03 | 8.267000e+03 |   7.93%| unknown
      0.1s|     1 |     0 |  1133 |     - |  9924k |   0 | 232 | 157 | 138 |   1 |  1 |  20 |   0 | 7.660000e+03 | 8.267000e+03 |   7.92%| unknown
      0.1s|     1 |     0 |  1134 |     - |  9924k |   0 | 232 | 157 | 138 |   1 |  1 |  20 |   0 | 7.660000e+03 | 8.267000e+03 |   7.92%| unknown
      0.1s|     1 |     0 |  1210 |     - |    15M |   0 | 232 | 157 | 141 |   4 |  2 |  20 |   0 | 7.671939e+03 | 8.267000e+03 |   7.76%| unknown
      0.1s|     1 |     0 |  1213 |     - |    15M |   0 | 232 | 159 | 141 |   4 |  2 |  22 |   0 | 7.672000e+03 | 8.267000e+03 |   7.76%| unknown
      0.1s|     1 |     0 |  1280 |     - |    18M |   0 | 232 | 157 | 143 |   6 |  3 |  22 |   0 | 7.685974e+03 | 8.267000e+03 |   7.56%| unknown
      0.1s|     1 |     0 |  1282 |     - |    18M |   0 | 232 | 157 | 143 |   6 |  3 |  22 |   0 | 7.686000e+03 | 8.267000e+03 |   7.56%| unknown
      0.2s|     1 |     0 |  1353 |     - |    21M |   0 | 232 | 156 | 145 |   8 |  4 |  22 |   0 | 7.701524e+03 | 8.267000e+03 |   7.34%| unknown
      0.2s|     1 |     0 |  1355 |     - |    21M |   0 | 232 | 156 | 145 |   8 |  4 |  22 |   0 | 7.702000e+03 | 8.267000e+03 |   7.34%| unknown
      0.2s|     1 |     0 |  1435 |     - |    24M |   0 | 232 | 156 | 147 |  10 |  5 |  22 |   0 | 7.706318e+03 | 8.267000e+03 |   7.28%| unknown
     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      0.2s|     1 |     0 |  1438 |     - |    24M |   0 | 232 | 158 | 147 |  10 |  5 |  24 |   0 | 7.707000e+03 | 8.267000e+03 |   7.27%| unknown
      0.2s|     1 |     0 |  1520 |     - |    30M |   0 | 232 | 158 | 149 |  12 |  6 |  24 |   0 | 7.711108e+03 | 8.267000e+03 |   7.21%| unknown
      0.2s|     1 |     0 |  1521 |     - |    30M |   0 | 232 | 158 | 149 |  12 |  6 |  24 |   0 | 7.712000e+03 | 8.267000e+03 |   7.20%| unknown
      0.2s|     1 |     0 |  1658 |     - |    34M |   0 | 232 | 158 | 151 |  14 |  7 |  24 |   0 | 7.715238e+03 | 8.267000e+03 |   7.15%| unknown
      0.2s|     1 |     0 |  1659 |     - |    34M |   0 | 232 | 158 | 151 |  14 |  7 |  24 |   0 | 7.716000e+03 | 8.267000e+03 |   7.14%| unknown
      0.3s|     1 |     0 |  1770 |     - |    40M |   0 | 232 | 158 | 153 |  16 |  8 |  24 |   0 | 7.717854e+03 | 8.267000e+03 |   7.12%| unknown
      0.3s|     1 |     0 |  1771 |     - |    40M |   0 | 232 | 158 | 153 |  16 |  8 |  24 |   0 | 7.718000e+03 | 8.267000e+03 |   7.11%| unknown
      0.3s|     1 |     0 |  1883 |     - |    40M |   0 | 232 | 157 | 154 |  17 |  9 |  24 |   0 | 7.730185e+03 | 8.267000e+03 |   6.94%| unknown
      0.3s|     1 |     0 |  1884 |     - |    40M |   0 | 232 | 157 | 154 |  17 |  9 |  24 |   0 | 7.731000e+03 | 8.267000e+03 |   6.93%| unknown
      0.3s|     1 |     0 |  1925 |     - |    46M |   0 | 232 | 157 | 156 |  19 | 10 |  24 |   0 | 7.734301e+03 | 8.267000e+03 |   6.89%| unknown
      0.3s|     1 |     0 |  1926 |     - |    46M |   0 | 232 | 157 | 152 |  19 | 10 |  24 |   0 | 7.735000e+03 | 8.267000e+03 |   6.88%| unknown
      0.3s|     1 |     0 |  1946 |     - |    46M |   0 | 232 | 157 | 154 |  21 | 11 |  24 |   0 | 7.735000e+03 | 8.267000e+03 |   6.88%| unknown
      0.4s|     1 |     0 |  1972 |     - |    46M |   0 | 232 | 157 | 156 |  23 | 12 |  24 |   0 | 7.735275e+03 | 8.267000e+03 |   6.87%| unknown
      0.4s|     1 |     0 |  1973 |     - |    46M |   0 | 232 | 158 | 156 |  23 | 12 |  25 |   0 | 7.736000e+03 | 8.267000e+03 |   6.86%| unknown
      0.4s|     1 |     0 |  2007 |     - |    46M |   0 | 232 | 158 | 157 |  24 | 13 |  25 |   0 | 7.736000e+03 | 8.267000e+03 |   6.86%| unknown
     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      0.4s|     1 |     0 |  2057 |     - |    46M |   0 | 232 | 155 | 158 |  25 | 14 |  27 |   0 | 7.737403e+03 | 8.267000e+03 |   6.84%| unknown
      0.4s|     1 |     0 |  2058 |     - |    46M |   0 | 232 | 155 | 153 |  25 | 14 |  27 |   0 | 7.738000e+03 | 8.267000e+03 |   6.84%| unknown
      0.4s|     1 |     0 |  2086 |     - |    46M |   0 | 232 | 155 | 154 |  26 | 15 |  27 |   0 | 7.738004e+03 | 8.267000e+03 |   6.84%| unknown
      0.4s|     1 |     0 |  2093 |     - |    46M |   0 | 232 | 155 | 156 |  28 | 16 |  27 |   0 | 7.738165e+03 | 8.267000e+03 |   6.83%| unknown
      0.4s|     1 |     0 |  2094 |     - |    46M |   0 | 232 | 156 | 156 |  28 | 16 |  28 |   0 | 7.739000e+03 | 8.267000e+03 |   6.82%| unknown
      0.5s|     1 |     0 |  2146 |     - |    46M |   0 | 232 | 156 | 157 |  29 | 17 |  28 |   0 | 7.739168e+03 | 8.267000e+03 |   6.82%| unknown
      0.5s|     1 |     0 |  2147 |     - |    46M |   0 | 232 | 156 | 157 |  29 | 17 |  28 |   0 | 7.740000e+03 | 8.267000e+03 |   6.81%| unknown
      0.5s|     1 |     0 |  2178 |     - |    46M |   0 | 232 | 156 | 157 |  30 | 18 |  28 |   0 | 7.740000e+03 | 8.267000e+03 |   6.81%| unknown
      0.5s|     1 |     0 |  2223 |     - |    46M |   0 | 232 | 157 | 159 |  32 | 19 |  29 |   0 | 7.740575e+03 | 8.267000e+03 |   6.80%| unknown
      0.5s|     1 |     0 |  2224 |     - |    46M |   0 | 232 | 157 | 159 |  32 | 19 |  29 |   0 | 7.741000e+03 | 8.267000e+03 |   6.79%| unknown
      0.5s|     1 |     0 |  2259 |     - |    46M |   0 | 232 | 157 | 160 |  33 | 20 |  29 |   0 | 7.741000e+03 | 8.267000e+03 |   6.79%| unknown
      0.5s|     1 |     0 |  2282 |     - |    46M |   0 | 232 | 157 | 161 |  34 | 21 |  29 |   0 | 7.741495e+03 | 8.267000e+03 |   6.79%| unknown
      0.5s|     1 |     0 |  2283 |     - |    46M |   0 | 232 | 157 | 161 |  34 | 21 |  29 |   0 | 7.742000e+03 | 8.267000e+03 |   6.78%| unknown
      0.5s|     1 |     0 |  2300 |     - |    46M |   0 | 232 | 157 | 159 |  35 | 22 |  29 |   0 | 7.742000e+03 | 8.267000e+03 |   6.78%| unknown
      0.6s|     1 |     0 |  2342 |     - |    46M |   0 | 232 | 157 | 160 |  36 | 23 |  29 |   0 | 7.742000e+03 | 8.267000e+03 |   6.78%| unknown
     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      0.6s|     1 |     0 |  2355 |     - |    46M |   0 | 232 | 159 | 162 |  38 | 24 |  31 |   0 | 7.742000e+03 | 8.267000e+03 |   6.78%| unknown
      0.6s|     1 |     0 |  2368 |     - |    46M |   0 | 232 | 159 | 163 |  39 | 25 |  31 |   0 | 7.742000e+03 | 8.267000e+03 |   6.78%| unknown
      0.6s|     1 |     0 |  2376 |     - |    46M |   0 | 232 | 160 | 164 |  40 | 26 |  32 |   0 | 7.742000e+03 | 8.267000e+03 |   6.78%| unknown
    L 0.8s|     1 |     0 |  2713 |     - |    rens|   0 | 232 | 165 | 164 |  40 | 27 |  37 |   0 | 7.742000e+03 | 8.135000e+03 |   5.08%| unknown
      0.8s|     1 |     0 |  2713 |     - |    46M |   0 | 232 | 165 | 164 |  40 | 27 |  37 |   0 | 7.742000e+03 | 8.135000e+03 |   5.08%| unknown
      0.8s|     1 |     0 |  2713 |     - |    46M |   0 | 232 | 165 | 164 |  40 | 27 |  37 |   0 | 7.742000e+03 | 8.135000e+03 |   5.08%| unknown
    (run 1, node 1) restarting after 26 global fixings of integer variables

    (restart) converted 26 cuts from the global cut pool into linear constraints

    presolving:
    (round 1, fast)       26 del vars, 9 del conss, 1 add conss, 0 chg bounds, 0 chg sides, 4 chg coeffs, 0 upgd conss, 231 impls, 0 clqs
    (round 2, fast)       26 del vars, 9 del conss, 1 add conss, 0 chg bounds, 119 chg sides, 123 chg coeffs, 0 upgd conss, 231 impls, 0 clqs
    (round 3, exhaustive) 26 del vars, 9 del conss, 1 add conss, 0 chg bounds, 119 chg sides, 123 chg coeffs, 13 upgd conss, 231 impls, 0 clqs
    (round 4, exhaustive) 26 del vars, 9 del conss, 1 add conss, 0 chg bounds, 119 chg sides, 143 chg coeffs, 13 upgd conss, 231 impls, 0 clqs
    presolving (5 rounds: 5 fast, 3 medium, 3 exhaustive):
     26 deleted vars, 9 deleted constraints, 1 added constraints, 0 tightened bounds, 0 added holes, 119 changed sides, 143 changed coefficients
     231 implications, 0 cliques
    presolved problem has 206 variables (205 bin, 0 int, 1 impl, 0 cont) and 182 constraints
         65 constraints of type <knapsack>
         20 constraints of type <linear>
         97 constraints of type <logicor>
    transformed objective value is always integral (scale: 1)
    Presolving Time: 0.02

     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      0.9s|     1 |     0 |  3076 |     - |    42M |   0 | 206 | 182 | 164 |   0 |  0 |  37 |   0 | 7.742000e+03 | 8.135000e+03 |   5.08%| unknown
      0.9s|     1 |     0 |  3101 |     - |    42M |   0 | 206 | 182 | 166 |   2 |  1 |  38 |   0 | 7.742241e+03 | 8.135000e+03 |   5.07%| unknown
      0.9s|     1 |     0 |  3102 |     - |    42M |   0 | 206 | 183 | 165 |   2 |  1 |  39 |   0 | 7.743000e+03 | 8.135000e+03 |   5.06%| unknown
      0.9s|     1 |     0 |  3103 |     - |    42M |   0 | 206 | 184 | 165 |   2 |  1 |  40 |   0 | 7.743000e+03 | 8.135000e+03 |   5.06%| unknown
      0.9s|     1 |     0 |  3184 |     - |    43M |   0 | 206 | 184 | 167 |   4 |  2 |  40 |   0 | 7.744329e+03 | 8.135000e+03 |   5.04%| unknown
      0.9s|     1 |     0 |  3186 |     - |    43M |   0 | 206 | 187 | 167 |   4 |  2 |  43 |   0 | 7.745000e+03 | 8.135000e+03 |   5.04%| unknown
      1.0s|     1 |     0 |  3233 |     - |    43M |   0 | 206 | 187 | 169 |   6 |  3 |  43 |   0 | 7.745000e+03 | 8.135000e+03 |   5.04%| unknown
      1.0s|     1 |     0 |  3252 |     - |    45M |   0 | 206 | 187 | 171 |   8 |  4 |  43 |   0 | 7.745123e+03 | 8.135000e+03 |   5.03%| unknown
      1.0s|     1 |     0 |  3255 |     - |    45M |   0 | 206 | 192 | 171 |   8 |  4 |  48 |   0 | 7.746000e+03 | 8.135000e+03 |   5.02%| unknown
      1.0s|     1 |     0 |  3290 |     - |    45M |   0 | 206 | 192 | 173 |  10 |  5 |  48 |   0 | 7.746000e+03 | 8.135000e+03 |   5.02%| unknown
      1.1s|     1 |     0 |  3434 |     - |    46M |   0 | 206 | 192 | 166 |  12 |  6 |  49 |   0 | 7.746946e+03 | 8.135000e+03 |   5.01%| unknown
      1.1s|     1 |     0 |  3435 |     - |    46M |   0 | 206 | 192 | 166 |  12 |  6 |  49 |   0 | 7.747000e+03 | 8.135000e+03 |   5.01%| unknown
      1.1s|     1 |     0 |  3459 |     - |    46M |   0 | 206 | 192 | 167 |  13 |  7 |  49 |   0 | 7.747318e+03 | 8.135000e+03 |   5.00%| unknown
      1.1s|     1 |     0 |  3460 |     - |    46M |   0 | 206 | 192 | 167 |  13 |  7 |  49 |   0 | 7.748000e+03 | 8.135000e+03 |   4.99%| unknown
      1.2s|     1 |     0 |  3588 |     - |    47M |   0 | 206 | 192 | 168 |  14 |  8 |  49 |   0 | 7.748652e+03 | 8.135000e+03 |   4.99%| unknown
     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      1.2s|     1 |     0 |  3589 |     - |    47M |   0 | 206 | 193 | 168 |  14 |  8 |  50 |   0 | 7.749000e+03 | 8.135000e+03 |   4.98%| unknown
      1.2s|     1 |     0 |  3622 |     - |    55M |   0 | 206 | 193 | 165 |  15 |  9 |  50 |   0 | 7.749000e+03 | 8.135000e+03 |   4.98%| unknown
      1.2s|     1 |     0 |  3736 |     - |    55M |   0 | 206 | 193 | 166 |  16 | 10 |  50 |   0 | 7.750062e+03 | 8.135000e+03 |   4.97%| unknown
      1.2s|     1 |     0 |  3737 |     - |    55M |   0 | 206 | 193 | 166 |  16 | 10 |  50 |   0 | 7.751000e+03 | 8.135000e+03 |   4.95%| unknown
      1.2s|     1 |     0 |  3759 |     - |    55M |   0 | 206 | 193 | 167 |  17 | 11 |  50 |   0 | 7.751000e+03 | 8.135000e+03 |   4.95%| unknown
      1.3s|     1 |     0 |  3823 |     - |    55M |   0 | 206 | 193 | 168 |  18 | 12 |  50 |   0 | 7.751152e+03 | 8.135000e+03 |   4.95%| unknown
      1.3s|     1 |     0 |  3824 |     - |    55M |   0 | 206 | 193 | 168 |  18 | 12 |  50 |   0 | 7.752000e+03 | 8.135000e+03 |   4.94%| unknown
      1.3s|     1 |     0 |  3829 |     - |    55M |   0 | 206 | 193 | 161 |  19 | 13 |  50 |   0 | 7.752000e+03 | 8.135000e+03 |   4.94%| unknown
      1.3s|     1 |     0 |  3836 |     - |    55M |   0 | 206 | 193 | 162 |  20 | 14 |  50 |   0 | 7.752000e+03 | 8.135000e+03 |   4.94%| unknown
      1.3s|     1 |     0 |  3838 |     - |    55M |   0 | 206 | 193 | 163 |  21 | 15 |  50 |   0 | 7.752000e+03 | 8.135000e+03 |   4.94%| unknown
      1.3s|     1 |     0 |  3874 |     - |    55M |   0 | 206 | 195 | 165 |  23 | 16 |  52 |   0 | 7.752000e+03 | 8.135000e+03 |   4.94%| unknown
      1.3s|     1 |     0 |  3878 |     - |    55M |   0 | 206 | 195 | 166 |  24 | 17 |  53 |   0 | 7.752000e+03 | 8.135000e+03 |   4.94%| unknown
      2.0s|     1 |     2 |  4001 |     - |    55M |   0 | 206 | 200 | 166 |  24 | 18 |  59 |  71 | 7.784907e+03 | 8.135000e+03 |   4.50%| unknown
    * 2.9s|    59 |    21 |  6175 |  44.6 |strongbr|  11 | 206 | 251 | 158 |  45 |  1 | 110 | 494 | 7.846000e+03 | 8.099000e+03 |   3.22%|  17.26%
    * 3.0s|    94 |    26 |  6897 |  35.7 |    LP  |  18 | 206 | 262 | 151 |  54 |  2 | 121 | 508 | 7.846000e+03 | 8.090000e+03 |   3.11%|  20.71%
     time | node  | left  |LP iter|LP it/n|mem/heur|mdpt |vars |cons |rows |cuts |sepa|confs|strbr|  dualbound   | primalbound  |  gap   | compl.
      3.0s|   100 |    24 |  7010 |  34.7 |    85M |  18 | 206 | 268 | 146 |  54 |  0 | 127 | 511 | 7.846000e+03 | 8.090000e+03 |   3.11%|  22.99%
      3.3s|   200 |    34 |  9281 |  28.7 |   109M |  18 | 206 | 294 | 156 | 104 |  1 | 153 | 539 | 7.868560e+03 | 8.090000e+03 |   2.81%|  35.07%
      3.5s|   300 |    32 | 10971 |  24.8 |   109M |  18 | 206 | 307 | 146 | 134 |  0 | 166 | 546 | 7.905000e+03 | 8.090000e+03 |   2.34%|  47.01%
      3.8s|   400 |    28 | 12714 |  22.9 |   109M |  18 | 206 | 322 | 146 | 159 |  0 | 181 | 557 | 7.927000e+03 | 8.090000e+03 |   2.06%|  58.37%
      4.0s|   500 |    16 | 14489 |  21.9 |   109M |  18 | 206 | 328 | 148 | 196 |  0 | 187 | 565 | 7.955492e+03 | 8.090000e+03 |   1.69%|  80.54%

    SCIP Status        : problem is solved [optimal solution found]
    Solving Time (sec) : 4.10
    Solving Nodes      : 584 (total of 585 nodes in 2 runs)
    Primal Bound       : +8.09000000000000e+03 (4 solutions)
    Dual Bound         : +8.09000000000000e+03
    Gap                : 0.00 %

Let's now walk through information that the log provides us. We'll break down this information into
smaller bits.

Presolve Information
********************

At the beginning of the run presolve information is output. The most important component is likely the final
few lines of this portion of output. For the log above, from those lines we know that our problem after presolve
has 232 variables and 137 constraints. 231 of those variables are binary with one variable being an implicit integer.
53 constraints are type knapsack, 6 are linear, and 78 are type logicor. The presolving time was 0.01s.

Branch-and-Bound Information
****************************

This section has the bulk of the solve information, and comes directly after the presolve section. It can
easily be identified by it's table like content that makes it easily machine readable. The columns of
the output above are information on the following

.. list-table:: Label Summaries
  :widths: 25 25
  :align: center
  :header-rows: 1

  * - Key
    - Full Description
  * - time
    - total solution time
  * - node
    - number of processed nodes
  * - left
    - number of unprocessed nodes
  * - LP iter
    - number of simplex iterations (see statistics for more accurate numbers)
  * - LP it/n
    - average number of LP iterations since the last output line
  * - mem/heur
    - total number of bytes in block memory or the creator name when a new incumbent solution was found
  * - mdpt
    - maximal depth of all processed nodes
  * - vars
    - number of variables in the problem
  * - cons
    - number of globally valid constraints in the problem
  * - rows
    - number of LP rows in current node
  * - cuts
    - total number of cuts applied to the LPs
  * - sepa
    - number of separation rounds performed at the current node
  * - confs
    - total number of conflicts found in conflict analysis
  * - strbr
    - total number of strong branching calls
  * - dualbound
    - current global dual bound
  * - primalbound
    - current primal bound
  * - gap
    - current (relative) gap using \|primal-dual\| / MIN(\|dual\| , \|primal\|)
  * - compl.
    - completion of search in percent (based on tree size estimation)

.. note:: When a new primal solution is found a letter or asterisk appears on the left side of the current row.
  An asterisk indicates that a primal solution has been found during the tree search, and a letter indicates that
  a primal heuristic has found a solution (letter maps to a specific heuristic)

The table shows the progress of the solver as it optimizes the problem. Each line snapshots a state of the
optimization process, and thereby gives users frequent updates on the quality of the current best solution,
how much memory is being used, and predicted amount of the tree search completed.

It should be mentioned that in the log file above there is a pause in between the branch-and-bound
output and SCIP provides more presolve information. This is due to SCIP identifying that
it is beneficial to start the branch-and-bound tree again but this time applying information
it has learnt to the beginning of the search process. In the example above this is explained by the lines:

.. code-block:: RST

    (run 1, node 1) restarting after 26 global fixings of integer variables

    (restart) converted 26 cuts from the global cut pool into linear constraints

Final Summarised Information
****************************

After the branch-and-bound search is complete, SCIP provides a small amount of summarised information
that is most important for the majority of users. This includes the status (was the problem proven optimal,
or was it infeasible, did we hit a time limit, etc), the total solving time, the amount of nodes explored
in the tree (if restarts were used then nodes of the current tree differ from total nodes of all trees),
the final primal bound (objective value of the best solution), the dual bound (the strongest valid bound
at the end of the solving process),
and finally the gap (the relative difference between the primal and dual bound).

How to Redirect SCIP Output
===========================

If you do not want this information output to your terminal than before calling ``optimize`` one can
call the following function:

.. code-block:: python

    scip.hideOutput()

If you want to redirect your output to Python instead of terminal then one can use the function:

.. code-block:: python

    scip.redirectOutput()

Finally, if you'd like to write the log to a file while optimizing, then one can use the function:

.. code-block:: python

    scip.setLogfile(path_to_file)


SCIP Statistics
===============

While much information is available from the log file or can be easily queried from the Model object,
more specific information is often difficult to find, e.g., how many cuts of a certain type were applied?
For this information one must use the statistics of SCIP. The statistics can be directly printed to terminal
or can be written to a file with the following commands:

.. code-block:: python

  scip.printStatistics()
  scip.writeStatistics(filename=path_to_file)

