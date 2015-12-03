\ SCIP STATISTICS
\   Problem name     : Wine blending
\   Variables        : 3 (0 binary, 0 integer, 0 implicit integer, 3 continuous)
\   Constraints      : 3
\   Obj. scale       : 1
\   Obj. offset      : 0
Maximize
 Obj: +18 x(Medium) +15 x(Dry) +30 x(Sweet)
Subject to
 Use(Alfrocheiro): +1 x(Medium) +2 x(Dry) +1 x(Sweet) <= +60
 Use(Castelao): +1 x(Sweet) <= +30
 Use(Baga): +2 x(Medium) +1 x(Dry) +1 x(Sweet) <= +60
Bounds
End
