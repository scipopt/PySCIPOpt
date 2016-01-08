def multidict(D):
    keys = list(D.keys())
    if len(keys) == 0:
        return [[]]
    try:
        N = len(D[keys[0]])
        islist = True
    except:
        N = 1
        islist = False
    dlist = [dict() for d in range(N)]
    for k in keys:
        if islist:
            for i in range(N):
                dlist[i][k] = D[k][i]
        else:
            dlist[0][k] = D[k]
    return [keys]+dlist
