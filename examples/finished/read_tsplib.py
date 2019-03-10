##@file read_tsplib.py
#@brief read standard instances of the  traveling salesman problem
"""
Functions provided:
    * read_tsplib  - read a symmetric tsp instance
    * read_atsplib -        asymmetric

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

import gzip
import math

def distL2(x1,y1,x2,y2):
    """Compute the L2-norm (Euclidean) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.sqrt(xdiff*xdiff + ydiff*ydiff) + .5)


def distL1(x1,y1,x2,y2):
    """Compute the L1-norm (Manhattan) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    return int(abs(x2-x1) + abs(y2-y1)+.5)


def distLinf(x1,y1,x2,y2):
    """Compute the Linfty distance between two points (see TSPLIB documentation)"""
    return int(max(abs(x2-x1),abs(y2-y1)))

def distATT(x1,y1,x2,y2):
    """Compute the ATT distance between two points (see TSPLIB documentation)"""
    xd = x2 - x1
    yd = y2 - y1
    rij = math.sqrt((xd*xd + yd*yd) /10.)
    tij = int(rij + .5)
    if tij < rij:
        return tij + 1
    else:
        return tij

def distCEIL2D(x1,y1,x2,y2):
    """returns smallest integer not less than the distance of two points"""
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.ceil(math.sqrt(xdiff*xdiff + ydiff*ydiff)))

def distGEO(x1,y1,x2,y2):
    print("Implementation is wrong")
    assert False
    PI = 3.141592
    deg = int(x1 + .5)
    min_ = x1 - deg
    lat1 = PI * (deg + 5.*min_/3)/180.
    deg = int(y1 + .5)
    min_ = y1 - deg
    long1 = PI * (deg + 5.*min_/3)/180.
    deg = int(x2 + .5)
    min_ = x2 - deg
    lat2 = PI * (deg + 5.*min_/3)/180.
    deg = int(y2 + .5)
    min_ = y2 - deg
    long2 = PI * (deg + 5.*min_/3)/180.


    RRR = 6378.388
    q1 = math.cos( long1 - long2 );
    q2 = math.cos( lat1 - lat2 );
    q3 = math.cos( lat1 + lat2 );
    return int(RRR * math.acos(.5*((1.+q1)*q2 - (1.-q1)*q3)) + 1.)

def read_explicit_lowerdiag(f,n):
    c = {}
    i,j = 1,1
    while True:
        line = f.readline()
        for data in line.split():
            c[j,i] = int(data)
            j += 1
            if j>i:
                i += 1
                j = 1
            if i > n:
                return range(1,n+1),c,None,None

def read_explicit_upper(f,n):
    c = {}
    i,j = 1,2
    while True:
        line = f.readline()
        for data in line.split():
            c[i,j] = int(data)
            j += 1
            if j>n:
                i += 1
                j = i+1
            if i == n:
                return range(1,n+1),c,None,None

def read_explicit_upperdiag(f,n):
    c = {}
    i,j = 1,1
    while True:
        line = f.readline()
        for data in line.split():
            c[i,j] = int(data)
            j += 1
            if j>n:
                i += 1
                j = i
            if i == n:
                return range(1,n+1),c,None,None

def read_explicit_matrix(f,n):
    c = {}
    i,j = 1,1
    while True:
        line = f.readline()
        for data in line.split():
            if j>i:
                c[i,j] = int(data)
            j += 1
            if j>n:
                i += 1
                j = 1
            if i == n:
                return range(1,n+1),c,None,None

def read_tsplib(filename):
    "basic function for reading a symmetric problem in the TSPLIB format"
    "data is stored in an upper triangular matrix"
    "NOTE: some distance types are not handled yet"
    if filename[-3:] == ".gz":
        f = gzip.open(filename, "rt")
    else:
        f = open(filename)

    line = f.readline()
    while line.find("DIMENSION") == -1:
        line = f.readline()
    n = int(line.split()[-1])

    while line.find("EDGE_WEIGHT_TYPE") == -1:
        line = f.readline()

    if line.find("EUC_2D") != -1:
        dist = distL2
    elif line.find("MAN_2D") != -1:
        dist = distL1
    elif line.find("MAX_2D") != -1:
        dist = distLinf
    elif line.find("ATT") != -1:
        dist = distATT
    elif line.find("CEIL_2D") != -1:
        dist = distCEIL2D
    # elif line.find("GEO") != -1:
    #     print("geographic"
    #     dist = distGEO
    elif line.find("EXPLICIT") != -1:
        while line.find("EDGE_WEIGHT_FORMAT") == -1:
            line = f.readline()
        if line.find("LOWER_DIAG_ROW") != -1:
            while line.find("EDGE_WEIGHT_SECTION") == -1:
                line = f.readline()
            return read_explicit_lowerdiag(f,n)
        if line.find("UPPER_ROW") != -1:
            while line.find("EDGE_WEIGHT_SECTION") == -1:
                line = f.readline()
            return read_explicit_upper(f,n)
        if line.find("UPPER_DIAG_ROW") != -1:
            while line.find("EDGE_WEIGHT_SECTION") == -1:
                line = f.readline()
            return read_explicit_upperdiag(f,n)
        if line.find("FULL_MATRIX") != -1:
            while line.find("EDGE_WEIGHT_SECTION") == -1:
                line = f.readline()
            return read_explicit_matrix(f,n)
        print("error reading line " + line)
        raise(Exception)
    else:
        print("cannot deal with '%s' distances" % line)
        raise Exception

    while line.find("NODE_COORD_SECTION") == -1:
        line = f.readline()

    x,y = {},{}
    while 1:
        line = f.readline()
        if line.find("EOF") != -1 or not line: break
        (i,xi,yi) = line.split()
        x[i] = float(xi)
        y[i] = float(yi)

    V = x.keys()
    c = {}      # dictionary to hold n times n matrix
    for i in V:
        for j in V:
            c[i,j] = dist(x[i],y[i],x[j],y[j])

    return V,c,x,y



def read_atsplib(filename):
    "basic function for reading a ATSP problem on the TSPLIB format"
    "NOTE: only works for explicit matrices"

    if filename[-3:] == ".gz":
        f = gzip.open(filename, 'r')
        data = f.readlines()
    else:
        f = open(filename, 'r')
        data = f.readlines()

    for line in data:
        if line.find("DIMENSION") >= 0:
            n = int(line.split()[1])
            break
    else:
        raise IOError("'DIMENSION' keyword not found in file '%s'" % filename)

    for line in data:
        if line.find("EDGE_WEIGHT_TYPE") >= 0:
            if line.split()[1] == "EXPLICIT":
                break
    else:
        raise IOError("'EDGE_WEIGHT_TYPE' is not 'EXPLICIT' in file '%s'" % filename)

    for k,line in enumerate(data):
        if line.find("EDGE_WEIGHT_SECTION") >= 0:
            break
    else:
        raise IOError("'EDGE_WEIGHT_SECTION' not found in file '%s'" % filename)

    c = {}
    # flatten list of distances
    dist = []
    for line in data[k+1:]:
        if line.find("EOF") >= 0:
            break
        for val in line.split():
            dist.append(int(val))

    k = 0
    for i in range(n):
        for j in range(n):
            c[i+1,j+1] = dist[k]
            k += 1

    return n,c



if __name__ == "__main__":
    import sys
    # Parse argument
    if len(sys.argv) < 2:
        print('Usage: %s instance' % sys.argv[0])
        exit(1)

    from read_tsplib import read_tsplib
    V,c,x,y = read_tsplib(sys.argv[1])
    print(len(V), "vertices,", len(c), "arcs")
    print("distance matrix:")
    for i in V:
        for j in V:
            if j > i:
                print(c[i,j],)
            elif j < i:
                print(c[j,i],)
            else:
                print(0,)
        print
    print
