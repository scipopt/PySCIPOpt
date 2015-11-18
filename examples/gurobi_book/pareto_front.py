"""
pareto_front.py:  tools for building a pareto front in multi-objective optimization

Copyright (c) by Joao Pedro PEDROSO and Mikio KUBO, 2012
"""

def dominates(a,b):
    dominating = False
    for i in range(len(a)):
        if a[i] > b[i]:
            return False
        if a[i] < b[i]:
            dominating = True
    return dominating


def is_dominated(a,front):
    for b in front:
        if dominates(b,a):
            return True
    return False

def pareto_front(cand):
    front = set([])
    for i in cand:
        add_i = True
        for j in list(front):
            if dominates(i,j):
                front.remove(j)
            if dominates(j,i):
                add_i = False
        if add_i:
            front.add(i)
    front = list(front)
    front.sort()
    return front


if __name__ == "__main__":
    import random
    # random.seed(1)
    cand = [(random.random()**.25,random.random()**.25) for i in range(100)]
    import matplotlib.pyplot as plt
    for (x,y) in cand:
        plt.plot(x,y,"bo")

    front = pareto_front(cand)
    plt.plot([x for (x,y) in front], [y for (x,y) in front])
    plt.show()
