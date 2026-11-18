"""Exact BFS for tiny puzzles (small value_limit): optimal program over sets of produced values."""
from collections import deque


def solve_tiny(sv, targets, limit, nreg=48, maxbits=20):
    if limit > maxbits:
        return None
    start = 0
    for v in sv:
        if 1 <= v <= limit:
            start |= 1 << v
    goal = 0
    for t in targets:
        goal |= 1 << t
    if start & goal == goal:
        return []
    prev = {start: None}
    q = deque([start])
    while q:
        S = q.popleft()
        vals = [v for v in range(1, limit + 1) if S >> v & 1]
        for i, a in enumerate(vals):
            for b in vals:
                for op in '+-*/':
                    if op == '+':
                        r = a + b
                    elif op == '-':
                        if a <= b:
                            continue
                        r = a - b
                    elif op == '*':
                        r = a * b
                    else:
                        if a % b:
                            continue
                        r = a // b
                    if r < 1 or r > limit or S >> r & 1:
                        continue
                    T = S | (1 << r)
                    if T in prev:
                        continue
                    prev[T] = (S, op, a, b, r)
                    if T & goal == goal:
                        steps = []
                        X = T
                        while prev[X] is not None:
                            P, op_, a_, b_, r_ = prev[X]
                            steps.append((op_, a_, b_, r_))
                            X = P
                        return steps[::-1]
                    q.append(T)
    return None
