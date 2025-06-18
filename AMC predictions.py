#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 19:02:44 2025

@author: mariabucheru
"""

import numpy as np
from pymol import cmd
objects = cmd.get_object_list()

print(objects)

for i in objects: print(i)
for idx, obj1 in enumerate(objects): print(idx, obj1)
for idx, obj1 in enumerate(objects):
    for obj2 in objects[idx+1:]: print(idx, obj1, obj2)

print(obj1, obj2)
print(cmd.align(objects[0], objects[1], cycles=0))
import time
print(-time.time() + (cmd.align(objects[0], objects[1], cycles=0) and time.time()))

alignments = []
for idx, obj1 in enumerate(objects):
    for obj2 in objects[idx+1:]:
        print(obj1, obj2, -time.time() + (alignments.append(cmd.align(obj1, obj2)) or time.time()))
print(alignments)




A= np.zeros((len (objects), len(objects)))
A[np.triu_indices(len(objects),1) ]= [aln[3] for aln in alignments]
A += A.T
print (A)

import matplotlib.pyplot as plt
plt.imshow(A)
plt.show(A)

m = A.mean(axis=0)
vals, vecs = np.linalg.eigh(-0.5 * (A - m[:,None] - m[None,:] + m.mean()))
print(vals)
pcoords = vecs[:, ::-1] * vals[::-1] **0.5

plt.scatter(pcoords[:, 0], pcoords[:, 1])
plt.gca().set_aspect('equal')
plt.show()

plt.scatter(pcoords[:, 0], pcoords[:, 1], c=pcoords[:, 2])
plt.show()