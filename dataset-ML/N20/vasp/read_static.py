#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""read energy and forces from 50 static calculation structure"""
import os
import sys
import numpy as np

def write_id_prop(filename="id_prop.csv", index=None, force=None, energy=None):
    row, col = force.shape[0], force.shape[1]
    force_str = ""
    # print("row, col=", row, col)
    with open(filename, "a") as f:
        for i in range(row):
            for j in range(col):
                force_str = force_str + "," + str(force[i][j])
        f.write(index+","+str(energy)+force_str+'\n')


def read_static():
    # title, lattice_constant, a, elements, numofatoms, selective_dynamics, \
    # selective_flags, direct_ac, atom_pos = read_poscar('N'+str(n)+'-static/' + 'N'+str(n)+'-static/0000/POSCAR')
    with open('0000/POSCAR', 'r') as f:
        data = [line.strip() for line in f.readlines()]
        numofatoms = int(data[6])
    # atom_force = np.zeros_like(atom_pos)
    atom_force = np.zeros((numofatoms, 3))
    for i in range(150):
        str_i = str(i).zfill(4)
        # filename = 'N'+str(n)+'-static/' + 'N'+str(n)+'-static/' + str_i + '/OUTCAR'
        filename = str_i + '/OUTCAR'
        with open (filename, 'r') as f:
            data = [line.strip() for line in f.readlines()]

        for j, items in enumerate(data):
            if 'TOTAL-FORCE' in items:
                energy = float(data[j + 14 + int(numofatoms)].split()[6])
                print("energy=", float(data[j + 12 + int(numofatoms)].split()[4]))
                for bias in range(int(numofatoms)):
                    content = data[j+2+bias].split()
                    for idx in range(len(content)):
                        if idx >= 3:
                            atom_force[bias][idx - 3] = float(content[idx])
                            # print("force=", atom_force[bias][idx - 3])
                        else:
                            pass
                # write_id_prop('N'+str(n)+'-static/' + 'N'+str(n)+'-static/' + 'id_prop.csv', str_i, atom_force, energy)
                write_id_prop('id_prop.csv', str_i, atom_force, energy)
            else:
                pass
