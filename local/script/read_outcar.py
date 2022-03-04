#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import sys
import os

from local.pso.pos_generator import read_poscar, write_poscar


def read_100random(num=None, natoms=None):
    id = 0
    for i in range(num):
        print(i)
        str_i = str(int(i)).zfill(4)
        os.chdir(str_i)
        id = read_outcar(num=id, natoms=natoms)
        os.chdir('../')
        

def read_outcar(filename1="OUTCAR", filename2="POSCAR", num=None, natoms=None):
    title, lattice_constant, a, elements, numofatoms, selective_dynamics, \
    selective_flags, direct_ac, atom_pos = read_poscar(filename=filename2)
    atom_force = np.zeros_like(atom_pos)

    with open(filename1, "r") as f:
        data = [line.strip() for line in f.readlines()]

    index = num
    for i, line in enumerate(data):
        if "TOTAL-FORCE" in line:
            if index < 1000000:
                index_str = str(index).zfill(6)
                energy = float(data[i+12+int(numofatoms)].split()[4])
                for bias in range(int(numofatoms)):
                    content = data[i+2+bias].split()
                    for idx in range(len(content)):
                        if idx < 3:
                            atom_pos[bias][idx] = float(content[idx]) / (lattice_constant*a[0][0])
                        else:
                            atom_force[bias][idx-3] = float(content[idx])
                index += 1
                new_filename = "../N"+str(natoms)+"_dataset/POSCAR_" + index_str
                write_poscar(new_filename, title, lattice_constant, a, elements, numofatoms, selective_dynamics,
                             selective_flags, direct_ac, atom_pos)
                write_id_prop("../N"+str(natoms)+"_dataset/id_prop.csv", index_str, atom_force, energy)
            else:
                pass
    return index


def write_id_prop(filename="id_prop.csv", index=None, force=None, energy=None):
    row, col = force.shape[0], force.shape[1]
    force_str = ""
    with open(filename, "a") as f:
        for i in range(row):
            for j in range(col):
                force_str = force_str + "," + str(force[i][j])
        f.write(index+","+str(energy)+force_str+'\n')

