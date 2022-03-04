#!/usr/bin/env python 

import numpy as np
from local.pso.pos_generator import H_generator,read_poscar


def RandStru(atoms, natoms=None):
    # title, lattice_constant, a, elements, numofatoms, selective_dynamics, selective_flags, direct_ac, atom_pos = read_poscar('POSCAR.H2xSqrt3_90')
    title, lattice_constant, a, elements, numofatoms, \
    selective_dynamics, selective_flags, direct_ac, atom_pos = \
        read_poscar('../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/initial_pso/POSCAR_0000')
    dr = 10.0
    dc = 10.0
    b_pos = H_generator(lattice_constant, a, numofatoms, atom_pos, selective_flags, dr, dc)
    # print(b_pos)
    atoms.set_cell(a)
    if direct_ac:
        atoms.set_abso_positions(b_pos[0:-1],cord_mod='d')
        atoms.set_subs_positions(b_pos[-1],cord_mod='d')
    else:
        atoms.set_abso_positions(b_pos[0:-1],cord_mod='c')
        atoms.set_subs_positions(b_pos[-1],cord_mod='c')
    atoms.set_lattice_constant(lattice_constant)




