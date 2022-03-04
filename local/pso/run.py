#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from local.pso.pso_sp import *
from local.pso.pso_atoms import PsoAtoms
from local.pso.nnmodel import Model
import numpy as np
from local.pso.pos_generator import read_poscar
import sys


def run(iter_index=0, natoms=None):
    if iter_index == 0:
        os.system('rm -rf pre_relax_0* pre_relax_1* velocities_0* velocities_1* pso_data')
        os.system('rm -rf pso_0* pso_1*')

        npar = 0
        NPAR = 150
        with open('pso_init', 'r') as f:
            data = [line for line in f.readlines()]
        data[4] = 'GER 1' + '\n'
        data[1] = 'NPAR ' + str(NPAR) + '\n'
        with open('pso_init', 'w') as f:
            f.writelines(data)

        atoms_list = []

        while npar < NPAR:
            num_str = str(npar).zfill(4)
            title, lattice_constant, a, elements, numofatoms, selective_dynamics, \
            selective_flags, direct_ac, atom_pos = \
                read_poscar("../../dataset-ML/N"+str(natoms)+"/N"+str(natoms)+"_dataset/initial_pso/POSCAR_" + num_str)
            title = title.strip()
            b_pos = atom_pos

            atoms = PsoAtoms(subs_symbols=['Au'], abso_symbols=['Au'] * (numofatoms[0] - 1),
                             name=title, cell=a, lattice_constant=lattice_constant, constraints=[0, 0, 0],
                             pbc=[0, 0, 0])
            if direct_ac:
                atoms.set_abso_positions(b_pos[0:-1], cord_mod='d')
                atoms.set_subs_positions(b_pos[-1], cord_mod='d')
            atoms_list.append(atoms)
            npar += 1

        nnmodel = Model(lattice_parameter=lattice_constant)
        pso = Pso(atoms_list=atoms_list, calculator=nnmodel)
        pso.set_calc(nnmodel)
        pso_evo(pso, natoms=natoms)

        updated = False
        with open('pso_init', 'r') as f:
            data = [line for line in f.readlines()]
        data[4] = 'GER 1' + '\n'
        data[-1] = 'UPDATED ' + str(updated) + '\n'
        with open('pso_init', 'w') as f:
            f.writelines(data)
        pso_evo(natoms=natoms)
    else:
        generation = iter_index
        updated = True
        with open('pso_init', 'r') as f:
            data = [line for line in f.readlines()]
        data[4] = 'GER ' + str(generation) + '\n'
        data[-1] = 'UPDATED ' + str(updated) + '\n'
        with open('pso_init', 'w') as f:
            f.writelines(data)
        if pso_evo(natoms=natoms) is not None:
            sys.exit(0)

        generation = iter_index + 1
        updated = False
        with open('pso_init', 'r') as f:
            data = [line for line in f.readlines()]
        data[4] = 'GER ' + str(generation) + '\n'
        data[-1] = 'UPDATED ' + str(updated) + '\n'
        with open('pso_init', 'w') as f:
            f.writelines(data)
        pso_evo(natoms=natoms)
