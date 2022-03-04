#!/usr/bin/env python

import os
import shutil
import numpy as np
from local.pso.pso_atoms import PsoAtoms


class Model(object):
    def __init__(self, lattice_parameter=None, cord_mod=None):
        self.set_cord_mod(cord_mod)
        self._name = "nnmodel"
        self._lattice_parameter = lattice_parameter

    def get_energy(self, updated=False, num=None, gen=None):
        if updated:
            energy = read_update(num=num, gen=gen)
        else:
            energy = 0.0

        return energy

    def get_stru(self, atoms, filename='POSCAR_pbest'):
        with open(filename, 'r') as f:
            data = [line.strip() for line in f.readlines()]
            data = data[9:]
        subs_stru = []
        abso_stru = []
        nsubs = len(atoms.get_subs_symbols())
        nabso = len(atoms.get_abso_symbols())

        for iabso in range(nabso):
            atom = [float(j) for j in data[iabso].split()[:3]]
            atom = np.array(atom)
            abso_stru.append(atom)
        abso_stru = np.array(abso_stru)
        for isubs in range(nsubs):
            atom = [float(j) for j in data[nabso + isubs].split()[:3]]
            atom = np.array(atom)
            subs_stru.append(atom)
        subs_stru = np.array(subs_stru)

        atoms.set_subs_positions(subs_stru, cord_mod='d')
        return abso_stru

    def set_cord_mod(self, cord_mod=None):
        if cord_mod==None:
            cord_mod='direct'#cordinate is based on the cell vetors
        elif cord_mod=='c' or cord_mod=='C':
            cord_mod='cart'
        else:
            cord_mod='direct'
        self.cord_mod=cord_mod

    def get_cord_mod(self):
        return self.cord_mod

    def read_atoms(self, atoms=None, filename='POSCAR'):
        "read atoms' information and write them in POSCAR file for vasp runuing"

        if not isinstance(atoms, PsoAtoms):
            raise ValueError('given atoms is not a PsoAtoms object')
        f = open(filename, 'w')
        f.write(atoms.get_system_name())
        f.write('\n')
        lc = atoms.get_lattice_constant()
        f.write('%.16f\n' % lc)
        for a in atoms.get_cell():
            write_ndarray(f, a)
        for i in atoms._abso_elements:
            f.write('%s  ' % i)
        f.write('\n')
        numofatom = atoms._abso_numbers[0] + atoms._subs_numbers[0]
        f.write('%d  ' % numofatom)
        f.write('\n')
        f.write('Selective dynamics\n')
        if self.get_cord_mod() == 'cart':
            f.write('Cartesian\n')
        else:
            f.write('direct\n')

        for atom in atoms.get_abso_positions():
            if self.get_cord_mod() == 'direct':
                atom = atoms.cart_direct(atom)
            for j in atom:
                f.write('%.16f  ' % j)
            f.write('T T T\n')
        for i, atom in enumerate(atoms.get_subs_positions()):
            if self.get_cord_mod() == 'direct':
                atom = atoms.cart_direct(atom)
            for j in atom:
                f.write('%.16f  ' % j)
            for k in atoms.get_constraints()[i]:
                if k == 1:
                    f.write('T ')
                elif isinstance(k, str):
                    if 'T' in k or 't' in k:
                        f.write('T ')
                    else:
                        f.write('F ')
                else:
                    f.write('F ')
            f.write('\n')

        f.close()

    def sp_run(self, input_dir=None, atoms=None):
        if isinstance(atoms, PsoAtoms):
            self.read_atoms(atoms)
        else:
            raise ValueError('The given atoms is not an PsoAtoms object')


def write_ndarray(f,a):
    "write a numpy ndarray to a line of a file"
    for n in a:
        f.write('%.16f  '%n)
    f.write('\n')


def read_update(num=None, gen=None):
    str_gen = str(int(gen)).zfill(3)
    with open('../update_'+str_gen+'/id_prop.csv', 'r') as f:
        data = [line.strip() for line in f.readlines()]
    energy = float(data[num].split(',')[1])
    return energy