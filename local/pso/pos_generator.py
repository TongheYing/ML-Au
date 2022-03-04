#!/usr/bin/env python

import numpy as np
from decimal import Decimal
from random import uniform
from math import pi, sin, cos


def main():
    title, lattice_constant, a, elements, numofatoms, selective_dynamics, selective_flags, direct_ac, atom_pos = read_poscar('POSCAR.H2x2')

    dr = 0.3
    dc = 0.5
    b_pos = H_generator(lattice_constant, a, numofatoms, atom_pos, selective_flags, dr, dc)
    write_poscar('POSCAR', title, lattice_constant, a, elements, numofatoms, selective_dynamics, selective_flags, direct_ac, b_pos)
    return


def H_generator(lattice_constant, a, numofatoms, atom_pos, selective_flags, dr, dc):
    natom = sum(numofatoms)
    sc = np.linalg.norm(a[2,:]) * lattice_constant
    b = np.linalg.inv(a * lattice_constant).T
    for i in range(natom):
        if selective_flags[i,0] == 'T':
            atom_pos[i, :] = np.random.uniform(0.25, 0.75, (1, 3))

    return atom_pos

def stacking(a, numofatoms, atom_pos, selective_flags, nstack):
    b = np.empty([3,3])
    for i in range(3):
        b[i] =  a[i] * nstack[i]
    n0_atoms = sum(numofatoms)
    nn_atoms = n0_atoms*nstack[0]*nstack[1]*nstack[2] 
    b_pos = np.empty([nn_atoms,3])    
    b_selective_flags = np.empty([nn_atoms,3],dtype='str')
    me = 0
    mn = 0
    for ie in numofatoms:
        for nc in range(nstack[2]):
            for nb in range(nstack[1]):
                for na in range(nstack[0]):
                    for k in range(ie):
                        m0 = me + k
                        b_pos[mn,0] = (atom_pos[m0,0] + na) / nstack[0]
                        b_pos[mn,1] = (atom_pos[m0,1] + nb) / nstack[1]
                        b_pos[mn,2] = (atom_pos[m0,2] + nc) / nstack[2]
                        b_selective_flags[mn] = selective_flags[m0]
                        mn = mn + 1
        me = me + ie
    b_natoms = numofatoms * nstack[0] * nstack[1] * nstack[2]    
    return b, b_natoms, b_pos, b_selective_flags

def rescale(a, atom_pos, nscale):
    b = np.empty([3,3])
    for i in range(3):
        b[i] =  a[i] * nscale[i]
    b_pos = atom_pos / nscale
    return b, b_pos
    
def shift(atom_pos, rshift):
    b_pos = (atom_pos + rshift) % 1
    return b_pos

def sort_column(numofatoms, atom_pos, selective_flags, axis, order):
    tot_atoms = sum(numofatoms)
    atom_type = np.empty(tot_atoms,dtype='int')
    k = 0
    for i, num in enumerate(numofatoms):
        atom_type[k:k+num] = i
        k = k+num
    for i in range(tot_atoms-1):
        for j in range(i+1,tot_atoms):
            if atom_type[i] == atom_type[j]:
                if (order == 'd' and atom_pos[i,axis] < atom_pos[j,axis]) or (order == 'a' and atom_pos[i,axis] > atom_pos[j,axis]):
                    for k in range(3):
                        temp = atom_pos[i,k]
                        atom_pos[i,k] = atom_pos[j,k]
                        atom_pos[j,k] = temp
                        temp = selective_flags[i,k]
                        selective_flags[i,k] = selective_flags[j,k]
                        selective_flags[j,k] = temp
    return atom_pos, selective_flags


def read_poscar(filename='POSCAR'):
    """Import POSCAR/CONTCAR type file.

    Reads unitcell, atom positions and constraints from the POSCAR/CONTCAR
    file and tries to read atom types from POSCAR/CONTCAR header.
    """
    f = open(filename)

    # read title
    title = f.readline()

    # read lattice constant
    lattice_constant = float(f.readline().split()[0])

    # read the lattice vectors
    a = np.empty([3,3])
    for ii in range(3):
        s = f.readline().split()
        a[ii] = Decimal(s[0]), Decimal(s[1]), Decimal(s[2])
        
    basis_vectors = a * lattice_constant
    
    # read atom types
    elements = f.readline().split()

    # Check whether we have a VASP 4.x or 5.x format file. If the
    # format is 5.x, use the fifth line to provide information about
    # the atomic symbols.
    vasp5 = False
    try:
        int(elements[0])
        print(filename,' is not in VASP 5.x format!')
        return
    except ValueError:
        vasp5 = True

    # read number of atoms
    numofatoms = f.readline().split()
    for i, num in enumerate(numofatoms):
        numofatoms[i] = int(num)
    numofatoms = np.array(numofatoms)

    # Check if Selective dynamics is switched on
    sdyn = f.readline()
    selective_dynamics = sdyn[0].lower() == "s"

    # Check if atom coordinates are cartesian or direct
    if selective_dynamics:
        ac_type = f.readline()
    else:
        ac_type = sdyn
    direct_ac = ac_type[0].lower() == "d"
    tot_atoms = sum(numofatoms)
    
    # read coordinates
    atom_pos = np.empty([tot_atoms,3])
    if selective_dynamics:
        selective_flags = np.empty([tot_atoms,3],dtype='str')
    for atom in range(tot_atoms):
        ac = f.readline().split()
        atom_pos[atom] = Decimal(ac[0]), Decimal(ac[1]), Decimal(ac[2])
        if selective_dynamics:
            selective_flags[atom] = ac[3:6]
    f.close()
    return title, lattice_constant, a, elements, numofatoms, selective_dynamics, selective_flags, direct_ac, atom_pos

def write_poscar(filename, title, lattice_constant, a, elements, numofatoms, selective_dynamics, selective_flags, direct_ac, atom_pos):
    f = open(filename,'w')
    f.write(title)
    f.write('{0:9.5f} \n'.format(lattice_constant))
    for i in range(3):
        f.write('{0:23.16f} {1:19.16f} {2:19.16f} \n'.format(a[i,0],a[i,1],a[i,2]))
    f.write(' '.join(x.rjust(4) for x in elements))
    f.write('\n')
    f.write(' '.join(str(m).rjust(4) for m in numofatoms))
    f.write('\n')
    if selective_dynamics: f.write('Selective dynamics\n')
    if direct_ac: 
        f.write('Direct\n')
    else:
        f.write('Cartesian\n')
    

    tot_atoms = sum(numofatoms)
    for i in range(tot_atoms):
        if selective_dynamics:
            f.write('{0:20.16f} {1:19.16f} {2:19.16f} {3} {4} {5}\n'.format(atom_pos[i,0],atom_pos[i,1],atom_pos[i,2],selective_flags[i,0].rjust(3),selective_flags[i,0].rjust(3),selective_flags[i,0].rjust(3)))
        else:
            f.write('{0:20.16f} {1:19.16f} {2:19.16f}\n'.format(atom_pos[i,0],atom_pos[i,1],atom_pos[i,2]))
    return
