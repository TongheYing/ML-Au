# coding=UTF-8
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase import io
import os
import numpy as np
import math


def calc_similarity(arr1=None, arr2=None, threshold=0.3, n=None):
    diff = np.linalg.norm(arr1-arr2) / np.sqrt(n*(n-1)/2.0)
    if diff < threshold:
        return True
    return False


def chk_similarity(filename=None, gen=0, thres=0.3):
    atoms = io.vasp.read_vasp(filename)
    nofatoms = atoms.get_global_number_of_atoms()
    dist = [atoms.get_distance(atom1, atom2) for atom1 in [i for i in range(nofatoms)] for atom2 in
            [j for j in range(nofatoms)]]
    dist = np.array(dist)
    dist = dist.reshape(nofatoms, nofatoms)
    dist = np.triu(dist)
    arr = []
    for i in range(nofatoms):
        for j in range(i, nofatoms):
            if dist[i][j] != 0.0:
                arr.append(dist[i][j])
    arr = np.array(arr)
    arr.sort()

    with open('../../geo_mat_traj', 'r') as f:
        data = [line.strip() for line in f.readlines()]
        sim = False
        length = len(data)
        for i in range(length-1, -1, -1):
            items = data[i]
            if items.startswith('generation'):
                continue
            tmp = [float(item) for item in items.split()]
            tmp = np.array(tmp)
            sim = calc_similarity(tmp, arr, threshold=thres, n=nofatoms)
            if sim: break

    if not sim:
        with open('../../geo_mat_traj', 'a') as f:
            for item in arr:
                f.write(str(item) + ' ')
            f.write('\n')
    return sim


def read_trajectory(gen=1, prev_update=0, prev_init=0, prev_test=0, natoms=None):
    str_gen = str(int(gen)).zfill(3)
    traj = Trajectory('../../../clustertut/ase_calcs/optimization.traj')
    clustertut = '../../../clustertut'
    ase_dir = os.path.join(clustertut, 'ase_calcs')
    if 'test_store_'+str_gen not in os.listdir('../'):
        os.mkdir('../test_store_'+str_gen)
    if 'updating_'+str_gen not in os.listdir('../'):
        os.mkdir('../updating_'+str_gen)

    update_id = 0
    init_id = 0
    test_id = 0
    count = 0

    for atoms in traj:
        nofatoms = atoms.get_global_number_of_atoms()
        str_test_id = str(prev_test+test_id).zfill(4)
        molecule_path = os.path.join(ase_dir, 'POSCAR_err')
        write_vasp(molecule_path, atoms, direct=True, vasp5=True)
        os.system('cp '+molecule_path+' ../test_store_'+str_gen+'/POSCAR_'+str_test_id)

        test_id = test_id + 1

        stddev = 1.0

        threshold = 0.42 / (gen ** 1) + 0.08


        if stddev > threshold:
            count += 1
            sim = True
            if not sim:
                str_update_id = str(prev_update + update_id).zfill(4)
                os.system('cp ../test_store_'+str_gen+'/POSCAR_'+str_test_id+' ../updating_'+str_gen+'/'+'POSCAR_'+str_update_id)
                update_id = update_id + 1

    return update_id+prev_update, init_id+prev_init, test_id+prev_test
