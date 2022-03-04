# coding=UTF-8
from ase import io
from ase.io.vasp import read_vasp
import os
import numpy as np


def calc_similarity_cluster(arr1=None, arr2=None, num=None):
    energy1 = arr1[0]
    energy2 = arr2[0]
    arr1 = arr1[1:]
    arr2 = arr2[1:]
    if np.abs(energy1 - energy2) <= 0.1:
        numerator = np.sum(np.abs(arr1 - arr2))
        denominator = 0.5 * np.sum(arr1 + arr2)
        delta = numerator / denominator
        d_max = np.max(np.abs(arr1 - arr2))

        if delta < 0.025 and d_max < 0.7:
            return True
    return False


def elim_similarity(noftraj=10, nofstru=3000, natoms=None, index=None):
    os.chdir('normal_N'+str(natoms)+'_dataset')

    with open('../../../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/geo_mat', 'r') as f:
        chk_data = [line.strip() for line in f.readlines()]
    chk_count = 0
    for chk_items in chk_data:
        if chk_items.startswith('index '+str(index).zfill(3)):
            break
        chk_count += 1
    chk_data = chk_data[:chk_count]
    with open('../../../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/geo_mat', 'w') as f:
        for item in chk_data:
            f.write(item)
            f.write('\n')
    with open('../../../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/geo_mat', 'a') as f:
        f.write('index '+str(index).zfill(3)+'\n')

    if 'simplified_dataset_' + 'N' + str(int(natoms)) in os.listdir('./'):
        # print('true')
        os.system('rm -rf ./simplified_dataset_' + 'N' + str(int(natoms)))
    os.mkdir('./simplified_dataset_' + 'N' + str(int(natoms)))

    files = os.listdir('../normal_N'+str(natoms)+'_dataset')
    n = len(files) - 3
    store = [n]
    count = 0
    for idir in range(1, 2):
        pwd = '../normal_N'+str(natoms)+'_dataset/'

        for id in range(store[idir-1]):
            print('id=', id)
            str_id = str(int(id)).zfill(5)
            filename = pwd + 'POSCAR_' + str_id
            atoms = read_vasp(filename)
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

            with open(pwd + 'id_prop.csv', 'r') as f:
                data = [line.strip() for line in f.readlines()]
            items = data[id]
            tmp_csv = [item for item in items.split(',')]
            energy = float(tmp_csv[1])
            arr.append(energy)

            arr = np.array(arr)
            arr.sort()

            with open('../../../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/geo_mat', 'r') as f:
                data = [line.strip() for line in f.readlines()]
                sim = False
                length = len(data)
                for i in range(length - 1, -1, -1):
                    items = data[i]
                    if items.startswith('index'):
                        continue
                    tmp = [float(item) for item in items.split()]
                    tmp = np.array(tmp)
                    sim = calc_similarity_cluster(tmp, arr, nofatoms)
                    if sim: break

            if not sim:
                str_count = str(int(count)).zfill(4)
                with open('../../../../dataset-ML/N'+str(natoms)+'/N'+str(natoms)+'_dataset/geo_mat', 'a') as f:
                    for item in arr:
                        f.write(str(item) + ' ')
                    f.write('\n')
                os.system('cp ' + filename + ' ./simplified_dataset_' + 'N' + str(int(natoms)) + '/POSCAR_' + str_count)
                with open("./simplified_dataset_N" + str(int(natoms)) + "/id_prop.csv", "a") as f:
                    tmp_csv[0] = str(int(count)).zfill(4)
                    n = len(tmp_csv)
                    for i in range(n - 1):
                        f.write(tmp_csv[i])
                        f.write(',')
                    f.write(tmp_csv[n - 1])
                    f.write('\n')
                count += 1


def remove_abnormal(natoms=None):
    for idir in range(1, 2):
        os.chdir('N'+str(natoms)+'_dataset/')
        if 'normal_N'+str(natoms)+'_dataset' in os.listdir('./'):
            # print('true')
            os.system('rm -rf normal_N'+str(natoms)+'_dataset')
        os.mkdir('./normal_N'+str(natoms)+'_dataset')

        with open('id_prop.csv', 'r') as f:
            data = [line.strip() for line in f.readlines()]

        record_id = []
        id = 0
        while id < len(data):
            print(id)
            items = data[id]
            temp = [float(item) for item in items.split(',')]
            abnormal = False
            if temp[1] >= -3.0:
                abnormal = True
            for i in range(2, len(temp)):
                if temp[i] >= 10.0:
                    abnormal = True
            if abnormal:
                record_id.append(id)
            id = id + 1

        count = 0
        id2 = 0
        with open('normal_N'+str(natoms)+'_dataset/id_prop.csv', 'w') as f:
            while id2 < len(data):
                # print(id2)
                if id2 not in record_id:
                    items2 = data[id2]
                    tmp = [float(item2) for item2 in items2.split(',')]
                    str_count = str(int(count)).zfill(5)
                    tmp[0] = str_count
                    n = len(tmp)
                    for j in range(n-1):
                        f.write(str(tmp[j]))
                        f.write(',')
                    f.write(str(tmp[n-1]))
                    f.write('\n')

                    str_id2 = str(int(id2)).zfill(6)
                    os.system('cp POSCAR_'+str_id2+' normal_N'+str(natoms)+'_dataset/POSCAR_'+str_count)

                    count += 1
                id2 += 1

