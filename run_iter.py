#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os
from remote.pso_step import *
from remote.correction_step import *
from remote.train_step import *


def record_iter(record, ii, jj):
    with open(record, 'a') as frec:
        frec.write("%d %d\n" % (ii, jj))


def main():
    # the atom size to be optimized
    natoms = 20 
    method = 'train'

    max_tasks = 100
    numb_task = 9
    record = 'auto_init'

    # it is used to record which stage the program is in
    iter_rec = [0, -1] 

    if os.path.isfile(record):
        with open(record) as frec:
            for line in frec:
                iter_rec = [int(x) for x in line.split()]

    cont = True
    ii = -1
    while cont:
        ii += 1
        tasks_name = 'tasks %03d' % ii
        print(tasks_name)
        for jj in range(numb_task):
            # restart from the checkpoint
            if ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1]:
                continue


            if jj == 0:
                print('make_pso %03d %02d' % (ii, jj))
                make_pso(ii, natoms=natoms)
            elif jj == 1:
                print('run_pso %03d %02d' % (ii, jj))
                run_pso(ii, natoms=natoms)
            elif jj == 2:
                print('post_pso %03d %02d' % (ii, jj))
                post_pso(ii, natoms=natoms)
            elif jj == 3:
                print('make_correction %03d %02d' % (ii, jj))
                make_correction(ii, natoms=natoms)
            elif jj == 4:
                print('run_correction %03d %02d' % (ii, jj))
                run_correction(ii, natoms=natoms, method=method)
            elif jj == 5:
                print('post_correction %03d %02d' % (ii, jj))
                post_correction(ii, natoms=natoms)
            elif jj == 6:
                print('make_train %03d %02d' % (ii, jj))
                make_train(ii, natoms=natoms)
            elif jj == 7:
                print('run_train %03d %02d' % (ii, jj))
                run_train(ii, natoms=natoms, method=method)
            elif jj == 8:
                print('post_train %03d %02d' % (ii, jj))
                post_train(ii, natoms=natoms)
            else:
                raise RuntimeError('unknown task %d, something wrong' % jj)
            record_iter(record, ii, jj)


if __name__ == '__main__':
    main()
