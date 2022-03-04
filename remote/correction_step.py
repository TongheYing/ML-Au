#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os
from remote.dispatcher.Dispatcher import make_dispatcher
from local.script.read_outcar import read_100random
from local.script.separa_chk_similarity import remove_abnormal, elim_similarity


def make_correction(iter_index, natoms=None):
    pass


def run_correction(iter_index, natoms=None, method=None):
    """submit jobs to corresponding supercomputer nodes and check whether they are finished periodically"""
    resources = {'task_max': 10,
                 'time_limit': False,
                 'job_name': 'N'+str(natoms),
                 'partition': '',
                 'node': ''}
    commands = ['    mpijob /opt/vasp/5.4.1/bin/vasp_gam']
    work_path = 'trans_cor'
    str_iter_index = str(iter_index+1).zfill(3)
    run_tasks = ['correction_' + str_iter_index, 'train_' + str_iter_index]

    files = os.listdir('trans_cor' + '/train_' + str_iter_index + '/updating_' + str_iter_index)
    noffiles = len(files)

    train_group_size = 1
    trans_comm_data = ['']
    forward_files = {
        'correction_' + str_iter_index: ['update_' + str_iter_index, 'INCAR', 'POTCAR', 'KPOINTS', 'read_static.py'],
        'train_' + str_iter_index: ['updating_' + str_iter_index, 'INCAR', 'POTCAR', 'KPOINTS']}
    backward_files = {'correction_' + str_iter_index: ['id_prop.csv'],
                      'train_' + str_iter_index: ['']}
    args = {'add_sub': True,
            'correction_' + str_iter_index: ['150', 'update_' + str_iter_index, 'smallopa', '56'],
            'train_' + str_iter_index: [str(noffiles), 'updating_' + str_iter_index, 'test', '20']}
    jdata = {'hostname': 'yingth_ustc',
             'remote_workpath': '/home/wgzhu/yingth/N'+str(natoms)+'_'+method+'_auto'}

    dispatcher = make_dispatcher(jdata=jdata)
    dispatcher.run_jobs(resources,
                        commands,
                        work_path,
                        run_tasks,
                        train_group_size,
                        trans_comm_data,
                        forward_files,
                        backward_files,
                        outlog='log',
                        errlog='err',
                        args=args,
                        natoms=natoms)


def post_correction(iter_index, natoms=None):
    """deal with OUTCAR which is downloaded to the local train directory, eliminating abnormal and similar structures"""
    cwd = os.getcwd()
    str_iter_index = str(iter_index + 1).zfill(3)
    os.system('rm -rf trans_cor/jr.json')
    files = os.listdir('trans_cor/train_'+str_iter_index+'/updating_'+str_iter_index)
    numoffiles = len(files)
    os.chdir('trans_cor/train_'+str_iter_index)
    if numoffiles > 0:
        if 'N'+str(natoms)+'_dataset' in os.listdir('./'):
            os.system('rm -rf N'+str(natoms)+'_dataset')
        os.mkdir('N'+str(natoms)+'_dataset')
        read_100random(num=numoffiles, natoms=natoms)
        remove_abnormal(natoms=natoms)
        elim_similarity(natoms=natoms, index=iter_index)
        os.chdir(cwd)
        if 'new_updating_' + str_iter_index in os.listdir('trans_cor/train_' + str_iter_index):
            os.system('rm -rf trans_cor/train_' + str_iter_index + '/new_updating_' + str_iter_index)
        os.system('cp -r trans_cor/train_' + str_iter_index +
                  '/N'+str(natoms)+'_dataset/normal_N'+str(natoms)+'_dataset/simplified_dataset_N'+str(natoms)+' trans_cor/train_' + str_iter_index)
        os.system('mv trans_cor/train_' + str_iter_index +
                  '/simplified_dataset_N'+str(natoms)+' trans_cor/train_' + str_iter_index + '/new_updating_' + str_iter_index)
    elif numoffiles == 0:
        os.system('touch updating_'+str_iter_index+'/id_prop.csv')
        os.chdir(cwd)
        if 'new_updating_' + str_iter_index in os.listdir('trans_cor/train_' + str_iter_index):
            os.system('rm -rf trans_cor/train_' + str_iter_index + '/new_updating_' + str_iter_index)
        os.system('mv trans_cor/train_'+str_iter_index+'/updating_'+str_iter_index +
                  ' trans_cor/train_'+str_iter_index+'/new_updating_'+str_iter_index)

    for ii in range(1, iter_index+2):
        ii_str = str(ii).zfill(3)
        files = os.listdir('trans_cor/train_'+ii_str+'/new_updating_' + ii_str)
        noffiles = len(files) - 1
        if noffiles < 0:
            os.system('touch trans_cor/train_'+ii_str+'/new_updating_'+ii_str+'/id_prop.csv')

    if 'train_'+str_iter_index+'-1' in os.listdir('trans_tra/'):
        os.system('rm -rf trans_tra/train_'+str_iter_index+'-1')
    os.mkdir('trans_tra/train_' + str_iter_index + '-1')
    if 'train_' + str_iter_index + '-2' in os.listdir('trans_tra/'):
        os.system('rm -rf trans_tra/train_' + str_iter_index + '-2')
    os.mkdir('trans_tra/train_' + str_iter_index + '-2')
    if 'train_'+str_iter_index+'-3' in os.listdir('trans_tra/'):
        os.system('rm -rf trans_tra/train_'+str_iter_index+'-3')
    os.mkdir('trans_tra/train_' + str_iter_index + '-3')
    if 'train_'+str_iter_index+'-4' in os.listdir('trans_tra/'):
        os.system('rm -rf trans_tra/train_'+str_iter_index+'-4')
    os.mkdir('trans_tra/train_' + str_iter_index + '-4')

    os.system('cp -r trans_cor/train_'+str_iter_index+'/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-1/')
    os.system('mv trans_tra/train_'+str_iter_index+'-1/new_updating_'+str_iter_index+
              ' trans_tra/train_'+str_iter_index+'-1/updating_'+str_iter_index)
    os.system('cp -r trans_cor/train_'+str_iter_index+'/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-2/')
    os.system('mv trans_tra/train_' + str_iter_index + '-2/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-2/updating_' + str_iter_index)
    os.system('cp -r trans_cor/train_'+str_iter_index+'/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-3/')
    os.system('mv trans_tra/train_' + str_iter_index + '-3/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-3/updating_' + str_iter_index)
    os.system('cp -r trans_cor/train_'+str_iter_index+'/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-4/')
    os.system('mv trans_tra/train_' + str_iter_index + '-4/new_updating_' + str_iter_index +
              ' trans_tra/train_' + str_iter_index + '-4/updating_' + str_iter_index)

    os.system('cp trans_cor/correction_'+str_iter_index +
              '/id_prop.csv local/pso/pso_'+str_iter_index+'/update_'+str_iter_index+'/')




