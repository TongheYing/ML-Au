#!/anaconda3/envs/schnet/bin/python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os
from remote.dispatcher.Dispatcher import make_dispatcher


def make_train(iter_index, natoms=None):
    pass


def run_train(iter_index, natoms=None, method=None):
    resources = {'task_max': 10,
                 'time_limit': False,
                 'job_name': 'N'+str(natoms)+'-'+method,
                 'node': 'k804'}

    commands = [""]
    work_path = 'trans_tra'
    str_iter_index = str(iter_index+1).zfill(3)
    run_tasks = ['train_' + str_iter_index + '-1', 'train_' + str_iter_index + '-2',
                 'train_' + str_iter_index + '-3', 'train_' + str_iter_index + '-4']

    data = []
    for ii in range(1, iter_index+2):
        ii_str = str(ii).zfill(3)
        files = os.listdir('trans_cor/train_'+ii_str+'/new_updating_' + ii_str)
        noffiles = len(files) - 1
        data.append(noffiles)

    with open('dataset-ML/N'+str(natoms)+'/model/record_model.json', 'r') as f:
        content = [line.strip() for line in f.readlines()]
        num = int(content[0])
    total = 0
    for i in range(num, len(data)):
        total += data[i]

    if total >= 100:
        str_data = '131'
        str_data += ' '
        for jj in range(len(data)-1):
            str_data += str(data[jj])
            str_data += ' '
        str_data += str(data[len(data)-1])

        train_group_size = 1
        trans_comm_data = ['']

        include_former = []
        for jj in range(num+1, iter_index+1):
            include_former.append('../train_'+str(jj).zfill(3)+'-1/updating_'+str(jj).zfill(3))
        include_former.append('updating_'+str_iter_index)

        forward_files = {'train_' + str_iter_index + '-1': include_former,
                         'train_' + str_iter_index + '-2': ['updating_' + str_iter_index],
                         'train_' + str_iter_index + '-3': ['updating_' + str_iter_index],
                         'train_' + str_iter_index + '-4': ['updating_' + str_iter_index]}
        backward_files = {'train_' + str_iter_index + '-1': ['clustertut-1/log.csv', 'clustertut-1/best_model'],
                          'train_' + str_iter_index + '-2': ['clustertut-2/log.csv', 'clustertut-2/best_model'],
                          'train_' + str_iter_index + '-3': ['clustertut-3/log.csv', 'clustertut-3/best_model'],
                          'train_' + str_iter_index + '-4': ['clustertut-4/log.csv', 'clustertut-4/best_model']}
        args = {'add_sub': False,
                'train_' + str_iter_index + '-1': [str(iter_index+1), str_data, 'k80', '1', str(natoms)],
                'train_' + str_iter_index + '-2': [str(iter_index+1), str_data, 'k80', '1', str(natoms)],
                'train_' + str_iter_index + '-3': [str(iter_index+1), str_data, 'k80', '1', str(natoms)],
                'train_' + str_iter_index + '-4': [str(iter_index+1), str_data, 'k80', '1', str(natoms)]}
        jdata = {'hostname': 'yingth_ustc',
                 'remote_workpath': '/home/wgzhu/yingth/home2/N'+str(natoms)+'_'+method+'_autoschnet'}

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


def post_train(iter_index, natoms=None):
    """keep the best_model and log.csv"""
    os.system('rm -rf trans_tra/jr.json')
    data = []
    for ii in range(1, iter_index + 2):
        ii_str = str(ii).zfill(3)
        files = os.listdir('trans_cor/train_'+ii_str+'/new_updating_' + ii_str)
        noffiles = len(files) - 1
        data.append(noffiles)

    with open('dataset-ML/N'+str(natoms)+'/model/record_model.json', 'r') as f:
        content = [line.strip() for line in f.readlines()]
        num = int(content[0])
    total = 0
    for i in range(num, len(data)):
        total += data[i]

    str_iter_index = str(iter_index + 1).zfill(3)
    if total >= 100:
        with open('dataset-ML/N'+str(natoms)+'/model/record_model.json', 'w') as f:
            f.write(str(iter_index+1))

        for ii in range(4):
            os.system('cp trans_tra/train_'+str_iter_index+'-'+str(ii+1)+'/clustertut-'+str(ii+1)+'/best_model '
                                                           'dataset-ML/N'+str(natoms)+'/model/best_model-'+str(ii+1)+'_'+str_iter_index)
            os.system('cp trans_tra/train_'+str_iter_index+'-'+str(ii+1)+'/clustertut-' + str(ii + 1)
                      + '/log.csv dataset-ML/N'+str(natoms)+'/model/log-' + str(ii + 1) + '_' + str_iter_index+'.csv')
    else:
        for ii in range(1, 5):
            os.system('cp dataset-ML/N'+str(natoms)+'/model/best_model-'+str(ii)+'_'+str(iter_index).zfill(3) +
                      ' dataset-ML/N'+str(natoms)+'/model/best_model-'+str(ii)+'_'+str_iter_index)
            os.system('cp dataset-ML/N'+str(natoms)+'/model/log-' + str(ii) + '_' + str(iter_index).zfill(3) +
                      '.csv dataset-ML/N'+str(natoms)+'/model/log-' + str(ii) + '_' + str_iter_index+'.csv')



