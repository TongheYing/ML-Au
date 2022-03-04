#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Tonghe Ying

import os, shutil
from local.pso.run import run


def make_pso(iter_index, natoms=None):
    if iter_index == 0:
        with open('dataset-ML/N'+str(natoms)+'/model/record_model.json', 'w') as f:
            f.write('0')
        for ii in range(1, 5):
            os.system('mv dataset-ML/N'+str(natoms)+'/model/best_model-'+str(ii)+'_000 dataset-ML/N'+str(natoms)+'/model/model_tmp-'+str(ii))
            os.system('mv dataset-ML/N' + str(natoms) + '/model/log-' + str(ii) + '_000.csv dataset-ML/N' + str(
                natoms) + '/model/tmp_log-' + str(ii))
        os.system('rm -rf dataset-ML/N'+str(natoms)+'/model/best_model* dataset-ML/N'+str(natoms)+'/model/log*.csv trans_cor/* trans_tra/*')
        for ii in range(1, 5):
            os.system('mv dataset-ML/N'+str(natoms)+'/model/model_tmp-'+str(ii)+' dataset-ML/N'+str(natoms)+'/model/best_model-'+str(ii)+'_000')
            os.system('mv dataset-ML/N' + str(natoms) + '/model/tmp_log-' + str(ii) + ' dataset-ML/N' + str(
                natoms) + '/model/log-' + str(ii) + '_000.csv')


def run_pso(iter_index, natoms=None):
    """run PSO"""
    cwd = os.getcwd()
    os.chdir('local/pso')
    run(iter_index=iter_index, natoms=natoms)
    os.chdir(cwd)


def post_pso(iter_index, natoms=None):
    """move update and updating directory after PSO is finished"""
    iter_index_str = str(iter_index+1).zfill(3)
    if 'correction_' + iter_index_str in os.listdir("trans_cor/"): shutil.rmtree(
        'trans_cor/correction_' + iter_index_str)
    os.mkdir('trans_cor/correction_'+iter_index_str)
    if 'train_' + iter_index_str in os.listdir("trans_cor/"): shutil.rmtree(
        'trans_cor/train_' + iter_index_str)
    os.mkdir('trans_cor/train_' + iter_index_str)
    os.system(
        'cp -r local/pso/pso_'+iter_index_str+'/update_'+iter_index_str+' trans_cor/correction_'+iter_index_str)
    os.system(
        'cp -r local/pso/pso_' + iter_index_str + '/updating_' + iter_index_str + ' trans_cor/train_' + iter_index_str)
    os.system('cp dataset-ML/N'+str(natoms)+'/vasp/INCAR dataset-ML/N'+str(natoms)+'/vasp/KPOINTS'
              ' dataset-ML/N'+str(natoms)+'/vasp/POTCAR dataset-ML/N'+str(natoms)+'/vasp/read_static.py trans_cor/correction_'+iter_index_str)
    os.system('cp dataset-ML/N'+str(natoms)+'/vasp/INCAR dataset-ML/N'+str(natoms)+'/vasp/KPOINTS'
              ' dataset-ML/N'+str(natoms)+'/vasp/POTCAR trans_cor/train_' + iter_index_str)
