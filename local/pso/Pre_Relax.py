#!/usr/bin/env python
# -*- coding:utf-8 -*-
'this program make sure that the distance between atoms in a structure generated randomly is not too small'

import numpy as np
import os
import re
from time import perf_counter
from local.pso.pso_atoms import PsoAtoms
import random
from local.pso.rand_stru import RandStru
import math


def PreRelax(pso,filename='pre_relax',dirt='.'):
    atoms_list=pso._atoms_list
    parameters=pso.get_parameters()
    gen = parameters['GER']
    subs_stru=atoms_list[0].get_subs_positions()
    prex=parameters['PREX']
    fix2=parameters['FIX2']
    PATH=parameters['PATH']
    cut_off_min=parameters['CUO'][0]
    cut_off_max=parameters['CUO'][1]
    f1=open(filename,'w')
    f1.write(atoms_list[0].get_system_name())
    f1.write('\n')
    f1.write('Parameters\n')
    f1.write('PATH  %.4f\n'%PATH)
    f1.write('CUO   %.4f  %.4f\n'%(cut_off_min,cut_off_max))
    f1.write('Cell\n')
    for i in atoms_list[0].get_cell():
        for j in i:
            f1.write('%.16f  '%j)
        f1.write('\n')
    f1.write('Lattice Constant %.8f\n'%atoms_list[0].get_lattice_constant())
    f1.write('Substrate Structure\n')
    for symbol in atoms_list[0].get_subs_symbols():
        f1.write(symbol+'  ')
    f1.write('\n')
    for atom in subs_stru:
        for i in atom:
            f1.write('%.16f  '%i)
        f1.write('\n')
    f1.write('Abso Symbols\n')
    for symbols in atoms_list[0].get_abso_symbols():
        f1.write(symbols+'  ')
    f1.write('\n')

    def write_file(atoms,forces):
        for i,atom in enumerate(atoms.get_abso_positions()):
            for cord in atom:
                f1.write('%.8f  '%cord)
            for j in forces[i]:
                f1.write('%.8f  '%j)
            f1.write('\n')
        f1.write('\n')

    def pre_relax(atoms_list, gen=1):
        sta=perf_counter()
        # print('Pre_relax Starting------------------------')
        step1=[0]*int(parameters['NPAR'])
        step2=[0]*int(parameters['NPAR'])
        for i,atoms1 in enumerate(atoms_list):
            if prex==2:
                cell=atoms1.get_cell()
                pbc=atoms1.get_pbc()
                constraints=atoms1.get_constraints()
                lc=atoms1.get_lattice_constant()
                symbols=atoms1.get_abso_symbols()
                positions=atoms1.get_abso_positions()
                radius=atoms1.get_abso_radius()
                subs_sym=[]
                subs_radius=[]
                subs_pos=[]
                abso_sym=[]
                abso_radius=[]
                abso_pos=[]
                for idx in range(len(symbols)):
                    if idx in fix2:
                        subs_sym.append(symbols[idx])
                        subs_radius.append(radius[idx])
                        subs_pos.append(positions[idx])
                    else:
                        abso_sym.append(symbols[idx])
                        abso_radius.append(radius[idx])
                        abso_pos.append(positions[idx])

                atoms=PsoAtoms(abso_symbols=abso_sym,subs_symbols=subs_sym,
                               abso_positions=abso_pos,subs_positions=subs_pos,
                               abso_radius=abso_radius,subs_radius=subs_radius,
                               cell=cell,lattice_constant=lc,pbc=pbc,
                               constraints=constraints)
            else:
                atoms=atoms1
            step=0
            while step < 100:
                # print('**************step start**********',step)
                f1.write('\n\n\n')
                f1.write('Structure-%d\n'%i)
                f1.write('**************step %d start**********\n'%step)
                f1.write('Compress the atoms\n')
                stru,step2[i]=compress(atoms,cut_off_max,PATH)
                # print('Compress Done  %d'%step2[i])
                f1.write('Compress Done  %d\n'%step2[i])
                atoms.set_abso_positions(stru)
                f1.write('Seperate the atoms\n')
                atoms.set_abso_positions(stru)
                bol,resultant_forces=get_resultant_forces(atoms,cut_off_min)
                init_forces=resultant_forces
                write_file(atoms,init_forces)
                step1[i]=0
                while bol and step1[i] < 500:
                    step1[i]+=1
                    for j,atom in enumerate(stru):
                        if np.linalg.norm(resultant_forces[j]) != 0:
                            move_path=PATH
                            f=resultant_forces[j]
                            atom=atom+f*move_path
                        atom=atoms.move_atom(atom)
                        stru[j]=atom
                    atoms.set_abso_positions(stru)
                    bol,resultant_forces=get_resultant_forces(atoms,cut_off_min)
                write_file(atoms,resultant_forces)
                f1.write('Seperate Done   %d\n'%step1[i])
                bol = 1
                if bol:
                    break

            if prex==2:
                abso_pos=atoms1.get_abso_positions()
                count=0
                for idx in range(len(abso_pos)):
                    if idx not in fix2:
                        abso_pos[idx]=atoms.get_abso_positions()[count]
                        count+=1
                atoms1.set_abso_positions(abso_pos)
            if prex==1 or prex==2:
                abso_pos=atoms1.get_abso_positions(cord_mod='d')
                idx1=np.argmin(abso_pos[:,2])
                subs_pos=atoms1.get_subs_positions(cord_mod='d')
                idx2=np.argmax(subs_pos[:,2])
                cc=np.linalg.norm(atoms1.get_cell()[2])*atoms1.get_lattice_constant()
                tmp=(atoms1.get_subs_radius()[idx2]+atoms1.get_abso_radius()[idx1])*cut_off_min/cc+subs_pos[idx2,2]
                dela=np.array([0,0,tmp-abso_pos[idx1,2]])
                abso_pos=abso_pos+dela
                atoms1.set_abso_positions(abso_pos,cord_mod='d')
        time_cost=perf_counter()-sta
        f1.write('    --------Result--------\n')
        f1.write('Time Cost   %.12f'%time_cost)
        f1.close()

    pre_relax(atoms_list, gen=gen)


def get_resultant_forces(atoms,cut_off_min=0.9):
        ll=len(atoms.get_abso_symbols())
        resultant_forces=[]
        forces=np.zeros(shape=(ll,ll,3),dtype=float)
        bol=False
        abso_stru=atoms.get_abso_positions()
        subs_stru=atoms.get_subs_positions()
        abso_radius=atoms.get_abso_radius()
        subs_radius=atoms.get_subs_radius()
        for i,atom1 in enumerate(abso_stru):
            radius_1=abso_radius[i]
            for j in range(ll):
                if j >i:
                    distance,vector=atoms.get_distance(atom1,abso_stru[j])
                    radius_2=abso_radius[j]
                    if distance<(radius_1+radius_2)*cut_off_min:
                        forces[i,j,:]=vector/np.linalg.norm(vector)
                elif j<i:
                    forces[i,j,:]=-forces[j,i,:]
            force=np.sum(forces[i,:,:],axis=0)

            for j,atom2 in enumerate(subs_stru):
                distance,vector=atoms.get_distance(atom1,atom2)
                radius_2=subs_radius[j]
                if distance<(radius_1+radius_2)*cut_off_min:
                    force=force+vector/np.linalg.norm(vector)
            if force.any()!=0.:bol=True
            resultant_forces.append(force)
        return bol,resultant_forces


def compress(atoms,cut_off_max=1.05,PATH=0.01):
        abso_stru=atoms.get_abso_positions()
        tmp = len(abso_stru)
        subs_stru=atoms.get_subs_positions()
        abso_radius=atoms.get_abso_radius()
        subs_radius=atoms.get_subs_radius()
        # move all atoms into a same supercell
        for n,atom in enumerate(abso_stru):
            atom=atoms.move_atom(atom)
            abso_stru[n]=atom
        atoms.set_abso_positions(abso_stru)
        abso_stru=atoms.get_abso_positions()
        #-------------------------
        nabso=len(abso_stru)
        idx1 = [i for i in range(nabso)]
        idx3=[]
        step=0
        maxstep=500
        ll=len(idx1)
        while ll>0 and step<maxstep:
            while ll>0:
                for n in idx1:
                    atom1=abso_stru[n]
                    radius1=abso_radius[n]
                    for m in idx3:
                        radius2=abso_radius[m]
                        if  atoms.get_distance(atom1,abso_stru[m])[0]<=cut_off_max*(radius1+radius2):
                            idx3.insert(0,n)
                            idx1.remove(n)
                            break
                    if n in idx1:
                        for m,atom2 in  enumerate(subs_stru):
                            radius2=subs_radius[m]
                            if  atoms.get_distance(atom1,atom2)[0]<=cut_off_max*(radius1+radius2):
                                idx3.insert(0,n)
                                idx1.remove(n)
                                break
                if ll==len(idx1):
                    break
                else:
                    ll=len(idx1)
            ll=len(idx1)
            if ll>0:
                step+=1
                for n in idx1:
                    atom1=abso_stru[n]
                    radius1=abso_radius[n]
                    s=np.sum(subs_stru,axis=0)
                    if len(idx3)==0:
                        #move the abso atoms together onto the subs atoms
                        distance,vector=atoms.get_distance(atom1,subs_stru[0])
                        atom1=atom1-vector*PATH*distance
                    else:
                        for m in idx3:
                            s=s+abso_stru[m]
                        core=s/(len(subs_stru)+len(idx3))
                        distance,vector=atoms.get_distance(atom1,core)
                        atom1=atom1-vector*PATH*distance/radius1**2
                    abso_stru[n]=atom1
        return abso_stru,step




