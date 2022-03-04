#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import copy
from local.pso.pso_atoms import PsoAtoms
from local.pso.Pre_Relax import PreRelax
from scipy.optimize import linear_sum_assignment


class Pso(object):
    """Pso object.

    The Pso object contains the main process of PSO.
    It must contains a list of atoms objects for optimization,a parameters
    dictionary and a calculator object for dft calculation.
    The main Pso equation:v(t+1)=w*v(t)+c1*r1*(pbest-x)+c2*r2*(gbest-x)
    The default value of this parameters is:WEIGHT=0.6,c1=c2=2,r1 and r2 is two
    seperately generated random numbers is range [0,1].
    NPAR:number of particles.It increases with the size of the system.Default
    value is 20.
    EDIFF:If the energy difference between the neighbour gbest is lower than
    it,the Pso_evolution will stop.The lower,the more precise.
    Set this parameters' value by a pso_init file or using set_parameters().
    """

    def __init__(self, atoms_list=None, parameters=None, calculator=None):

        self.set_atoms_list(atoms_list)
        self.default_parameters={'UNF':1.,'c1':2.0,'c2':2.0,'COC':0.7298,'NPAR':150,'EDIFF':1e-4,'LSS':3,'ELIR':0.4,'WEIGHT':0.9,'GER':1,'PREX':1,'FIX2':[0],'PATH':0.01,'VMAX':0.5,'SCO':0.05,'DCO':2.,'ZMAX':[1.,1.,1],'ZMIN':[0.,0.,0.],'VAC':[0.,0.,10.],'PBC':[0,0,0],'CUO':[0.8,1.6],'LPAIR':1,'LPRELAX':1}
        if parameters is None:#Use the default value
            parameters=self.default_parameters
        if 'pso_init'  in os.listdir(os.getcwd()):
            from local.pso.Read_Pso_Init import PSO_INIT

            self.pso_init=PSO_INIT()
            tmp=self.default_parameters
            for key in tmp.keys():
                if key in self.pso_init.parameters.keys():
                    tmp[key]=self.pso_init.parameters[key]
            parameters=tmp
        self.set_parameters(parameters)

        if calculator is None:
            from local.pso.nnmodel import Model
            nnmodel = Model()
            self.set_calc(nnmodel)
        self.set_calc(calculator)

    def set_calc(self,calculator):
        self._calc=calculator

    def get_calc(self):
        return self._calc

    def set_atoms_list(self,atoms_list=None):
        self._atoms_list=[]
        if atoms_list!=None:
            for atoms in atoms_list:
                self.add_atoms(atoms)

    def add_atoms(self,atoms=None):
        if not isinstance(atoms,PsoAtoms):
            raise TypeError('The given atoms is not a PsoAtoms object.')
        self._atoms_list.append(atoms)

    def del_atoms(self,atoms=None):
        self._atoms_list.remove(atoms)

    def get_atoms(self):
        return self._atoms

    def set_parameters(self,parameters):
        if not isinstance(parameters,dict):
            raise ValueError('The given format is not a dictionary')
        a=set(parameters.keys())
        b=set(self.default_parameters.keys())
        if not a<=b:
            raise ValueError
        c=self.default_parameters
        for key in parameters.keys():
            c[key]=parameters[key]
        self._parameters=c

    def get_parameters(self):
        return copy.deepcopy(self._parameters)


def pso_evo(pso=None, natoms=None):
        from local.pso.rand_stru import RandStru

        "It is the main function of Pso."
        import os
        import shutil

        def write_pso_data():
            filename='pso_data_%03d'%GER
            atoms=pso._atoms_list[0]
            f=open(filename,'w')
            f.write('System\n')
            f.write(atoms.get_system_name())
            f.write('\n')
            f.write('Subs Atom\n')
            for elem in atoms._subs_elements:
                f.write(elem+'  ')
            f.write('\n')
            for num in atoms._subs_numbers:
                f.write('%3d  '%num)
            f.write('\n')
            f.write('Abso Atom\n')
            for elem in atoms._abso_elements:
                f.write(elem+'  ')
            f.write('\n')
            for num in atoms._abso_numbers:
                f.write('%3d  '%num)
            f.write('\n')
            f.write('Lattice Constant %.16f\n'%atoms.get_lattice_constant())
            f.write('Cell\n')
            for i in atoms.get_cell():
                for j in i:
                    f.write('%.16f  '%j)
                f.write('\n')
            f.write('Pbc    ')
            for i in atoms.get_pbc():
                f.write('%d  '%i)
            f.write('\n')
            f.write('Subs Mass\n')
            for mass in atoms.get_subs_masses():
                f.write('%.4f  '%mass)
            f.write('\n')
            f.write('Abso Mass\n')
            for mass in atoms.get_abso_masses():
                f.write('%.4f   '%mass)
            f.write('\n')
            f.write('Subs Radius\n')
            for radius in atoms.get_subs_radius():
                f.write('%.4f  '%radius)
            f.write('\n')
            f.write('Abso Radius\n')
            for radius in atoms.get_abso_radius():
                f.write('%.4f  '%radius)
            f.write('\n')
            f.write('Constraints\n')
            for i in atoms.get_constraints():
                for j in i:
                   f.write('%d '%j)
            f.write('\n')
            f.write('Subs Structure\n')
            for atom in atoms.get_subs_positions():
                for cord in atom:
                    f.write('%.16f  '%cord)
                f.write('\n')
            f.write('Pso Parameters\n')
            for key in pso._parameters.keys():
                f.write('%s  '%key)
                value=pso._parameters[key]
                if isinstance(value,list):
                    for i in value:
                        f.write('%.8f  '%i)
                    f.write('\n')
                else:
                    f.write('%.8f  \n'%pso._parameters[key])
            f.write('Calculator  '+pso._calc._name+'\n')
            f.write('!!!!!!!!!!!!!!!!!!!!!!!\n')
            f.write('          Generation  %d\n'%GER)
            f.write('\n')
            if GER==1:
                f.write('Last Gbest  %.16f\n'%0)
            else:
                f.write('Last Gbest  %.16f\n'%gbest[1])
                f.write('\n&&&&&&&&&&&&&&&  Number of Eliminated structures  &&&&&&&&&&&&&&&\n')
                for i in elim_list:
                    f.write('%2d  '%i)
                f.write('\n\n')
            for i1,atoms in enumerate(pso._atoms_list):
                stru=atoms.get_abso_positions()
                f.write('----------Particle  %d----------\n'%i1)
                f.write('Positions\n')
                for atom in stru:
                    for cord in atom:
                        f.write('%.16f    '%cord)
                    f.write('\n')
                f.write('Velocities\n')
                for i2 in range(len(stru)):
                    for v in velocities[i1,i2,:]:
                        f.write('%.16f    '%v)
                    f.write('\n')
                f.write('            *******************************\n')
            f.close()

        def new_velocity(atoms,v,pbest,gbest,lpair):
            from math import e
            v1,v2=[],[]
            c1,c2=pso._parameters['c1'],pso._parameters['c2']
            x=pso._parameters['COC']
            w=pso._parameters['WEIGHT']
            r1=np.random.rand()
            r2=np.random.rand()
            w = 0.4 + 0.5 / GER
            if np.abs(pbest[0]-gbest[1])<=5e-2:
                w=0.25
            f2.write('x  %.3f; w  %.3f; c1  %.3f; c2  %.3f; r1  %.3f; r2  %.3f\n'%(x,w,c1,c2,r1,r2))
            f2.write('Last Velocities\n')
            for i in v:
                 for j in i:
                     f2.write('%.16f  '%j)
                 f2.write('\n')
            temp=0
            stru=atoms.get_abso_positions(cord_mod='r')
            pbest=pbest[1].get_abso_positions(cord_mod='r')
            gbest=gbest[2].get_abso_positions(cord_mod='r')
            f2.write('Pbest\n')
            for n in atoms.get_abso_numbers():
                dist=[atoms.get_distance(atom1,atom2)[0] for atom1 in stru[temp:temp+n] for atom2 in pbest[temp:temp+n]]
                dist=np.array(dist)
                dist=dist.reshape(n,n)
                for i in dist:
                    for j in i:
                        f2.write('%.16f    '%j)
                    f2.write('\n')
                if lpair:
                    path1=get_pair(dist)[0]
                else:
                    path1=range(len(dist))
                for i in path1:
                    f2.write('%d    '%i)
                f2.write('\n')
                for i,j in enumerate(path1):
                    v1.append(atoms.get_distance(pbest[temp+j],stru[temp+i])[1])
                temp+=n
            v1=np.array(v1)
            f2.write('v1\n')
            for i in v1:
                for j in i:
                    f2.write('%.16f  '%j)
                f2.write('\n')
            temp=0
            f2.write('Gbest\n')
            for n in atoms.get_abso_numbers():
                dist=[atoms.get_distance(atom1,atom2)[0] for atom1 in stru[temp:temp+n] for atom2 in gbest[temp:temp+n] ]
                dist=np.array(dist)
                dist=dist.reshape(n,n)
                for i in dist:
                    for j in i:
                        f2.write('%.16f    '%j)
                    f2.write('\n')
                if lpair:
                    path1=get_pair(dist)[0]
                else:
                    path1=range(len(dist))
                for i in path1:
                    f2.write('%d    '%i)
                f2.write('\n')
                for i,j in enumerate(path1):
                    v2.append(atoms.get_distance(gbest[temp+j],stru[temp+i])[1])
                temp+=n
            v2=np.array(v2)
            f2.write('v2\n')
            for i in v2:
                for j in i:
                    f2.write('%.16f  '%j)
                f2.write('\n')
            f2.write('\n')
            new_velo=x*(c1*r1*v1+c2*r2*v2)+w*v
            return new_velo

        init_dir=os.getcwd()
        pbest=[]
        gbest=[0,1e5,None]

        # initialization, generate the initial velocity randomly in the range [-0.1,0.1]
        if pso is not None:
            if not isinstance(pso, Pso):
                raise ValueError('NO Pso Object')
            ediff=pso._parameters['EDIFF']
            npar=pso._parameters['NPAR']
            c1=pso._parameters['c1']
            c2=pso._parameters['c2']
            unf=pso._parameters['UNF']
            coc=pso._parameters['COC']
            lss=pso._parameters['LSS']
            elir=pso._parameters['ELIR']
            GER=pso._parameters['GER']
            vmax=pso._parameters['VMAX']
            vac=pso._parameters['VAC']
            dis_cutoff=pso._parameters['DCO']
            lprelax=pso._parameters['LPRELAX']
            lpair=pso._parameters['LPAIR']
            ntsubs=len(pso._atoms_list[0]._subs_symbols)
            ntabso=len(pso._atoms_list[0]._abso_symbols)
            f0 = open('pso_data','a')
            f0.write('%s\n' % pso._atoms_list[0].get_system_name())
            f0.write('Parameters\n')
            for key in pso._parameters.keys():
                f0.write('%s  '%key)
                value = pso._parameters[key]
                if isinstance(value, list):
                    for i in pso._parameters[key]:
                        f0.write('%.3f  ' % i)
                    f0.write('\n')
                else:
                    f0.write('%.3f\n' % pso._parameters[key])
            f0.write("--------Substrate's Atoms' Positions:--------\n")
            for atom in pso._atoms_list[0].get_subs_positions():
                for cord in atom:
                    f0.write('%.16f  '%cord)
                f0.write('\n')
            f0.write('**********************************\n')
            velocities = vmax-2*vmax*np.random.rand(npar, ntabso, 3)
            dirname = 'pso_'+'001'
            if lprelax:
                PreRelax(pso, filename='pre_relax_001', dirt=init_dir)
            if dirname in os.listdir(os.getcwd()):shutil.rmtree(dirname)
            os.mkdir(dirname)
            os.chdir(dirname)
            for n, atoms in enumerate(pso._atoms_list):
                dirname1 = dirname+'_'+'%04d'%n
                if dirname1 in os.listdir(os.getcwd()):shutil.rmtree(dirname1)
                os.mkdir(dirname1)
                os.chdir(dirname1)
                pso._calc.sp_run(atoms=atoms, input_dir=init_dir)
                os.chdir('..')
            write_pso_data()
            os.chdir('..')
        else:
            # Read information from pso_data
            from local.pso.Read_Pso_Init import PSO_INIT
            pso_init=PSO_INIT()
            GER=pso_init.parameters['GER']

            updated = pso_init.parameters['UPDATED']
            f0=open('pso_data','a')
            os.chdir('pso_%03d'%GER)
            pso,last_gbest,last_elim=read_data(filename='pso_data_%03d'%GER)
            ediff=pso._parameters['EDIFF']
            npar=pso._parameters['NPAR']
            c1=pso._parameters['c1']
            c2=pso._parameters['c2']
            unf=pso._parameters['UNF']
            coc=pso._parameters['COC']
            lss=pso._parameters['LSS']
            elir=pso._parameters['ELIR']
            vmax=pso._parameters['VMAX']
            vac=pso._parameters['VAC']
            zmax=pso._parameters['ZMAX']
            zmin=pso._parameters['ZMIN']
            sim_cutoff=pso._parameters['SCO']
            dis_cutoff=pso._parameters['DCO']
            lprelax=pso._parameters['LPRELAX']
            lpair=pso._parameters['LPAIR']
            gen =pso._parameters['GER']
            ntsubs=len(pso._atoms_list[0]._subs_symbols)
            ntabso=len(pso._atoms_list[0]._abso_symbols)

            # Main Loop
            # Read information from result
            os.chdir('..')
            os.chdir('pso_%03d'%GER)
            dir_list=os.listdir(os.getcwd())
            numofupdate = 0
            numofinit = 0
            numoftest = 0

            for n in range(int(npar)):
                print("it's a {0} particle".format(n))
                dirname = 'pso_%03d'%GER+'_%04d'%n
                os.chdir(dirname)

                if not updated:
                    from local.optimize_cluster import structure_optimization
                    from local.error_indicator import read_trajectory
                    if n == 0:
                        if 'update_' + str(int(GER)).zfill(3) in os.listdir("../"):
                            shutil.rmtree('../update_' + str(int(GER)).zfill(3))
                        os.mkdir('../update_' + str(int(GER)).zfill(3))

                        if 'updating_' + str(int(GER)).zfill(3) in os.listdir('../'):
                            os.system('rm -rf ../updating_' + str(int(GER)).zfill(3))
                        if 'test_store_' + str(int(GER)).zfill(3) in os.listdir('../'):
                            os.system('rm -rf ../test_store_' + str(int(GER)).zfill(3))

                    structure_optimization(filename='POSCAR', gen=GER, natoms=natoms)
                    numofupdate, numofinit, numoftest = \
                        read_trajectory(gen=GER, prev_update=numofupdate, prev_init=numofinit, prev_test=numoftest, natoms=natoms)
                    os.system('cp ../../../clustertut/ase_calcs/optimization.poscar ./POSCAR_pbest')
                    os.system('cp ./POSCAR_pbest ../update_' + str(int(GER)).zfill(3) + '/POSCAR_' + str(n).zfill(4))

                energy=pso._calc.get_energy(updated=updated, num=n, gen=GER)
                pso._atoms_list[n].set_atoms_energy(energy)
                pi = [energy, pso._atoms_list[0].copy()]
                abso_stru=pso._calc.get_stru(pi[1])
                pi[1].set_abso_positions(abso_stru, cord_mod='d')
                pbest.append(pi)
                os.chdir('..')


            if updated:
                energies=[i[0] for i in pbest]
                energies=np.array(energies)
                energy_sort=np.argsort(energies)
                gbest=[energies.argmin(),energies.min(),pbest[energies.argmin()][1].copy()]
                velocities=[atoms.get_atoms_velo() for atoms in pso._atoms_list]
                velocities=np.array(velocities)
                filename='pso_data_%03d'%GER
                f2=open(filename,'r')
                f3=open('tmp','w')
                count=0
                for line in f2:
                    if 'Last Gbest' in line:
                        f3.write('Energies sort list\n')
                        for i in energy_sort:
                            f3.write('%d    %.16f\n'%(i,energies[i]))
                    if '            *******************************' in line:
                        f3.write('Pbest Positions and Energy\n')
                        for i in pbest[count][1].get_abso_positions():
                            for j in i:
                                f3.write('%.16f  '%j)
                            f3.write('\n')
                        f3.write('Pbest Free Energy  %.16f\n'%pbest[count][0])
                        count+=1
                    f3.write(line)

                f3.write('----------Gbest Positions and Energy----------\n')
                f3.write('Gbest Positions\n')
                for i in gbest[2].get_abso_positions():
                    for j in i:
                        f3.write('%.16f  '%j)
                    f3.write('\n')
                f3.write('Gbest Free Energy  %.16f\n'%gbest[1])
                f3.write('Gbest Number  %d\n'%gbest[0])
                f3.close()
                f2.close()
                os.rename('tmp',filename)
                os.chdir(init_dir)
                if np.abs(gbest[1]-last_gbest)>=np.abs(ediff):
                    GER+=1
                    pso._parameters['GER']+=1
                    # Update Swarm
                    f2=open('velocities_%03d'%GER,'w')
                    for n,atoms in enumerate(pso._atoms_list):
                        f2.write('*************** Particle  %d ***************\n'%n)
                        velocities[n]=new_velocity(atoms,velocities[n],pbest[n],gbest,lpair)
                    f2.close()
                    #eliminate the high energy structures,and substitute them by new random structures.
                    neli=int(elir*npar)
                    elim_list=energy_sort[-neli:]
                    surv_list=energy_sort[:-neli]
                    elim_list=list(elim_list)

                    surv_list=list(surv_list)

                    surv_list.reverse()
                    tmp1=[]
                    tmp2=[]
                    #if one structure is both in last_elim and elim,we do not eliminate it and keep it oen generation!
                    for n in elim_list:
                        if n in last_elim:
                            for m in surv_list:
                                if m not in last_elim:
                                    surv_list.remove(m)
                                    tmp1.append(m)
                                    tmp2.append(n)
                                    break
                                if m==surv_list[-1]:
                                    tmp1.append(n)
                        else:
                            tmp1.append(n)
                    elim_list=tmp1
                    surv_list.extend(tmp2)
                    for n in elim_list:
                        atoms=pso._atoms_list[n]
                        RandStru(atoms, natoms=natoms)
                        velocities[n]=vmax-vmax*2*np.random.rand(ntabso,3)
                    for n in surv_list:
                        atoms=pso._atoms_list[n]
                        stru=atoms.get_abso_positions(cord_mod='r')
                        stru=stru+velocities[n]
                        atoms.set_abso_positions(stru,cord_mod='r')
                     #Evaluate Swarm
                    if lprelax:
                        PreRelax(pso,filename='pre_relax_%03d'%GER,dirt=init_dir)
                    f0.write('Generation %d\n'%GER)
                    dirname='pso_'+'%03d'%GER
                    if dirname in os.listdir(os.getcwd()):shutil.rmtree(dirname)
                    os.mkdir(dirname)
                    os.chdir(dirname)
                    for n,atoms in enumerate(pso._atoms_list):
                        dirname1=dirname+'_'+'%04d'%n
                        if dirname1 in os.listdir(os.getcwd()):shutil.rmtree(dirname1)
                        os.mkdir(dirname1)
                        os.chdir(dirname1)
                        pso._calc.sp_run(atoms=atoms,input_dir=init_dir)
                        os.chdir('..')
                    temp = pso._atoms_list
                    write_pso_data()
                    print('Done!')
                    os.chdir('..')
                else:  # energy converge
                    print('COMPLETED!')
                    f0.write('\n\n***************Energy Converged!***************\n')
                    return gbest[2]
        f0.close()


def read_data(filename='pso_data_001'):
    # from vasp import Vasp
    from local.pso.nnmodel import Model
    'read information from pso-data file and return a Pso object'
    f=open(filename,'r')
    print(filename)
    data=[line.strip() for line in f ]
    count=0
    last_elim=[]
    last_gbest=0.
    for n,line in enumerate(data):
        if 'System' in line:
            system_name=data[n+1]
            continue
        if 'Subs Atom' in line:
            subs_elements=data[n+1].split()
            subs_numbers=[int(i) for i in data[n+2].split()]
            nsubs=sum(subs_numbers)
            continue
        if 'Abso Atom' in line:
            abso_elements=data[n+1].split()
            abso_numbers=[int(i) for i in data[n+2].split()]
            nabso=sum(abso_numbers)
            continue
        if 'Lattice Constant' in line:
            lattice_constant=float(line.split()[2])
            continue
        if 'Cell' in line:
            cell=[float(j) for i in range(3) for j in data[n+i+1].split()]
            continue
        if 'Pbc' in line:
            pbc=[int(i) for i in line.split()[1:]]
        if 'Subs Mass' in line:
            subs_masses=[float(i) for i in data[n+1].split()]
            continue
        if 'Abso Mass' in line:
            abso_masses=[float(i) for i in data[n+1].split()]
            continue
        if 'Subs Radius' in line:
            subs_radius=[float(i) for i in data[n+1].split()]
            continue
        if 'Abso Radius' in line:
            abso_radius=[float(i) for i in data[n+1].split()]
            continue
        if 'Constraints' in line:
            const=[int(i) for i in data[n+1].split()]
            continue
        if 'Subs Structure' in line:
            subs_positions=[float(j) for i in range(nsubs) for j in data[n+1+i].split()]
            continue
        if 'Parameters' in line:
            parameters={}
            for i in data[n+1:]:
                if 'Calculator' in i:
                    calc=i
                    break
                key=i.split()[0]
                a=[float(j) for j in i.split()[1:]]
                if len(a)==1 and key!='FIX2':a=a[0]
                parameters[key]=a
            cot=1
            atoms_list=[]
            while cot<=parameters['NPAR']:
                subs_symbols=[]
                abso_symbols=[]
                for i,elem in enumerate(subs_elements):
                    subs_symbols.extend([elem]*subs_numbers[i])
                for i,elem in enumerate(abso_elements):
                    abso_symbols.extend([elem]*abso_numbers[i])
                atoms=PsoAtoms(name=system_name,subs_symbols=subs_symbols,                                                         abso_symbols=abso_symbols,subs_positions=subs_positions,                                            subs_masses=subs_masses,abso_masses=abso_masses,
                               subs_radius=subs_radius,abso_radius=abso_radius,cell=cell,
                               lattice_constant=lattice_constant,constraints=const,pbc=pbc)
                atoms_list.append(atoms)
                cot+=1
            nnmodel = Model(lattice_parameter=lattice_constant)
            pso=Pso(atoms_list=atoms_list,calculator=nnmodel,parameters=parameters)
            print('Pso Object Done!')
            print(len(pso._atoms_list))
        if 'Last Gbest' in line:
            last_gbest=float(line.split()[2])
            continue
        if 'Number of Eliminated structures' in line:
            last_elim=[int(i) for i in data[n+1].split()]
            continue
        if 'Positions' in line and 'Pbest' not in line and 'Gbest' not in line:
            abso_positions=[float(j) for i in range(nabso) for j in data[n+1+i].split()]
            pso._atoms_list[count].set_abso_positions(abso_positions)
            continue
        if 'Velocities' in line:
            velocities=[float(j) for i in range(nabso) for j in data[n+1+i].split()]
            pso._atoms_list[count].set_atoms_velo(velocities)
            count+=1
            continue

    #print abso_elements,abso_numbers,abso_symbols
    return pso,last_gbest,last_elim


def get_pair(dist):

    row,col=linear_sum_assignment(dist)
    return col,dist[row,col].sum()
