#!/usr/bin/env python 
'this program is reading the information in the pso_init file'

import os
import re

class PSO_INIT():
    def __init__ (self,dirt='.'):
        orgi_dirt=os.getcwd()
        os.chdir(dirt)
        f1=open('pso_init','r')
        data=[line.strip() for line in f1]
        f1.close()
        os.chdir(orgi_dirt)
        self.atoms_radius=[]
        self.parameters={}
        for n,j in enumerate(data):
           j=j.strip()
           if re.match('NPAR|PATH|EDIFF|GER|ELIR|VMAX|PREX',j) is not None:
               self.parameters[j.split()[0]]=float(j.split()[1])
           if re.match('LPAIR|LPRELAX|UPDATED',j) is not None:
               if j.split()[1]=='True' or j.split()[1]=='T':
                   self.parameters[j.split()[0]]=1
               elif  j.split()[1]=='False' or j.split()[1]=='F':
                   self.parameters[j.split()[0]]=0
           if re.match('Radius',j) is not None:
               tmp=data[n+1].strip()
               tmp=tmp.split(';')
               self.atoms_radius=tmp     
           if re.match('ZMAX|ZMIN|VAC|PBC|CUO|FIX2',j) is not None:
               key=j.split()[0]
               a=j.split()[1:]
               if len(a)==1 :
                   b=[]
                   if key=='VAC':
                       b=[0.,0.]
                   elif key=='ZMAX':
                       b=[1.,1.]
                   elif key=='ZMIN':
                       b=[0.,0.]
                   b.append(float(a[0]))
               else:
                   b=[float(m) for m in a]
               self.parameters[key]=b
        if 'NPAR' in self.parameters.keys():
            self.parameters['NPAR']=int(self.parameters['NPAR'])
        #print self.parameters,self.atoms_radius
