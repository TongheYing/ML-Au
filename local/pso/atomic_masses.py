#!/usr/bin/env python 

def atomic_masses(symbols):
    "This function return the stantard value of atomic mass,the given symbols should be a str,such as 'H20'."
    masses={'C':12.0107,'H':1.00794,'O':15.9994,'N':14.0067,'P':30.97,'Si':28.0855,'B':10.811,'In':114.82,'Se':78.96,
             'Cd':112.421,'Te':127.60,'Cu':63.546,'Sn':118.710,'S':32.06,'Mo':95.96,'Pd':106.42,'Au':196.97,'Bi':208.98,'Ni':58.693,
             'Nb':92.906,'Zr':91.224,'W':183.84,'Ta':180.95,'Hf':178.49,'Ti':47.867,'V':50.942,'Fe':55.845,'Co':58.933,
             'Mn':54.94,'Pb':207.20,'Cr':52.00,'Cl':35.45,'Ga':69.72,
       'Ag':107.87,'Pt':195.1,'Pd':106.4}
    a=[]
    for i in symbols:
        a.append(masses[i])
    return a
