# -*- coding:utf-8 -*-
import numpy as np
import copy
import os


class PsoAtoms(object):
    """PsoAtoms object.

    The PsoAtoms object can represent some atoms or  molecules absorbing in
    a substrate.  It has a supercell which describes the substrate's structure.

    Information about the atoms (cell and  position) is
    stored in numpy ndarrays.  Optionally, there can be information about
    masses, magnetic moments and charges.

    In order to calculate energies, forces and stresses, a calculator
    object has to attached to the atoms object.Default one is vasp!

    Parameters:

    subs_symbols: a list of str of substrates' atomic symbols
        It must be offered.
        Examples: water molecule ['H','H','O'] or 'H2O'
    abso_symbols:similarly str of adsorbing atoms' symbols
    subs_positions: ndarray of xyz-positions of subs' atoms.
        Anything that can be converted to an numpy ndarray of shape (n, 3) will do
    abso_positions:similarly,the abso atoms' positions.
    subs_masses:ndarray of float
        Masses of atoms in substrate in atomic units.
    abso_masses:list of float.
    subs_radius:list of float
        Radius of atoms in substrate.Unit is angstrom.
    abso_radius: list of float

    cell: 3*3 numpy ndarray.
        Unit cell vectors.  Can also be given as just three
        numbers for orthorhombic cells.  Default value: [1., 1., 1.].
    lattice_constant:float
        Default value is 1,all the cordinates(cell and positions) will multiply
        it.
    constraints:a list of bool.
        Each substarte' atom(obviously,that of absorbing  atomshas is True)
        a [bool,bool,bool] which represents x,y,z direction.
        If it is True,cordinates of that direction will change a little while
        structure optimizations.

    Example:
    A Ni atom on cu(111):
        Simply,supercell is 1*1.
        a= PsoAtoms(abso_symbols=['Ni'],subs_symbol=['Cu'],name='Ni on Cu(111)',
            subs_positions=[0.,0.,0.],abso_positons=[0.5,0.5,0.5],
            cell=[3.63,3.63,15])

    We do not recommend you do like this.Use the set function
    (set_abso_positons()) or offer an imput file corresponding to a calculator
    (e.g. a POSCAR file for vasp).

    """
    def __init__(self,abso_symbols=None,subs_symbols=None,
                  name=None,abso_positions=None,subs_positions=None,
                  abso_masses=None,subs_masses=None,
                  abso_radius=None,subs_radius=None,
                  cell=None,lattice_constant=None,
                  pbc=None,constraints=None):
        self.pso_init=None
        self.atoms_radius=None
        if 'pso_init' in os.listdir(os.getcwd()):
            from local.pso.Read_Pso_Init import PSO_INIT
            self.pso_init=PSO_INIT()
            if hasattr(self.pso_init,'atoms_radius'):
                self.atoms_radius=self.pso_init.atoms_radius

        if subs_symbols is not None:
            self._subs_numbers=symbols_numbers(subs_symbols)
            self._subs_elements=atoms_elements(subs_symbols)
        else:
            subs_symbols=[]

        if abso_symbols is not None:
            self._abso_numbers=symbols_numbers(abso_symbols)
            self._abso_elements=atoms_elements(abso_symbols)
        else:
            abso_symbols=[]

        if subs_positions is None:
            subs_positions=np.zeros((len(subs_symbols),3),dtype=float)
        if abso_positions is None:
            abso_positions=np.zeros((len(abso_symbols),3),dtype=float)

        if cell is None:
            cell=np.eye(3)

        if lattice_constant is None:
            lattice_constant=1.

        if pbc is None:
            pbc=[1,1,1]

        self.set_lattice_constant(lattice_constant)
        self.set_subs_symbols(subs_symbols)
        self.set_abso_symbols(abso_symbols)
        self.set_system_name(name)
        self.set_subs_positions(subs_positions)
        self.set_cell(cell)
        self.set_abso_positions(abso_positions)
        self.set_subs_masses(subs_masses)
        self.set_abso_masses(abso_masses)
        self.set_abso_radius(abso_radius)
        self.set_subs_radius(subs_radius)
        self.set_pbc(pbc)
        self.set_constraints(constraints)

    def set_subs_symbols(self,subs_symbols):
        if isinstance(subs_symbols,str):
            a=[]
            for n,char in enumerate(subs_symbols):
                if char.isalpha():
                    a.append(char)
                if char.isdigit():
                    a.extend(list(a[-1])*(int(char)-1))

        elif isinstance(subs_symbols,list):
            a=subs_symbols
        else:
            raise ValueError('The format of subs_symbols is nor right.')
        self._subs_symbols=a
        self._subs_numbers=symbols_numbers(self._subs_symbols)
        self._subs_elements=atoms_elements(self._subs_symbols)
        self.set_constraints()
        self.set_system_name()

    def set_abso_symbols(self,abso_symbols):
        if isinstance(abso_symbols,str):
            a=[]
            for char in abso_symbols:
                if char.isalpha():
                    a.append(char)
                if char.isdigit():
                    a.extend(list(a[-1])*(int(char)-1))
        elif isinstance(abso_symbols,list):
            a=abso_symbols
        else:
            raise ValueError('The format of subs_symbols is nor right.')
        self._abso_symbols=a
        self._abso_numbers=symbols_numbers(self._abso_symbols)
        self._abso_elements=atoms_elements(self._abso_symbols)

    def get_subs_symbols(self):
        return self._subs_symbols

    def get_abso_symbols(self):
        return self._abso_symbols

    def set_system_name(self,name=None):
        if name==None:
            name=get_chemical_formula(self.get_subs_symbols())
        if not isinstance(name,str):
            raise ValueError('The system name must be a str')
        self._name=name

    def get_system_name(self):
        return self._name

    def get_subs_numbers(self):
        return copy.deepcopy(self._subs_numbers)

    def get_abso_numbers(self):
        return copy.deepcopy(self._abso_numbers)

    def set_subs_positions(self,subs_positions,cord_mod='c'):
        "cord_mod---two modes:c--cart,d--direct"
        subs_positions=np.array(subs_positions)
        subs_positions=subs_positions.reshape(-1,3)
        if cord_mod=='c':
            self._subs_positions=subs_positions
        else:
            for n,atom in enumerate(subs_positions):
                subs_positions[n]=self.direct_cart(atom)
            self._subs_positions=subs_positions

    def set_abso_positions(self,abso_positions,cord_mod='c',atom=None):
        "cord_mod---three modes:c--cart,d--direct,r--relative"
        abso_positions=np.array(abso_positions)
        abso_positions=abso_positions.reshape(-1,3)
        if atom is None:
            a=self.get_subs_positions()
            if len(a)==0:a=0.
            else:a=a[-1]
        else:
            a=atom
        if cord_mod=='d':
            for n,atom in enumerate(abso_positions):
                abso_positions[n]=self.direct_cart(atom)
        elif cord_mod=='r':
            abso_positions=abso_positions+a
        self._abso_positions=abso_positions
        self.set_rela_positions(a)

    def get_subs_positions(self,cord_mod='c'):
        if cord_mod=='c':
            return self._subs_positions.copy()
        else:
            tmp=[self.cart_direct(i) for i in self._subs_positions]
            return np.array(tmp)

    def get_abso_positions(self,cord_mod='c'):
        if cord_mod=='c':
            return self._abso_positions.copy()
        elif cord_mod=='d':
            tmp=[self.cart_direct(i) for i in self._abso_positions]
            return np.array(tmp)
        else:
            return self._rela_positions

    def set_rela_positions(self,atom=None):
        "return the relative positions of abso atoms to the substrate"
        if atom is None:
            a=self.get_subs_positions()
            a=a[-1]
            if len(a)==0:a=0.
        else:
            a=atom
        rela_positions=self.get_abso_positions()-a
        self._rela_positions=rela_positions

    def get_rela_positions(self):
        return self._rela_positions.copy()

    def get_positions(self,cord_mod='c'):
        abso=self.get_abso_positions(cord_mod)
        subs=self.get_subs_positions(cord_mod)
        return np.append(subs,abso,axis=0)

    def set_cell(self,cell):
        cell=np.array(cell)
        cell=cell.reshape(3,3)
        self._cell=cell

    def get_cell(self):
        return self._cell.copy()

    def set_pbc(self,pbc=None):
        " Periodic boundary conditions The default value is False "
        if pbc is None:
            pbc=[0,0,0]
        self._pbc=pbc

    def get_pbc(self):
        return copy.deepcopy(self._pbc)

    def set_lattice_constant(self,a=1.):
        self._lattice_constant=a

    def get_lattice_constant(self):
        return self._lattice_constant

    def set_subs_masses(self,masses=None):
        if masses is None:
            from local.pso.atomic_masses import atomic_masses
            masses=atomic_masses(self._subs_symbols)#use the stantard values
        self._subs_masses=masses

    def set_abso_masses(self,masses=None):
        if masses is None:
            from local.pso.atomic_masses import atomic_masses
            masses=atomic_masses(self._abso_symbols)
        self._abso_masses=masses

    def get_subs_masses(self):
        return copy.deepcopy(self._subs_masses)

    def get_abso_masses(self):
        return copy.deepcopy(self._abso_masses)

    def set_subs_radius(self,radius=None):
        if self.atoms_radius is not None and radius==None:
            radius=[]
            for atom in self._subs_symbols:
                for i in self.atoms_radius:
                    if i.split()[0]==atom:
                        radius.append(float(i.split()[1]))
                        break
        self._subs_radius=radius

    def set_abso_radius(self,radius=None):
        if self.atoms_radius is not None:
            radius=[]
            for atom in self._abso_symbols:
                bool=False
                for i in self.atoms_radius:
                    if i.split()[0]==atom:
                        radius.append(float(i.split()[1]))
                        bool=True
                        break
                if not bool:
                    raise ValueError("No "+i.split()[0]+" atom's radius in pso_init file.")
        self._abso_radius=radius

    def set_constraints(self,constraints=None):
        if constraints is None:
            constraints=[[1,1,1]]*len(self.get_subs_symbols())
        constraints=np.array(constraints)
        constraints=constraints.reshape(-1,3)
        self._constraints=constraints

    def set_vac(self,vac=[0.,0.,10.]):
        subs=self.get_subs_positions(cord_mod='d')
        abso=self.get_abso_positions(cord_mod='d')
        lc=self.get_lattice_constant()
        cell=self.get_cell()
        for n,dd in enumerate(vac):
            if dd:
                if len(subs)!=0:
                    tmp=np.append(subs[:,n],abso[:,n])
                else:
                    tmp=abso[:,n]
                dmax=np.max(tmp)
                dmin=np.min(tmp)
                ll=lc*np.linalg.norm(cell[n])
                new_ll=(dmax-dmin)*ll+dd
                cell[n]=new_ll*cell[n]/ll
                self.set_cell(cell)
                for m,i in enumerate(subs):
                    i[n]=((i[n]-dmin)*ll+dd/2.)/new_ll
                    subs[m]=i
                for m,i in enumerate(abso):
                    i[n]=((i[n]-dmin)*ll+dd/2.)/new_ll
                    abso[m]=i
        self.set_subs_positions(subs,cord_mod='d')
        self.set_abso_positions(abso,cord_mod='d')

    def get_constraints(self):
        return self._constraints.copy()

    def get_subs_radius(self):
        return self._subs_radius

    def get_abso_radius(self):
        return copy.deepcopy(self._abso_radius)

    def set_atoms_velo(self,velo=None):
        velo=np.array(velo)
        velo=velo.reshape(-1,3)
        self._atoms_velo=velo

    def get_atoms_velo(self):
        if hasattr(self,'_atoms_velo'):
            return self._atoms_velo.copy()
        else:
            raise AttributeError('atoms do not have velo attribute!')

    def set_atoms_energy(self,energy=None):
        self._atoms_energy=energy

    def get_atoms_energy(self):
        if hasattr(self,'_atoms_energy'):
            return self._atoms_energy
        else:
            raise AttributeError('atoms do not have velo attribute!')

    def set_atoms_relaxpos(self,relaxpos=None):
        relaxpos=np.array(relaxpos)
        relaxpos=relaxpos.reshape(-1,3)
        self._atoms_relaxpos=relaxpos

    def get_atoms_relaxpos(self):
        "return relaxed positions of abso atoms."
        if hasattr(self,'_atoms_relaxpos'):
            return self._atoms_relaxpos
        else:
            raise AttributeError('atoms do not have velo attribute!')

    def get_zmax(self):
        "return the highest atoms' location"
        abso_positions=self.get_abso_positions(cord_mod='d')
        tmp=[i[2] for i in abso_positions]
        tmp=np.array(tmp)
        zmax=tmp.max()
        return zmax

    def get_cmax(self):
        "return the surface location in c vector"
        subs_positions=self.get_subs_positions(cord_mod='d')
        tmp=[i[2] for i in subs_positions]
        tmp=np.array(tmp)
        cmax=tmp.max()
        return cmax

    def get_cmin(self):
        "return the surface bottom location in c vector"
        subs_positions=self.get_subs_positions(cord_mod='d')
        tmp=[i[2] for i in subs_positions]
        tmp=np.array(tmp)
        cmin=tmp.min()
        return cmin

    def get_distance(self,atom_1,atom_2):
        "calculate the distance of two atoms in a supercell"
        dist=[]
        pbc=self.get_pbc()
        a=[]
        for i in range(3):
            if pbc[i]:a.append([-1,0,1])
            else:a.append([0])
        for i1 in a[0]:
            for i2 in a[1]:
                for i3 in a[2]:
                    atom_2i=atom_2+i1*self._cell[0,:]+i2*self._cell[1,:]+i3*self._cell[2,:]
                    dist.append(atom_1-atom_2i)
        length=[np.linalg.norm(i)*self.get_lattice_constant() for i in dist]
        length=np.array(length)
        return length.min(),dist[length.argmin()]

    def cart_direct(self,atom):
        "transform the cartesian coordinates to the direct(fractional) coordinates"
        cell_1=np.linalg.inv(self.get_cell())
        atom=np.array(atom)
        atom_direct = np.dot(atom, cell_1)
        return atom_direct

    def direct_cart(self,atom):
        "transform the direct coordinates to the cartesian coordinates"
        atom=np.array(atom)
        atom_cart = np.dot(atom, self.get_cell())
        return atom_cart

    def move_atom(self,atom):
        "return the same atom as the given one in the unit cell"
        atom_direct=self.cart_direct(atom)
        pbc=self.get_pbc()
        for n,cord in enumerate(atom_direct):
            if pbc[n]:
                count=0
                while count<=1000:
                    if cord-count>=0 and cord-count<=1:
                        atom_direct[n]=cord-count
                        break
                    if cord+count>=0 and cord+count<=1:
                        atom_direct[n]=cord+count
                        break
                    count+=1
        atom=np.dot(atom_direct,self._cell)
        return atom

    def move_atom2(self,atom):
        "return the same atom as the given one in the unit cell"
        atom_direct=self.cart_direct(atom)
        pbc=self.get_pbc()
        for n,cord in enumerate(atom_direct):
            if pbc[n]:
                count=0
                while count<=1000:
                    if cord-count>=0 and cord-count<=1:
                        atom_direct[n]=cord-count
                        break
                    if cord+count>=0 and cord+count<=1:
                        atom_direct[n]=cord+count
                        break
                    count+=1
        atom=np.dot(atom_direct,self._cell)
        return atom
        
    def copy(self):
        'return a copy of the current object'
        abso_symbols=self.get_abso_symbols()
        subs_symbols=self.get_subs_symbols()
        name=self.get_system_name()
        abso_positions=self.get_abso_positions()
        subs_positions=self.get_subs_positions()
        abso_masses=self.get_abso_masses()
        subs_masses=self.get_subs_masses()
        abso_radius=self.get_abso_radius()
        subs_radius=self.get_subs_radius()
        atoms_velo=self.get_atoms_velo()
        cell=self.get_cell()
        pbc=self.get_pbc()
        lattice_constant=self.get_lattice_constant()
        constraints=self.get_constraints()
        a=PsoAtoms(abso_symbols=abso_symbols,subs_symbols=subs_symbols,name=name,
                   abso_positions=abso_positions,subs_positions=subs_positions,
                   abso_radius=abso_radius,subs_radius=subs_radius,
                   abso_masses=abso_masses,subs_masses=subs_masses,cell=cell,
                   lattice_constant=lattice_constant,
                   pbc=pbc,constraints=constraints)
        a.set_atoms_velo(atoms_velo)
        return a

def symbols_numbers(symbols):
    "calculate the numbes of different atoms from the input symbols"
    if isinstance(symbols,str):
        tmp='1'
        a=[]
        symbols+='E'
        for n,char in enumerate(symbols[1:]):
            if char.isdigit():
                if symbols[n].isdigit():tmp=tmp+char
                else:tmp=char
            else:
                a.append(int(tmp))
                tmp='1'
    elif isinstance(symbols,list):
        if symbols==[]:
            a=[]
            return a
        else:a=[1]
        for n,char in enumerate(symbols[1:]):
            if char in symbols[:(n+1)]:
                a[-1]=a[-1]+1
            else:a.append(1)
    else:
        raise ValueError('The format of input symbols is not right.')
    return a

def get_chemical_formula(symbols):
    "return the chemical formula of the given symbols,such as 'H2O'"
    if isinstance(symbols,str):
        return str
    if isinstance(symbols,list):
        if symbols==[]:
            a=''
        else:
            a=symbols[0]
            tmp=1
            for n,char in enumerate(symbols[1:]):
                if char==symbols[n]:
                    tmp+=1
                else:
                    a=a+str(tmp)
                    tmp=1
                    a=a+char
    return a

def atoms_elements(symbols):
        "convert list of the atoms' symbols to the elements' list"
        if isinstance(symbols,str):
            a=[char for char in symbols if char.isalpha()]
        elif isinstance(symbols,list):
            if symbols==[]:a=[]
            else:
                a=[symbols[0]]
                for n,char in enumerate(symbols[1:]):
                    if char not in symbols[:(n+1)]:
                        a.append(char)
        else:
            raise ValueError('The format of input symbols is not right.')
        return a
