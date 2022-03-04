import os
import local.src.schnetpack as spk
import torch
from ase import io
from ase.io.trajectory import Trajectory


def read_trajectory():
    os.mkdir('./clustertut/ase_calcs/store')
    clustertut = './clustertut'
    ase_dir = os.path.join(clustertut, 'ase_calcs')
    traj = Trajectory('./clustertut/ase_calcs/optimization.traj')
    count = 0
    for atoms in traj:
        str_count = str(int(count)).zfill(4)
        molecule_path = os.path.join(ase_dir, 'POSCAR_err')
        io.vasp.write_vasp(molecule_path, atoms, direct=True, vasp5=True)
        os.system('cp ' + molecule_path + ' ./clustertut/ase_calcs/store/POSCAR_' + str_count)
        count += 1


def structure_optimization(filename=None, gen=1, natoms=None):
    atoms = io.vasp.read_vasp(filename)
    clustertut = '../../../clustertut'
    best_model_group = []
    g = gen - 1  # modify generation
    for i in range(1):
        str_i = str(int(i+1))
        str_g = str(int(g)).zfill(3)
        tmp = torch.load(os.path.join(clustertut, '../../dataset-ML/N'+str(natoms)+'/model/best_model-'+str_i+'_'+str_g), map_location='cpu')
        best_model_group.append(tmp)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    calculator = spk.interfaces.SpkCalculator(
        model=best_model_group,
        device=device,
        energy='energy',
        forces='forces',
        energy_units='eV',
        forces_units='eV/A'
    )

    atoms.set_calculator(calculator)
    ase_dir = os.path.join(clustertut, 'ase_calcs')

    os.system('rm -rf ../../../clustertut/ase_calcs')  # TODO

    if not os.path.exists(ase_dir):
        os.mkdir(ase_dir)

    # Write a sample molecule
    molecule_path = os.path.join(ase_dir, 'POSCAR_initial')
    io.vasp.write_vasp(molecule_path, atoms, direct=True, vasp5=True)

    cluster_ase = spk.interfaces.AseInterface(
        molecule_path,
        best_model_group,  # TODO
        ase_dir,
        device,
        energy='energy',
        forces='forces',
        energy_units='eV',
        forces_units='eV/A'
    )

    cluster_ase.optimize(fmax=5e-2, steps=1000)
