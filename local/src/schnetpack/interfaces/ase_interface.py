"""
This module provides a ASE calculator class [#ase1]_ for SchNetPack models, as
well as a general Interface to all ASE calculation methods, such as geometry
optimisation, normal mode computation and molecular dynamics simulations.

References
----------
.. [#ase1] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
    Groves, Hammer, Hargus: The atomic simulation environment -- a Python
    library for working with atoms.
    Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import os
import numpy as np
import torch

from local.src.schnetpack.data.atoms import AtomsConverter
from local.src.schnetpack.utils.spk_utils import DeprecationHelper
from local.src.schnetpack import Properties

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz, write_xyz

from ase.io.vasp import write_vasp  # TODO 用来读取和写入POSCAR文件

from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations

from local.src.schnetpack.md.utils import MDUnits

from local.src.schnetpack.environment import SimpleEnvironmentProvider


class SpkCalculatorError(Exception):
    pass


class SpkCalculator(Calculator):
    """
    ASE calculator for schnetpack machine learning models.

    Args:
        ml_model (schnetpack.AtomisticModel): Trained model for
            calculations
        device (str): select to run calculations on 'cuda' or 'cpu'
        collect_triples (bool): Set to True if angular features are needed,
            for example, while using 'wascf' models
        environment_provider (callable): Provides neighbor lists
        pair_provider (callable): Provides list of neighbor pairs. Only
            required if angular descriptors are used. Default is none.
        **kwargs: Additional arguments for basic ase calculator class
    """

    energy = Properties.energy
    forces = Properties.forces
    stress = Properties.stress
    implemented_properties = [energy, forces, stress]

    def __init__(
        self,
        model,
        device="cpu",
        collect_triples=False,
        environment_provider=SimpleEnvironmentProvider(),
        energy=None,
        forces=None,
        stress=None,
        energy_units="eV",
        forces_units="eV/Angstrom",
        stress_units="eV/Angstrom/Angstrom/Angstrom",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.model = model
        for i in range(1):  # TODO
            self.model[i].to(device)  # TODO

        self.atoms_converter = AtomsConverter(
            environment_provider=environment_provider,
            collect_triples=collect_triples,
            device=device,
        )

        self.model_energy = energy
        self.model_forces = forces
        self.model_stress = stress

        # Convert to ASE internal units
        # MDUnits parses the given energy units and converts them to atomic units as the common denominator.
        # These are then converted to ASE units
        self.energy_units = MDUnits.parse_mdunit(energy_units) * units.Ha
        self.forces_units = MDUnits.parse_mdunit(forces_units) * units.Ha / units.Bohr
        self.stress_units = (
            MDUnits.parse_mdunit(stress_units) * units.Ha / units.Bohr ** 3
        )

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)
            # Convert to schnetpack input format
            model_inputs = self.atoms_converter(atoms)
            # Call model
            tmp_results = []
            model_results = {'energy': 0.0, 'forces': 0.0}
            for i in range(1):  # TODO
                tmp_results.append(self.model[i](model_inputs))
                model_results['energy'] += tmp_results[i]['energy']
                model_results['forces'] += tmp_results[i]['forces']
            model_results['energy'] = model_results['energy'] / 1.0
            model_results['forces'] = model_results['forces'] / 1.0

            """记录预测出来的标准差"""
            # numofatoms = atoms.get_number_of_atoms()
            # force = []
            # for ii in range(4):
            #     tmp_std_force = tmp_results[ii]['forces']
            #     force.append(tmp_std_force.squeeze(0).detach().numpy().reshape(-1))
            # force = np.array(force)
            # force = force.transpose()
            # stddev = np.zeros(3 * numofatoms)
            # for i in range(3 * numofatoms):
            #     stddev[i] = np.std(force[i])
            # std = max(stddev)
            #
            # with open('../../../clustertut/ase_calcs/record_std', 'a') as frec:
            #     frec.write(str(std)+'\n')

            results = {}
            # Convert outputs to calculator format
            if self.model_energy is not None:
                if self.model_energy not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(self.model_energy)
                    )
                energy = model_results[self.model_energy].cpu().data.numpy()
                results[self.energy] = (
                    energy.item() * self.energy_units
                )  # ase calculator should return scalar energy

            if self.model_forces is not None:
                if self.model_forces not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(self.model_forces)
                    )
                forces = model_results[self.model_forces].cpu().data.numpy()
                results[self.forces] = (
                    forces.reshape((len(atoms), 3)) * self.forces_units
                )

            if self.model_stress is not None:
                if atoms.cell.volume <= 0.0:
                    raise SpkCalculatorError(
                        "Cell with 0 volume encountered for stress computation"
                    )

                if self.model_stress not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties! If desired, stress tensor computation can be "
                        "activated via schnetpack.utils.activate_stress_computation "
                        "at ones own risk.".format(self.model_stress)
                    )
                stress = model_results[self.model_stress].cpu().data.numpy()
                results[self.stress] = stress.reshape((3, 3)) * self.stress_units

            self.results = results


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)

    Args:
        molecule_path (str): Path to initial geometry
        ml_model (object): Trained model
        working_dir (str): Path to directory where files should be stored
        device (str): cpu or cuda
    """

    def __init__(
        self,
        molecule_path,
        ml_model,
        working_dir,
        device="cpu",
        energy="energy",
        forces="forces",
        energy_units="eV",
        forces_units="eV/Angstrom",
        environment_provider=SimpleEnvironmentProvider(),
    ):
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = None
        self._load_molecule(molecule_path)

        # Set up calculator
        calculator = SpkCalculator(
            ml_model,
            device=device,
            energy=energy,
            forces=forces,
            energy_units=energy_units,
            forces_units=forces_units,
            environment_provider=environment_provider,
        )
        self.molecule.set_calculator(calculator)

        # Unless initialized, set dynamics to False
        self.dynamics = False

    def _load_molecule(self, molecule_path):
        """
        Load molecule from file (can handle all ase formats).

        Args:
            molecule_path (str): Path to molecular geometry
        """
        file_format = os.path.splitext(molecule_path)[-1]
        if file_format == "xyz":
            self.molecule = read_xyz(molecule_path)
        else:
            self.molecule = read(molecule_path)

    def save_molecule(self, name, file_format="xyz", append=False):
        """
        Save the current molecular geometry.

        Args:
            name (str): Name of save-file.
            file_format (str): Format to store geometry (default xyz).
            append (bool): If set to true, geometry is added to end of file
                (default False).
        """
        molecule_path = os.path.join(self.working_dir, "%s.%s" % (name, file_format))
        if file_format == "xyz":
            # For extended xyz format, plain is needed since ase can not parse
            # the extxyz it writes
            write_xyz(molecule_path, self.molecule, plain=True)
        elif file_format == "poscar":
            write_vasp(molecule_path, self.molecule, direct=True, ignore_constraints=False, vasp5=True)  # TODO
        else:
            write(molecule_path, self.molecule, format=file_format, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule("single_point", file_format="extxyz")

    def init_md(
        self,
        name,
        time_step=0.5,
        temp_init=300,
        temp_bath=None,
        reset=False,
        interval=1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.

        Args:
            name (str): Basic name of logfile and trajectory
            time_step (float): Time step in fs (default=0.5)
            temp_init (float): Initial temperature of the system in K
                (default is 300)
            temp_bath (float): Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset (bool): Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval (int): Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if not self.dynamics or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(
                self.molecule,
                time_step * units.fs,
                temp_bath * units.kB,
                1.0 / (100.0 * units.fs),
            )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "%s.log" % name)
        trajfile = os.path.join(self.working_dir, "%s.traj" % name)
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(
        self, temp_init=300, remove_translation=True, remove_rotation=True
    ):
        """
        Initialize velocities for molecular dynamics

        Args:
            temp_init (float): Initial temperature in Kelvin (default 300)
            remove_translation (bool): Remove translation components of
                velocity (default True)
            remove_rotation (bool): Remove rotation components of velocity
                (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.

        Args:
            steps (int): Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError(
                "Dynamics need to be initialized using the" " 'setup_md' function"
            )

        self.dynamics.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)

        Args:
            fmax (float): Maximum residual force change (default 1.e-2)
            steps (int): Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(
            self.molecule,
            trajectory="%s.traj" % optimize_file,
            restart="%s.pkl" % optimize_file,
        )
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name, file_format='poscar')  # TODO 改成存储为POSCAR文件

    def compute_normal_modes(self, write_jmol=True):
        """
        Use ase calculator to compute numerical frequencies for the molecule

        Args:
            write_jmol (bool): Write frequencies to input file for
                visualization in jmol (default=True)
        """
        freq_file = os.path.join(self.working_dir, "normal_modes")

        # Compute frequencies
        frequencies = Vibrations(self.molecule, name=freq_file)
        frequencies.run()

        # Print a summary
        frequencies.summary()

        # Write jmol file if requested
        if write_jmol:
            frequencies.write_jmol()


MLPotential = DeprecationHelper(SpkCalculator, "MLPotential")
