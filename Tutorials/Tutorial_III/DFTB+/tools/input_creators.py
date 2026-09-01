import numpy as np

def inp_file(mxll_dimensions=None, mxll_boundaries=None, mxll_npml=None,
                    mxll_2D_mode=None, mxll_n_media=None, mxll_n_q_groups=None,
                    mxll_n_detectors=None, mpi_procs_per_axis=None, mxll_box_size=None,
                    mxll_total_time=None, mxll_dr=None, mxll_dt=None,
                    mxll_dt_q=None, mxll_dt_det_print=None, mxll_dt_q_print=None,
                    mxll_density_factor=None, help=False):

    if help:
            print("The following parameters can be set in the 'inp' file for the Mxll simulation:")
            print("")
            print(
                   "mxll_dimensions       -> Mxll dimensions: 1, 2, 3\n"
                   "mxll_boundaries(3)    -> Mxll boundary conditions: \"close\", \"periodic\", \"cpml\", \"none\"\n"
                   "mxll_npml             -> Mxll PML thickness in number of grid points.\n"
                   "mxll_2D_mode          -> Mxll 2D mode: \"TMz\", \"TEz\", \"full\", \"none\"\n"
                   "mxll_n_media          -> Number of media to check in the \"medium_xxxx.in\" file.\n"
                   "mxll_n_q_groups       -> Number of quantum groups to check in the \"mol_group_xxxxxxx.in\" file.\n"
                   "mxll_n_detectors      -> Number of detectors to check in the \"detectors.in\" file.\n"
                   "mpi_procs_per_axis(3) -> Number of MPI processes per axis, when running in parallel.\n"
                   "mxll_box_size(3)      -> Size of each dimension in nm for the Mxll simulation box.\n" 
                   "                         The origin is always moved at the center of the box.\n"
                   "mxll_total_time       -> Total simulation time in fs.\n"
                   "mxll_dr               -> Spatial step in nm.\n"
                   "mxll_dt               -> Time step in fs. It will be automatically reduced if it is larger \n"
                   "                         than the stability limit for the Maxwell solver (dr/(2*c)).\n"
                   "mxll_dt_q             -> Quantum time step in atomic units.\n"
                   "mxll_dt_det_print     -> Printing time step for the detectors in fs.\n"
                   "mxll_dt_q_print       -> Printing time step for the quantum systems in fs.\n"
                   "mxll_density_factor   -> Density factor in units of 1/nm^3.\n")
            return

    inp_file = open("inp", "w")

    inp_file.write("&MARADO\n")

    if mxll_dimensions is not None:
        inp_file.write("mxll_dimensions = "+str(int(mxll_dimensions))+"\n")

    if mxll_boundaries is not None:
        if len(mxll_boundaries) == 1:
            inp_file.write("mxll_boundaries = \""+ mxll_boundaries[0] + "\"\n")
        elif len(mxll_boundaries) == 2:
            inp_file.write("mxll_boundaries = \""+ mxll_boundaries[0] + "\", \"" + mxll_boundaries[1] + "\"\n")
        elif len(mxll_boundaries) == 3:
            inp_file.write("mxll_boundaries = \""+ mxll_boundaries[0] + "\", \"" + mxll_boundaries[1] + "\", \"" + mxll_boundaries[2] + "\"\n")

    if mxll_npml is not None:
        inp_file.write("mxll_npml = "+str(int(mxll_npml))+"\n")

    if mxll_2D_mode is not None:
        inp_file.write("mxll_2D_mode = \""+ mxll_2D_mode + "\" \n")

    if mxll_n_media is not None:
        inp_file.write("mxll_n_media = "+str(int(mxll_n_media))+"\n")

    if mxll_n_q_groups is not None:
        inp_file.write("mxll_n_q_groups = "+str(int(mxll_n_q_groups))+"\n")

    if mxll_n_detectors is not None:
        inp_file.write("mxll_n_detectors = "+str(int(mxll_n_detectors))+"\n")

    if mpi_procs_per_axis is not None:
        if len(mpi_procs_per_axis) == 1:
            inp_file.write("mpi_procs_per_axis = "+str(int(mpi_procs_per_axis[0]))+"\n")
        elif len(mpi_procs_per_axis) == 2:
            inp_file.write("mpi_procs_per_axis = "+str(int(mpi_procs_per_axis[0]))+", "+str(int(mpi_procs_per_axis[1]))+"\n")
        elif len(mpi_procs_per_axis) == 3:
            inp_file.write("mpi_procs_per_axis = "+str(int(mpi_procs_per_axis[0]))+", "+str(int(mpi_procs_per_axis[1]))+", "+str(int(mpi_procs_per_axis[2]))+"\n")

    if mxll_box_size is not None:
        if len(mxll_box_size) == 1:
            inp_file.write("mxll_box_size = "+str(float(mxll_box_size[0]))+"\n")
        elif len(mxll_box_size) == 2:
            inp_file.write("mxll_box_size = "+str(float(mxll_box_size[0]))+", "+str(float(mxll_box_size[1]))+"\n")
        elif len(mxll_box_size) == 3:
            inp_file.write("mxll_box_size = "+str(float(mxll_box_size[0]))+", "+str(float(mxll_box_size[1]))+", "+str(float(mxll_box_size[2]))+"\n")

    if mxll_total_time is not None:
        inp_file.write("mxll_total_time = "+str(float(mxll_total_time))+"\n")

    if mxll_dr is not None:
        inp_file.write("mxll_dr = "+str(float(mxll_dr))+"\n")

    if mxll_dt is not None:
        inp_file.write("mxll_dt = "+str(float(mxll_dt))+"\n")

    if mxll_dt_q is not None:
        inp_file.write("mxll_dt_q = "+str(float(mxll_dt_q))+"\n")

    if mxll_dt_det_print is not None:
        inp_file.write("mxll_dt_det_print = "+str(float(mxll_dt_det_print))+"\n")

    if mxll_dt_q_print is not None:
        inp_file.write("mxll_dt_q_print = "+str(float(mxll_dt_q_print))+"\n")

    if mxll_density_factor is not None:
        inp_file.write("mxll_density_factor = "+str(float(mxll_density_factor))+"\n")

    inp_file.write("/\n")

    inp_file.close()

    return

########################################################################################################################################################################################################################

def detectors_file(detector_type=None, detected_field=None, 
                        x_min=0.0, x_max=0.0, 
                        y_min=0.0, y_max=0.0, 
                        z_min=0.0, z_max=0.0,
                        file_exist=False, help=False):
    
    if help:
        print("The following parameters can be set in the 'detectors.in' file for the Mxll simulation:")
        print("")
        print(
            "detector_type   -> Type of detector: \"point\", \"line_x\", \"line_y\", \"line_z\",\n"
            "                                     \"plane_xy\", \"plane_yz\", \"plane_xz\", \"volume\".\n"
            "detected_field  -> Detected field: \"Ex\", \"Ey\", \"Ez\", \"Hx\", \"Hy\", \"Hz\".\n"
            "x_min, x_max    -> Minimum and maximum x coordinates of the detector in nm.\n"
            "y_min, y_max    -> Minimum and maximum y coordinates of the detector in nm.\n"
            "z_min, z_max    -> Minimum and maximum z coordinates of the detector in nm.\n"
            "                   Use x_min, y_min and z_min for the position of a point, line center or plane center.\n"
            "file_exist      -> Whether the detectors.in file already exists. If True, the new detector \n"
            "                   will be added at the end of the file. Default: False.\n")
        return        

    if detector_type is None or detected_field is None:
        print("Error: detector_type and detected_field must be specified.")
        return
    
    if file_exist:
        mode = "a"
    else:
        mode = "w"

    det_file = open("detectors.in", mode)


    det_file.write("\"" + detector_type + "\"" +"   \""+ detected_field +"\"   "+
                   str(x_min)+" "+str(x_max)+" "+str(y_min)+" "+str(y_max)+" "+str(z_min)+" "+str(z_max)+"\n")

    det_file.close()

    return

########################################################################################################################################################################################################################

def medium_file(file_number=1, medium_type="dielectric", relative_permitivity=1.0,
                      omega=None, gamma=None, material=None, coordinates=None, help=False):
    
    if help:
        print("The following parameters can be set in the 'medium_xxxx.in' file for the Mxll simulation:")
        print("")
        print(
            "file_number          -> Number of the medium file, starting from 1. It should be consistent with \n"
            "                        the number of media specified in the 'inp' file.\n"
            "medium_type          -> Type of medium: \"dielectric\", \"drude\", \"lorentz-drude\". Default: \"dielectric\".\n"
            "relative_permitivity -> Relative permitivity for a dielectric medium and drude. Default: 2.0.\n"
            "omega                -> Resonance frequency in eV for a drude medium.\n"
            "gamma                -> Damping factor in eV for a drude medium.\n"
            "material             -> Material name for drude-lorentz medium: 'Ag', 'Au' and 'Al' for the moment.\n"
            "coordinates          -> Array of 1,2 or 3 elements, with the x, y and z coordinates of every grid point with the medium.\n")
        return

    if coordinates is None:
        print("Error: coordinates must be specified.")
        return

    # Accept either a single coordinate vector (e.g. [x, y, z]) or
    # a list/array of coordinate vectors (e.g. [[x1, y1, z1], [x2, y2, z2]]).
    coord_array = np.asarray(coordinates)
    coord_array = np.squeeze(coord_array)

    if coord_array.ndim == 0:
        # Single scalar coordinate.
        coord_array = coord_array.reshape(1, 1)
    elif coord_array.ndim == 1:
        # Single point provided.
        coord_array = coord_array.reshape(1, -1)
    elif coord_array.ndim != 2:
        print("Error: coordinates should be a 1D or 2D array-like object.")
        return

    # Accept transposed point arrays such as shape (dim, n_points).
    if coord_array.ndim == 2:
        if coord_array.shape[1] not in (1, 2, 3) and coord_array.shape[0] in (1, 2, 3):
            coord_array = coord_array.T

    n = coord_array.shape[0]
    dimensions = coord_array.shape[1]
    print("Your medium occupies " + str(n) + " grid points in " + str(dimensions) + " dimensions.")

    if dimensions < 1 or dimensions > 3:
        print("Error: coordinates should have 1, 2 or 3 elements for the x, y and z coordinates of the medium grid points.")
        return

    id_number = str(file_number).zfill(4)

    medium_file = open("medium_" + id_number + ".in", "w")

    if medium_type not in ["dielectric", "drude", "lorentz-drude"]:
        print("Error: medium_type should be one of the following: \"dielectric\", \"drude\", \"lorentz-drude\".")
        return

    if medium_type == "dielectric":
        medium_file.write("\"" + medium_type + "\"\n")
        medium_file.write(str(float(relative_permitivity)) + "\n")

    if medium_type == "drude":
        if omega is None or gamma is None:
            print("Error: omega and gamma must be specified for a drude medium.")
            return
        medium_file.write("\"" + medium_type + "\"\n")
        medium_file.write(str(omega) + " " + str(gamma) + "  " + str(relative_permitivity) + "\n")

    if medium_type == "lorentz-drude":
        if material is None:
            print("Error: material must be specified for a lorentz-drude medium.")
            return
        medium_file.write("\"" + medium_type + "\"\n")
        medium_file.write("\"" + material + "\"\n")

    if dimensions == 1:
        for i in range(n):
            medium_file.write(str(coord_array[i][0]) + "\n")

    if dimensions == 2:
        for i in range(n):
            medium_file.write(str(coord_array[i][0]) + " " + str(coord_array[i][1]) + "\n")

    if dimensions == 3:
        for i in range(n):
            medium_file.write(str(coord_array[i][0]) + " " + str(coord_array[i][1]) + " " + str(coord_array[i][2]) + "\n")

    return

########################################################################################################################################################################################################################

def mol_group_file(file_number=1, number_of_q_systems=None, group_type="material", q_system_type="dftb", print_options="print_none",
                         mol_file_list=None, coordinates=None, print_list=None, help=False):
    
    if help:
        print("The following parameters can be set in the 'mol_group_xxxxxxx.in' file for the Mxll simulation:")
        print("")
        print(
            "file_number         -> Number of the molecule group file, starting from 1. It should be consistent with \n"
            "                       the number of quantum groups specified in the 'inp' file.\n"
            "number_of_q_systems -> Number of quantum systems in the group.\n"
            "group_type          -> Type of quantum group: \"material\" for a quantum group with quantum systems placed one next to the other\n"
            "                       or \"single\" for single quantum systems. Default: \"material\".\n"
            "q_system_type       -> Type of quantum systems: \"dftb\" is the only option for the moment. Default: \"dftb\".\n"
            "print_options       -> Printing options for the quantum system: \"print_none\", \"print_all\", \"print_selected\". Default: \"print_none\".\n"
            "molecule_file_list  -> List of strings with the filenumber of the file molecule_xxxxxxx.in for each quantum system.\n"
            "coordinates         -> Array of 1,2 or 3 elements, with the x, y and z coordinates of every grid point with the quantum systems.\n"
            "print_list          -> List of integers with the index of the quantum systems to print when print_options is \"print_selected\".\n"
            "   \n"
            "REMEMBER: You need the \".skf\" files with all the atomic interactions for the atoms in your molecule, in the same directory as your inp file.")

        return    

    if number_of_q_systems is None or mol_file_list is None or coordinates is None:
        print("Error: number_of_q_systems, mol_file_list and coordinates must be specified.")
        return
    
    if print_options == "print_selected" and print_list is None:
        print("Error: print_list must be specified when print_options is \"print_selected\".")
        return

    n   = len(coordinates)

    if n != number_of_q_systems:
        print("Error: The number of coordinates should be equal to the number of quantum systems.")
        return
    
    if number_of_q_systems != len(mol_file_list):
        print("Error: The number of elements in mol_file_list should be equal to the number of quantum systems.")
        return

    if isinstance(coordinates[0], (float)):
        dim = 1
    else:
        dim = len(coordinates[0])

    if dim < 1 or dim > 3:
        print("Error: coordinates should have 1, 2 or 3 elements for the x, y and z coordinates of the quantum system position.")
        return

    mol_group_file = open("mol_group_" + str(file_number).zfill(7) + ".in", "w")

    mol_group_file.write("\"" + group_type + "\"   \"" + q_system_type + "\"   " + str(number_of_q_systems) + "   \"" + print_options + "\"\n")

    if dim == 1:
        for i in range(n):
            if print_options == "print_selected":
                if i in print_list:
                    mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i]) + "   \"print_on\"\n")
                else:
                    mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i]) + "   \"print_off\"\n")
            else:
                mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i]) + "\n")

    if dim == 2:
        for i in range(n):
            if print_options == "print_selected":
                if i in print_list:
                    mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i][0]) + " " + str(coordinates[i][1]) + "   \"print_on\"\n")
                else:
                    mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i][0]) + " " + str(coordinates[i][1]) + "   \"print_off\"\n")
            else:
                mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i][0]) + " " + str(coordinates[i][1]) + "\n")

    if dim == 3:
        for i in range(n):
            if print_options == "print_selected":
                if i in print_list:
                    mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i][0]) + " " + str(coordinates[i][1]) + " " + str(coordinates[i][2]) + "   \"print_on\"\n")
                else:
                    mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i][0]) + " " + str(coordinates[i][1]) + " " + str(coordinates[i][2]) + "   \"print_off\"\n")
            else:
                mol_group_file.write(str(i+1)+"   "+str(mol_file_list[i]) + "   " + str(coordinates[i][0]) + " " + str(coordinates[i][1]) + " " + str(coordinates[i][2]) + "\n")

    return

########################################################################################################################################################################################################################

def dftb_molecule_file(file_number=1, n_atoms=None, n_atom_types=None,
                              dynamics_type=None, print_coordinates=False, euler_steps=500,
                              atom_types_list=None, max_angular_momentum_list=None, scc_tolerance=1e-6,
                              atoms_list=None, atom_coordinates=None, help=False):
    
    if help:
        print("The following parameters can be set in the 'molecule_xxxxxxx.in' file for the Mxll simulation:")
        print("")
        print(
            "file_number               -> Number of the molecule file, starting from 1. It should be consistent with \n"
            "                             the file list specified in the 'mol_group_xxxxxxx.in' file.\n"
            "n_atoms                   -> Number of atoms in the molecule.\n"
            "n_atom_types              -> Number of different atom types in the molecule.\n"
            "dynamics_type             -> Type of dynamics: \"electrons\", \"ehrenfest\" or \"born-oppenheimer\".\n"
            "print_coordinates         -> Whether to print the coordinates of the atoms at every time step: True or False. Default: False.\n"
            "euler_steps               -> Number of time steps before to make one Euler step to reduce numerical noise. Default: 500.\n"
            "atom_types_list           -> List of strings with the name of the atoms in atom_list with no repetition.\n"
            "max_angular_momentum_list -> Orbital name of the maximum angular momentum for each atom type in atom_types_list.\n"
            "scc_tolerance             -> Tolerance for the self-consistent charge calculation. Default: 1e-6.\n"
            "atoms_list                -> List of atoms in the molecule with dimension (n_atoms).\n"
            "atom_coordinates          -> Array of shape (n_atoms, 3) with the x, y and z coordinates of each atom in Angstroms.\n")

        return
    
    if n_atoms is None or n_atom_types is None or dynamics_type is None or atom_types_list is None \
    or max_angular_momentum_list is None or atoms_list is None or atom_coordinates is None:
        print("Error: n_atoms, n_atom_types, dynamics_type, atom_types_list, max_angular_momentum_list, atoms_list and atom_coordinates must be specified.")

        return
    
    molecule_file = open("molecule_" + str(file_number).zfill(7) + ".in", "w")

    molecule_file.write(str(n_atoms) + "   " + str(n_atom_types) + "\n")

    if print_coordinates:
        print_coordinates_str = "print_coordinates_on"
    else:        
        print_coordinates_str = "print_coordinates_off"

    molecule_file.write("\"" + dynamics_type + "\"   \"" + print_coordinates_str + "\"    " + str(euler_steps) + "\n")
    for i in range(n_atom_types):
        molecule_file.write("  \"" + atom_types_list[i] + "\"")
    molecule_file.write("\n")

    for i in range(n_atom_types):
        molecule_file.write("  \"" + max_angular_momentum_list[i] + "\"")
    molecule_file.write("\n")

    molecule_file.write(str(float(scc_tolerance)) + "\n")

    molecule_file.write("  \n")

    for i in range(n_atoms):
        molecule_file.write("  \"" + atoms_list[i] + "\"   " + str(atom_coordinates[i][0]) + " " +
                            str(atom_coordinates[i][1]) + " " + str(atom_coordinates[i][2]) + "\n")

    molecule_file.close()

    return

########################################################################################################################################################################################################################

def sources_file(source_type=None, polarization=None, field_amp=None, frequency=None, t0=None, tau=None, t_init=None, t_end=None, phase=None,
                r0=None, radius=None, phi=None, theta=None, psi=None, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None, 
                file_exists=False, help=False):
    
    if help:
        print("The following parameters can be set in the 'sources.in' file for the Mxll simulation:")
        print("")
        print(
            "source_type  -> Type of source: \"point\", \"plane_wave\".\n"
            "polarization -> Polarization of the \"point\" source: \"x\", \"y\" or \"z\" \n"
            "field_amp    -> Field amplitude in atomic units for the sources.\n"
            "frequency    -> Source frequency in eV.\n"
            "t0           -> Time origin of the source in fs.\n"
            "tau          -> Full width at half maximum (FWHM) of the pulse in fs.\n"
            "t_init       -> Initial time of the source in fs. The source will be turned on at this time step.\n"
            "t_end        -> Final time of the source in fs. The source will be turned off after this time step.\n"
            "phase        -> Phase of the source in degrees.\n"
            "r0(3)        -> Position of the center of the \"point\" source in nm. \n"
            "radius       -> FWHM of the gaussian spatial profile of the \"point\" source in nm.\n" 
            "phi          -> Angle between the x-axis and the projection of the wavevector of the \"plane_wave\" source on the xy-plane in degrees. \n"
            "theta        -> Angle between the z-axis and the wavevector of the \"plane_wave\" source in degrees. \n"
            "psi          -> Rotation angle of the electric field around the wavevector of the \"plane_wave\" source in degrees. \n"
            "                For psi=0, the electric field is parallel to \"k x z\", where k is the wavevector of the plane wave. \n"
            "x_min, x_max -> Minimum and maximum x coordinates of the region limited for the plane wave source in nm. \n"
            "y_min, y_max -> Minimum and maximum y coordinates of the region limited for the plane wave source in nm. \n"
            "z_min, z_max -> Minimum and maximum z coordinates of the region limited for the plane wave source in nm. \n"
            "file_exists  -> Whether the sources.in file already exists. If True, the new source will be added \n"
            "                at the end of the file. Default: False.\n")

        return

    if source_type is None:
        print("Error: source_type must be specified.")
        return
    
    if source_type == "point":
        if polarization is None or field_amp is None or frequency is None or t0 is None or tau is None or t_init is None or t_end \
        is None or phase is None or r0 is None or radius is None:
        
            print("Error: polarization, field_amp, frequency, t0, tau, t_init, t_end, phase, r0 and radius must be specified for a \"point\" source.")
            return

    if source_type == "plane_wave":
        if field_amp is None or frequency is None or t0 is None or tau is None or t_init is None or t_end is None or phase is None \
        or phi is None or theta is None or psi is None or x_min is None or x_max is None or y_min is None or y_max is None:
    
            print("Error: field_amp, frequency, t0, tau, t_init, t_end, phase, phi, theta, psi, x_min, x_max, y_min and y_max \n"
                  "       must be specified for a \"plane_wave\" source.")
            return

    if z_min is not None and z_max is not None and source_type == "plane_wave":
        print("Warning: z_min and z_max will be equal to 0.0 for the \"plane_wave\" source. This is only right for 2D simulations.")
        z_min = 0.0
        z_max = 0.0

    if file_exists:
        mode = "a"
    else:
        mode = "w"

    source_file = open("sources.in", mode)

    if source_type == "point":
        source_file.write("\"" + source_type + "\"   \"" + polarization + "\"   " + str(field_amp) + "   " + str(frequency) + "   " +
                          str(t0) + "   " + str(tau) + "   " + str(r0[0]) + " " + str(r0[1]) + " " + str(r0[2]) + "   " +
                          str(radius) + "   " + str(t_init) + "   " + str(t_end) + "   " + str(phase) + "\n")
        
    if source_type == "plane_wave":
        source_file.write("\"" + source_type + "\"   " + str(field_amp) + "   " + str(phi)+ "   " + str(theta) + "   " + str(psi) + "   " + \
        str(frequency) + "   " + str(t0) + "   " + str(tau) + "   " + str(x_min) + " " + str(x_max) + " " + str(y_min) + " " +  \
        str(y_max) + " " + str(z_min) + " " + str(z_max) + "   " + str(t_init) + "   " + str(t_end) + "   " + str(phase) + "\n")

    source_file.close()

    return

########################################################################################################################################################################################################################
