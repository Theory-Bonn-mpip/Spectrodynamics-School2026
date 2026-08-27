import numpy as np
import matplotlib.pyplot as plt

au_to_eV = 27.2114
au_to_fs = 0.02418884

def read_file(filename, col_x, col_y):
    x_array = []
    y_array = []

    infile = open(filename, "r")

    for val in infile:
        read = not any(char == "#" for char in val)
        if read:
            x_array.append(float(val.split()[col_x]))
            y_array.append(float(val.split()[col_y]))

    infile.close()

    x_array = np.array(x_array)
    y_array = np.array(y_array)

    return x_array, y_array

def get_fourier_transform(data, time_step):
    n = len(data)

    # Calculate the frequencies corresponding to the Fourier transform
    freq = np.fft.fftfreq(10*n, d=time_step)

    # Compute the Fourier transform of the data
    ft_data = np.fft.fft(data, n=10*n)

    # Return the frequencies and the Fourier-transformed data
    return freq*2*np.pi, ft_data

def plot_abs_spectrum_1D(only_pulse_dir=None, sample_dir=None,
                         Ex_detector=1, Hy_detector=2,  
                         x_lim=[0, 20], y_lim=None, help=False):

    if help:
        print("This function plots the absorption spectrum of a 1D simulation using OMxRTA.\n"
                "For this, previously run simulations printing the Ex field and Hy field are required.\n"
                "The following parameters can be specified:\n"
            "       only_pulse_dir -> Directory of the simulation with only the pulse (reference simulation). \n"
            "       sample_dir     -> Directory of the simulation with the sample (test simulation). \n"
            "       Ex_detector    -> Index of the Ex detector. Default: 1.\n"
            "       Hy_detector    -> Index of the Hy detector. Default: 2.\n"
            "       x_lim          -> Limits of the x-axis in eV. Default: [0, 20].\n"
            "       y_lim          -> Limits of the y-axis.\n")
        return

    #Check the existence of the file
    if only_pulse_dir is None:
        print("Error: only_pulse_dir must be specified.")
        return
    if sample_dir is None:
        print("Error: sample_dir must be specified.")
        return
    
    filename = only_pulse_dir+"/output_detector_"+str(Ex_detector).zfill(7)+"/point.dat"

    time, E0xL_t = read_file(filename, 0, 1)

    filename = only_pulse_dir+"/output_detector_"+str(Hy_detector).zfill(7)+"/point.dat"
    time, H0yL_t = read_file(filename, 0, 1)

    dt = time[2]-time[1]

    freq, E0xL_w = get_fourier_transform(E0xL_t, dt)
    freq, H0yL_w = get_fourier_transform(H0yL_t, dt)

    filename = sample_dir+"/output_detector_"+str(Ex_detector).zfill(7)+"/point.dat"

    time, ExL_t = read_file(filename, 0, 1)

    filename = sample_dir+"/output_detector_"+str(Hy_detector).zfill(7)+"/point.dat"
    time, HyL_t = read_file(filename, 0, 1)

    dt = time[2]-time[1]

    freq, ExL_w = get_fourier_transform(ExL_t, dt)
    freq, HyL_w = get_fourier_transform(HyL_t, dt)

    Pw = - np.conj(ExL_w-E0xL_w) * (HyL_w-H0yL_w)

    freq = freq*au_to_eV

    fig, ax1 = plt.subplots()

    N = len(freq)
    spec = np.real(Pw[:N//2])
    ax1.plot(freq[:N//2], spec)

    ax1.set_xlim(x_lim[0], x_lim[1])

    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])

    plt.tight_layout()
    plt.show()

    return

def plot_trans_spectrum_1D(directory=None, Ex_detector=1, Hy_detector=2, 
                           x_lim=[0, 20], y_lim=None, help=False):

    if help:
        print("This function plots the transmission spectrum of a 1D simulation using OMxRTA.\n"
                "For this, a previously run simulation printing the Ex field and Hy field is required.\n"
                "The following parameters can be specified:\n"
            "       directory      -> Directory of the simulation. \n"
            "       Ex_detector    -> Index of the Ex detector. Default: 1.\n"
            "       Hy_detector    -> Index of the Hy detector. Default: 2.\n"
            "       x_lim          -> Limits of the x-axis in eV. Default: [0, 20].\n"
            "       y_lim          -> Limits of the y-axis.\n")
        return


    filename = directory+"/output_detector_"+str(Ex_detector).zfill(7)+"/point.dat"

    time, ExL_t = read_file(filename, 0, 1)

    filename = directory+"/output_detector_"+str(Hy_detector).zfill(7)+"/point.dat"
    time, HyL_t = read_file(filename, 0, 1)

    dt = time[2]-time[1]

    freq, ExL_w = get_fourier_transform(ExL_t, dt)
    freq, HyL_w = get_fourier_transform(HyL_t, dt)

    Pw = - np.conj(ExL_w) * (HyL_w)

    freq = freq*au_to_eV

    fig, ax1 = plt.subplots()

    N = len(freq)
    spec = np.real(Pw[:N//2])
    ax1.plot(freq[:N//2], spec)

    ax1.set_xlim(x_lim[0], x_lim[1])

    if y_lim is not None:
        ax1.set_ylim(y_lim[0], y_lim[1])

    plt.tight_layout()
    plt.show()

    return


def _read_energy_and_dipole_file(filename):
    time_values = []
    molecule_data = {}
    current_time = None

    with open(filename, "r") as infile:
        for line in infile:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("# Time"):
                # Header format: "# Time = <value> (a.u.)"
                try:
                    current_time = float(stripped.split("=")[1].split()[0])
                    time_values.append(current_time)
                except (IndexError, ValueError):
                    current_time = None
                continue

            if stripped.startswith("#"):
                continue

            if current_time is None:
                continue

            parts = stripped.split()
            if len(parts) < 5:
                continue

            try:
                mol_id = int(parts[0])
                energy = float(parts[1])
                mu_x = float(parts[2])
                mu_y = float(parts[3])
                mu_z = float(parts[4])
            except ValueError:
                continue

            if mol_id not in molecule_data:
                molecule_data[mol_id] = {
                    "time": [],
                    "energy": [],
                    "mu_x": [],
                    "mu_y": [],
                    "mu_z": [],
                }

            molecule_data[mol_id]["time"].append(current_time)
            molecule_data[mol_id]["energy"].append(energy)
            molecule_data[mol_id]["mu_x"].append(mu_x)
            molecule_data[mol_id]["mu_y"].append(mu_y)
            molecule_data[mol_id]["mu_z"].append(mu_z)

    for mol_id in molecule_data:
        for key in molecule_data[mol_id]:
            molecule_data[mol_id][key] = np.array(molecule_data[mol_id][key])

    return molecule_data


def plot_energy_and_dipole(filename=None, molecule_ids=None, quantity="energy",
                           x_lim=None, y_lim=None, help=False):

    if help:
        print("This function plots a molecular observable from energy_and_dipole.dat.\n"
              "The following parameters can be specified:\n"
              "       filename      -> Path to energy_and_dipole.dat.\n"
              "       molecule_ids  -> List of molecule indices to plot (e.g. [1, 2, 3]).\n"
              "       quantity      -> Observable to plot: 'energy', 'mu_x', 'mu_y', or 'mu_z'.\n"
              "       x_lim         -> Limits of the x-axis in atomic units of time.\n"
              "       y_lim         -> Limits of the y-axis.\n")
        return

    if filename is None:
        print("Error: filename must be specified.")
        return

    if molecule_ids is None or len(molecule_ids) == 0:
        print("Error: molecule_ids must be a non-empty list.")
        return

    valid_quantities = {"energy", "mu_x", "mu_y", "mu_z"}
    if quantity not in valid_quantities:
        print("Error: quantity must be one of: energy, mu_x, mu_y, mu_z.")
        return

    data = _read_energy_and_dipole_file(filename)

    if len(data) == 0:
        print("Error: no molecular data found in file.")
        return

    fig, ax = plt.subplots()
    plotted_any = False

    for mol_id in molecule_ids:
        if mol_id not in data:
            print("Warning: molecule", mol_id, "not found in file and will be skipped.")
            continue

        mol_time = data[mol_id]["time"]
        mol_values = data[mol_id][quantity]

        ax.plot(mol_time*au_to_fs, mol_values, label="mol " + str(mol_id))
        plotted_any = True

    if not plotted_any:
        print("Error: none of the requested molecules were found in file.")
        return

    y_labels = {
        "energy": "Energy (eV)",
        "mu_x": "Dipole_x (a.u.)",
        "mu_y": "Dipole_y (a.u.)",
        "mu_z": "Dipole_z (a.u.)",
    }

    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel(y_labels[quantity])
    ax.legend()

    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])

    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    plt.tight_layout()
    plt.show()

    return

