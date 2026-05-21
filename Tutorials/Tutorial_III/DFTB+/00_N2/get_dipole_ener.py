# Filename: extract_time_dipole.py

input_file = "energy_and_dipole.dat"     # Your input file name
output_file = "time_dipole.dat"  # Output file name

times = []
dipoles = []

with open(input_file, "r") as f:
    current_time = None
    for line in f:
        line = line.strip()
        
        # Look for time lines
        if line.startswith("# Time"):
            # The time value is after the colon
            parts = line.split("=")
            parts2 = parts[1].split("(")
            current_time = float(parts2[0])
        
        # Look for the molecule 1 data line
        elif current_time is not None and line.startswith("1"):
            # Split the columns: N mol, dipole value, energy
            parts = line.split()
            dipole_val = float(parts[2])
            
            # Save time and dipole
            times.append(current_time)
            dipoles.append(dipole_val)
            
            current_time = None  # Reset until next # time block

# Write output file
with open(output_file, "w") as out:
    for t, d in zip(times, dipoles):
        out.write(f"{t:.15e} {d:.15e}\n")

print(f"Extracted {len(times)} entries to '{output_file}'")
