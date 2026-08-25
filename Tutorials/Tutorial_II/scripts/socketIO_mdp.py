"""
MDP_SocketClient — i-PI socket client for DipolePolarizabilityMACE models.

Designed for i-PI replay mode: energy and forces are always zero (replay drives
positions from a trajectory file), while dipole, polarizability, Born effective
charges (BEC), and Raman tensors are computed and sent in the extras payload.

Each quantity is independently enabled at init time.

Usage
-----
    from socketIO_mdp import MDP_SocketClient
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths="model.model",
        device="cpu",
        model_type="DipolePolarizabilityMACE",
    )
    atoms.calc = calc

    client = MDP_SocketClient(
        unixsocket="localhost-12",
        has_dipole=True,
        has_polarizability=True,
        has_bec=True,
        has_raman=True,
    )
    client.run(atoms, use_stress=False)

i-PI extras keys (JSON, one dict per step)
------------------------------------------
    "dipole"                  [3]         e·bohr
    "polarizability"          [9]         bohr³
    "polarizability_shape"    [3, 3]
    "bec"                     [N*9]       dimensionless (e)
    "bec_shape"               [N, 3, 3]
    "raman_tensors"           [N*27]      bohr²
    "raman_tensors_shape"     [N, 9, 3]   N atoms, 9 polarizability components, 3 displacements
    "atomic_charges"          [N]         e (dimensionless)
    "atomic_dipoles"          [N*3]       e·bohr (per-atom dipoles nu_i)
    "atomic_dipoles_shape"    [N, 3]
    "atomic_polarizabilities" [N*9]       bohr³ (per-atom polarizabilities, row-major 3x3)
    "atomic_polarizabilities_shape" [N, 3, 3]

The atomic_* keys are the per-atom pieces of the model (charges, dipoles,
polarizabilities) used by the SFG analysis (sfg_atomic.py); they are
cheaper to evaluate than bec/raman (13N instead of 36N values per frame),
and summing them reproduces the total dipole and polarizability.
"""

import json
import numpy as np
from ase import units
from ase.calculators.calculator import all_changes
from ase.calculators.socketio import SocketClient, SocketClosed

# 1 Debye = 0.393430307 e·bohr  (NIST 2018 CODATA)
_DEBYE_TO_EBOHR = 0.393430307


class MDP_SocketClient(SocketClient):
    """
    Socket client for DipolePolarizabilityMACE in i-PI replay mode.

    Energy and forces sent to i-PI are always zero — replay mode reads
    positions from a trajectory file and ignores them.

    Parameters
    ----------
    has_dipole : bool
        Send total dipole vector in extras["dipole"] (converted to e·bohr).
    has_polarizability : bool
        Send polarizability tensor in extras["polarizability"] (converted to bohr³).
    polarizability_units : str
        Units in which the calculator returns polarizability:
          "A3"    — Å³  (default for MACECalculator; converted to bohr³ for i-PI)
          "bohr3" — bohr³ (no conversion applied)
    has_bec : bool
        Send Born effective charges in extras["bec"] / extras["bec_shape"].
        Shape [N, 3, 3].  Units: dimensionless (e), same in Å or bohr.
        Triggers the dielectric-derivative autograd pass (slower).
    has_raman : bool
        Send Raman susceptibility tensors in extras["raman_tensors"] /
        extras["raman_tensors_shape"].  Shape [N, 9, 3]: N atoms, 9
        polarizability components (α_xx…α_zz flattened), 3 displacement
        directions.  Converted to bohr².
        Triggers the dielectric-derivative autograd pass (slower).
    raman_units : str
        Units in which the calculator returns Raman tensors (dα/dr):
          "A2"    — Å²  (default for MACECalculator; converted to bohr² for i-PI)
          "bohr2" — bohr² (no conversion applied)
    has_atomic : bool
        Send the response-sited per-atom pieces in extras["atomic_charges"] /
        ["atomic_dipoles"] / ["atomic_polarizabilities"] (+ *_shape). Units:
        e, e·bohr (from e·Å like the total dipole), bohr³ (from Å³ like the
        total α). FORWARD-ONLY — does NOT trigger the dielectric-derivative
        pass, so it is cheap to add alongside or instead of bec/raman.
    dipole_units : str
        Units in which the calculator returns dipoles:
          "eang"  — e·Å  (default for MACECalculator)
          "debye" — Debye
          "ebohr" — e·bohr (no conversion applied)
    All remaining args/kwargs are forwarded to SocketClient.
    """

    def __init__(
        self,
        *args,
        has_dipole=True,
        has_polarizability=True,
        has_bec=False,
        has_raman=False,
        has_atomic=False,
        dipole_units="eang",
        polarizability_units="A3",
        raman_units="A2",
        **kwargs,
    ):
        self.has_dipole = has_dipole
        self.has_polarizability = has_polarizability
        self.has_bec = has_bec
        self.has_raman = has_raman
        self.has_atomic = has_atomic
        if dipole_units not in ("eang", "debye", "ebohr"):
            raise ValueError(
                f"dipole_units must be 'eang', 'debye', or 'ebohr', got {dipole_units!r}"
            )
        self.dipole_units = dipole_units
        if polarizability_units not in ("A3", "bohr3"):
            raise ValueError(
                f"polarizability_units must be 'A3' or 'bohr3', got {polarizability_units!r}"
            )
        self.polarizability_units = polarizability_units
        if raman_units not in ("A2", "bohr2"):
            raise ValueError(
                f"raman_units must be 'A2' or 'bohr2', got {raman_units!r}"
            )
        self.raman_units = raman_units
        super().__init__(*args, **kwargs)

    def _to_bohr3(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if self.polarizability_units == "A3":
            return arr / (units.Bohr ** 3)
        return arr

    def _to_bohr2(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if self.raman_units == "A2":
            return arr / (units.Bohr ** 2)
        return arr

    def _to_ebohr(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if self.dipole_units == "eang":
            return arr / units.Bohr
        if self.dipole_units == "debye":
            return arr * _DEBYE_TO_EBOHR
        return arr

    def _build_morebytes(self, results):
        extras = {}

        if self.has_dipole:
            d = results.get("dipole")
            if d is not None:
                extras["dipole"] = self._to_ebohr(d).tolist()

        if self.has_polarizability:
            alpha = results.get("polarizability")
            if alpha is not None:
                alpha = self._to_bohr3(np.asarray(alpha, dtype=np.float64))
                extras["polarizability"] = alpha.flatten().tolist()
                extras["polarizability_shape"] = list(alpha.shape)

        if self.has_bec:
            bec = results.get("bec")
            if bec is not None:
                bec = np.asarray(bec, dtype=np.float64)
                extras["bec"] = bec.flatten().tolist()
                extras["bec_shape"] = list(bec.shape)

        if self.has_raman:
            raman = results.get("raman_tensors")
            if raman is not None:
                raman = self._to_bohr2(np.asarray(raman, dtype=np.float64).reshape(raman.shape[0], 9, 3))
                extras["raman_tensors"] = raman.flatten().tolist()
                extras["raman_tensors_shape"] = list(raman.shape)  # [N, 9, 3]

        if self.has_atomic:
            q = results.get("charges")
            if q is not None:
                # atomic charges: dimensionless (e), same in A or bohr
                extras["atomic_charges"] = np.asarray(q, dtype=np.float64).tolist()
            nu = results.get("atomic_dipoles")
            if nu is not None:
                # per-atom dipoles: e*A -> e*bohr (same conversion as the total)
                nu = self._to_ebohr(np.asarray(nu, dtype=np.float64))
                extras["atomic_dipoles"] = nu.flatten().tolist()
                extras["atomic_dipoles_shape"] = list(nu.shape)  # [N, 3]
            ap = results.get("atomic_polarizabilities")
            if ap is not None:
                # per-atom polarizabilities: A^3 -> bohr^3 (as the total alpha)
                ap = self._to_bohr3(np.asarray(ap, dtype=np.float64))
                extras["atomic_polarizabilities"] = ap.flatten().tolist()
                extras["atomic_polarizabilities_shape"] = list(ap.shape)  # [N, 3, 3]

        if not extras:
            return np.zeros(1, dtype=np.byte)
        return np.frombuffer(json.dumps(extras).encode("utf-8"), dtype=np.byte)

    def irun_rank0(self, atoms, use_stress=False):
        n_atoms = len(atoms)
        # Replay mode: energy and forces are always zero.
        energy = 0.0
        forces = np.zeros((n_atoms, 3))
        virial = np.zeros((3, 3))
        morebytes = np.zeros(1, dtype=np.byte)

        # Properties to request from the calculator each step.
        # BEC and Raman need compute_dielectric_derivatives=True (slower).
        props = []
        if self.has_dipole:
            props.append("dipole")
        if self.has_polarizability:
            props.append("polarizability")
        if self.has_bec:
            props.append("bec")
        if self.has_raman:
            props.append("raman_tensors")
        if self.has_atomic:
            # forward-only per-atom pieces; requesting them does NOT trigger
            # the dielectric-derivative branch (keyed on bec/raman_tensors)
            props += ["charges", "atomic_dipoles", "atomic_polarizabilities"]

        try:
            while True:
                try:
                    msg = self.protocol.recvmsg()
                except SocketClosed:
                    msg = "EXIT"

                if msg == "EXIT":
                    self.comm.broadcast(np.ones(1, bool), 0)
                    return
                elif msg == "STATUS":
                    self.protocol.sendmsg(self.state)
                elif msg == "POSDATA":
                    assert self.state == "READY"
                    cell, _icell, positions = self.protocol.recvposdata()
                    atoms.cell[:] = cell
                    atoms.positions[:] = positions
                    self.comm.broadcast(np.zeros(1, bool), 0)

                    if props:
                        atoms.calc.calculate(
                            atoms,
                            properties=props,
                            system_changes=all_changes,
                        )
                    morebytes = self._build_morebytes(atoms.calc.results)
                    self.state = "HAVEDATA"
                    yield
                elif msg == "GETFORCE":
                    assert self.state == "HAVEDATA", self.state
                    self.protocol.sendforce(energy, forces, virial, morebytes)
                    self.state = "NEEDINIT"
                elif msg == "INIT":
                    assert self.state == "NEEDINIT"
                    bead_index, initbytes = self.protocol.recvinit()
                    self.bead_index = bead_index
                    self.bead_initbytes = initbytes
                    self.state = "READY"
                else:
                    raise KeyError("Bad message", msg)
        finally:
            self.close()
