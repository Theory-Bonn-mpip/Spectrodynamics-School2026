#!/usr/bin/env bash
# Emergency stop for a stuck tutorial run (troubleshooting only -- the
# notebooks never call this).
#
#   bash scripts/stop_all_runs.sh
#
# Use it when a simulation has to be interrupted and the notebook is left
# with orphaned processes: a kernel restart does not stop the i-PI server or
# the MACE client it launched, and the leftover i-PI socket then makes every
# later run fail with "Error opening unix socket ... exists, remove it".
#
# What it does, in order:
#   1. asks every i-PI server to stop *cleanly* by dropping an EXIT file in
#      each exercise folder (i-PI polls for it and shuts down on its own),
#      waits 10 s, then removes the EXIT files again;
#   2. terminates any i-PI server or MACE client still alive (TERM, then
#      KILL for whatever survives);
#   3. deletes the leftover i-PI sockets belonging to this copy.
#
# It is safe on a shared machine. It only touches processes owned by the
# current user, only those running this tutorial's own scripts, and (on
# Linux) only those whose working directory is inside *this* copy of the
# tutorial -- so a second student working from their own checkout under the
# same login is left alone. On systems without /proc it cannot tell copies
# apart and says so before stopping every match.
#
# It does NOT delete simulation output -- use the per-exercise clean.sh, or
# clean_all.sh, for that.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD

# Command-line patterns of the processes this tutorial starts.
PATTERNS=(
    'i-pi .*\.xml'          # the i-PI server (input.xml, input_mdp.xml, ...)
    'run-ase_ex[0-9]'       # the MACE clients of Exercises 1 and 2
    'run-mace-mdp_ex[0-9]'  # the MACE-MDP clients of Exercises 3 and 6
)

# Can we tell one copy of the tutorial from another? On Linux every process
# exposes its working directory as /proc/<pid>/cwd; elsewhere (macOS) we
# cannot, and fall back to stopping every matching process of this user.
if [[ -r "/proc/$$/cwd" ]]; then
    SCOPED=1
else
    SCOPED=0
fi

# True if the process runs inside *this* copy of the tutorial. Two students
# sharing one login keep their own checkouts, so this is what stops us from
# killing somebody else's run.
owned_by_this_copy() {
    local cwd
    cwd=$(readlink "/proc/$1/cwd" 2>/dev/null) || return 1
    [[ "$cwd" == "$ROOT" || "$cwd" == "$ROOT"/* ]]
}

# Print the PIDs of this user's matching processes that belong to this copy,
# never our own subtree.
matching_pids() {
    local pat pid pids=()
    for pat in "${PATTERNS[@]}"; do
        while read -r pid; do
            [[ -z "$pid" || "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
            if [[ $SCOPED -eq 1 ]] && ! owned_by_this_copy "$pid"; then
                continue
            fi
            pids+=("$pid")
        done < <(pgrep -u "$USER" -f "$pat" 2>/dev/null)
    done
    printf '%s\n' "${pids[@]+"${pids[@]}"}" | sort -un | sed '/^$/d'
}

# The sockets this copy can own: run_ex{2,3,6}.sh name them after a cksum of
# the exercise folder (keep this formula in sync with those scripts).
our_sockets() {
    local d tag
    for d in "$ROOT"/part_*/excercise_*/; do
        [[ -d "$d" ]] || continue
        tag=$(printf '%s' "${d%/}" | cksum | cut -d' ' -f1)
        find /tmp -maxdepth 1 -name "ipi_*_${tag}" -user "$USER" 2>/dev/null
    done
    # Exercise 1 deliberately keeps the fixed address, so its socket name is
    # shared between copies: only claim it if no process is holding it.
    local ex1=/tmp/ipi_h2o-mace_ex1
    if [[ -S "$ex1" && -O "$ex1" ]] && ! fuser "$ex1" >/dev/null 2>&1; then
        echo "$ex1"
    fi
}

echo "Tutorial copy: $ROOT"
if [[ $SCOPED -eq 0 ]]; then
    echo "WARNING: no /proc on this system -- cannot tell one copy of the"
    echo "         tutorial from another, so every matching process of user"
    echo "         '$USER' will be stopped."
fi
echo

echo "== 1/3  asking i-PI to exit cleanly (EXIT files, 10 s) =="
exit_files=()
for d in "$ROOT"/part_*/excercise_*/; do
    [[ -d "$d" ]] || continue
    touch "${d}EXIT" 2>/dev/null && exit_files+=("${d}EXIT")
done
if [[ ${#exit_files[@]} -eq 0 ]]; then
    echo "   no exercise folders found -- nothing to signal"
else
    echo "   dropped EXIT in ${#exit_files[@]} folders"
    sleep 10
    rm -f "${exit_files[@]}"
    echo "   EXIT files removed"
fi

echo "== 2/3  terminating anything still running =="
left=$(matching_pids)
if [[ -z "$left" ]]; then
    echo "   nothing left running"
else
    echo "   still alive:"
    # shellcheck disable=SC2086
    ps -o pid,etime,cmd -p $(echo "$left" | tr '\n' ' ') 2>/dev/null | tail -n +2 | sed 's/^/     /'
    # shellcheck disable=SC2086
    kill $left 2>/dev/null
    sleep 3
    stubborn=$(matching_pids)
    if [[ -n "$stubborn" ]]; then
        # shellcheck disable=SC2086
        kill -9 $stubborn 2>/dev/null
        sleep 1
    fi
    if [[ -z "$(matching_pids)" ]]; then
        echo "   all stopped"
    else
        echo "   WARNING: some processes could not be stopped:"
        matching_pids | sed 's/^/     pid /'
    fi
fi

echo "== 3/3  removing leftover i-PI sockets of this copy =="
sockets=$(our_sockets)
if [[ -z "$sockets" ]]; then
    echo "   none"
else
    while read -r sock; do
        [[ -n "$sock" ]] || continue
        echo "   removing $sock"
        rm -f "$sock"
    done <<< "$sockets"
fi

echo
echo "Done. You can now re-run the cell that launches the simulation."
echo "(To also delete the output of a previous run, use the clean.sh of the"
echo " exercise folder, or ./clean_all.sh for everything.)"
