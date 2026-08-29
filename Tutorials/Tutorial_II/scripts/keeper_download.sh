#!/usr/bin/env bash
# Shared helper for the student-facing downloaders (download_models.sh,
# download_trajectories.sh). It mirrors folders of the Keeper (MPDL) public
# share onto the local repository, whose layout it reproduces exactly.
#
# Not meant to be run directly. A downloader sources it and then calls
#   fetch_folders  <local/remote folder> ...
#   check_required <file> ...
# with REPO_DIR pointing at the repository root.
#
# Files already present with the correct size are skipped and interrupted
# transfers are resumed, so the downloaders are safe to re-run.

KEEPER="https://keeper.mpdl.mpg.de"
TOKEN="f6e5df518a0942a7bc4d"
REMOTE_ROOT="/Tutorial_II"          # path inside the share

# Local artifacts of the reference recipe that need not be downloaded. The
# README.md of every folder is kept in git, so the copy on the share never
# overwrites it.
SKIP_REGEX='(\.tmp|\.log|/README\.md)$'

failed=0

# Recursively print "size<TAB>path" for every file under the remote dir $1.
walk() {
    local dir="$1" size path
    while IFS=$'\t' read -r size path; do
        if [[ "$size" == DIR ]]; then
            walk "$path"
        else
            printf '%s\t%s\n' "$size" "$path"
        fi
    done < <(
        curl -sL --fail "${KEEPER}/api/v2.1/share-links/${TOKEN}/dirents/?path=${dir}/" |
        python3 -c '
import sys, json
for e in json.load(sys.stdin).get("dirent_list", []):
    if e["is_dir"]:
        print("DIR\t" + e["folder_path"].rstrip("/"))
    else:
        print(str(e["size"]) + "\t" + e["file_path"])'
    )
}

# Mirror each folder given as an argument (path relative to the share root
# and to REPO_DIR, which are the same).
fetch_folders() {
    local folder size rpath rel local_path
    for folder in "$@"; do
        echo "== ${folder} =="
        while IFS=$'\t' read -r size rpath; do
            rel="${rpath#${REMOTE_ROOT}/}"
            [[ "$rel" =~ $SKIP_REGEX ]] && continue
            local_path="${REPO_DIR}/${rel}"
            if [[ -f "$local_path" && "$(stat -c%s "$local_path")" == "$size" ]]; then
                echo "  [ok]   ${rel}"
                continue
            fi
            mkdir -p "$(dirname "$local_path")"
            echo "  [get]  ${rel}  ($(numfmt --to=iec "$size"))"
            if ! curl -L --fail --retry 3 -C - --progress-bar \
                    "${KEEPER}/d/${TOKEN}/files/?p=${rpath}&dl=1" -o "$local_path"; then
                echo "  [FAILED] ${rel}"; failed=1
            elif [[ "$(stat -c%s "$local_path")" != "$size" ]]; then
                echo "  [SIZE MISMATCH] ${rel}"; failed=1
            fi
        done < <(walk "${REMOTE_ROOT}/${folder}")
    done
}

# Check the files the tutorial actually reads and report the result. $1 is
# the name of the downloader to suggest on failure, the rest are the files.
check_required() {
    local script="$1"; shift
    local f missing=0
    echo
    for f in "$@"; do
        [[ -f "${REPO_DIR}/${f}" ]] || { echo "MISSING: ${f}"; missing=1; }
    done
    if [[ $failed -eq 0 && $missing -eq 0 ]]; then
        return 0
    fi
    echo "Download incomplete — re-run ./${script} (or report the missing files)."
    return 1
}
