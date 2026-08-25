#!/usr/bin/env bash
# Remove everything that can be regenerated: the run/analysis outputs of
# every exercise (via its clean.sh) and the ~1 GB of data downloaded by
# ./download_trajectories.sh.
#
#   ./clean_all.sh            outputs + the Part III trajectory data
#   ./clean_all.sh --models   the above and the downloaded MACE models
#
# ./download_all.sh restores the data; the notebooks restore the rest.
set -e
cd "$(dirname "$0")"

clean_models=0
case "${1-}" in
    "")        ;;
    --models)  clean_models=1 ;;
    *)         echo "usage: $0 [--models]"; exit 1 ;;
esac

for script in part_*/excercise_*/clean.sh; do
    echo "# cleaning ${script%/clean.sh}"
    (cd "$(dirname "$script")" && bash clean.sh)
done

echo "# deleting the downloaded data in part_iii/trajectory_files"
# .gitkeep and the folder's own README.md belong to the repository.
find part_iii/trajectory_files -mindepth 1 \
     -not -name .gitkeep -not -name README.md -delete

if [[ $clean_models -eq 1 ]]; then
    echo "# deleting the downloaded models in MODELS"
    rm -f MODELS/*.model
fi

echo "# done"
