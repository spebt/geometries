#!/bin/bash
# Gather the current deck figures into deck_figures/ so they can travel by git.
#
# The Mac has no SSH access to CCR (Open OnDemand only), so scp/rsync are not
# available. The repo is synced both ways daily, which makes it the practical
# channel for the two or three plots a weekly deck needs.
#
# Usage (on CCR, from anywhere in the repo):
#   ./collect_deck_figures.sh
#   git add deck_figures && git commit -m "Refresh deck figures" && git push
# then on the Mac:  git pull

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DEST="${ROOT}/deck_figures"
SRC="${ROOT}/spebt/optimization/results/mobo_plots"
mkdir -p "${DEST}"

# Only the plots a deck actually uses. Adding more defeats the point: this
# directory is committed, so it should stay small enough that nobody thinks
# twice about pulling it.
WANTED=(
  hypervolume_convergence.png
  pareto_expansion.png
  metric_scatter_combined.png
  headtohead.png
)

found=0
for f in "${WANTED[@]}"; do
  if [[ -f "${SRC}/${f}" ]]; then
    cp -f "${SRC}/${f}" "${DEST}/${f}"
    printf '  %-36s %s\n' "${f}" "$(du -h "${DEST}/${f}" | cut -f1)"
    found=$((found + 1))
  else
    echo "  [skip] ${f} not found in ${SRC}"
  fi
done

if [[ ${found} -eq 0 ]]; then
  echo "Nothing copied. Has analyze_mobo_convergence.py been run?" >&2
  exit 1
fi

echo
echo "${found} figure(s) in ${DEST}"
du -sh "${DEST}" | awk '{print "  total: " $1}'
echo
echo "Next:  git add deck_figures && git commit -m 'Refresh deck figures' && git push"
