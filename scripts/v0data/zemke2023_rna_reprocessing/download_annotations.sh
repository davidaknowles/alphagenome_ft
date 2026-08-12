#!/bin/bash
# Download exact-assembly NCBI RefSeq gene annotations from UCSC.

set -euo pipefail

output_dir="${1:-outputs/v0data/zemke2023-gene-supervision/annotations}"
mkdir -p "$output_dir"

urls=(
  "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz"
  "https://hgdownload.soe.ucsc.edu/goldenPath/rheMac10/bigZips/genes/rheMac10.ncbiRefSeq.gtf.gz"
  "https://hgdownload.soe.ucsc.edu/goldenPath/calJac4/bigZips/genes/ncbiRefSeq.gtf.gz"
  "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/genes/mm10.ncbiRefSeq.gtf.gz"
)
names=(hg38 rheMac10 calJac4 mm10)

for index in "${!urls[@]}"; do
  destination="$output_dir/${names[$index]}.ncbiRefSeq.gtf.gz"
  if [[ ! -s "$destination" ]]; then
    curl -fL --retry 3 -o "${destination}.part" "${urls[$index]}"
    mv "${destination}.part" "$destination"
  fi
done
