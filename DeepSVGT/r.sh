#!/bin/bash

python ./src/pyscripts/run.py \
					--chrom $1 \
					--query_start $2 \
					--query_svtype $3 \
					--bam_files ./data/bam-A3/Hair_44_trimmed.sorted.bam \
					--bam_files ./data/bam-A3/Hair_40_trimmed.sorted.bam \
					--bam_files ./data/bam-A3/Hair_09_trimmed.sorted.bam \
					--vcf_file ./data/bam-A3/Hair_44.sniffles_genotypes.vcf