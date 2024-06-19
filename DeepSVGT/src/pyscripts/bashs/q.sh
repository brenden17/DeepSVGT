#!/bin/bash

bcftools query -f './r.sh %CHROM %POS %SVTYPE && ./v.sh %CHROM %POS %SVLEN %SVTYPE\n' ~/workspace/phd/1phd/1phd/data/cutevcf/Hair_44.vcf.gz | grep $1 | less
