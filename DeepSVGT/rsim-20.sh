#!/bin/bash

result_dir="1phd/data/sim5-result-e12/bam-depth20-del-del-result"


docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
					--rm nvcr.io/nvidia/pytorch:22.07-py3-04 \
					python ./1phd/pyscripts/run.py \
					--bam_files 1phd/data/sim5/bam-depth20-del/NC_037332.1-1.sorted.bam \
					--bam_files 1phd/data/sim5/bam-depth20-del/NC_037332.1-2.sorted.bam \
					--bam_files 1phd/data/sim5/bam-depth20-del/NC_037332.1-3.sorted.bam \
					--vcf_file 1phd/data/sim5/bam-depth20-del/NC_037332.1-1.vcf \
					--target_bam_file 0 \
					--result_dir ${result_dir}

# sleep 600

# result_dir="1phd/data/sim5-result-e12/bam-depth20-ins-ins-result


# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
# 					--rm nvcr.io/nvidia/pytorch:22.07-py3-04 \
# 					python ./1phd/pyscripts/run.py \
# 					--bam_files 1phd/data/sim5/bam-depth20-ins/NC_037332.1-1.sorted.bam \
# 					--bam_files 1phd/data/sim5/bam-depth20-ins/NC_037332.1-2.sorted.bam \
# 					--bam_files 1phd/data/sim5/bam-depth20-ins/NC_037332.1-3.sorted.bam \
# 					--vcf_file 1phd/data/sim5/bam-depth20-ins/NC_037332.1-2.vcf \
# 					--target_bam_file 1 \
# 					--result_dir ${result_dir}

cat ${result_dir}/*.txt | grep UNMATCH | wc -l

# sleep 600

# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
# 					--rm nvcr.io/nvidia/pytorch:22.07-py3-04 \
# 					python ./1phd/pyscripts/run.py \
# 					--bam_files 1phd/data/sim4/bam-depth20-inv/NC_037332.1-1.sorted.bam \
# 					--bam_files 1phd/data/sim4/bam-depth20-inv/NC_037332.1-2.sorted.bam \
# 					--bam_files 1phd/data/sim4/bam-depth20-inv/NC_037332.1-3.sorted.bam \
# 					--vcf_file 1phd/data/sim4/bam-depth20-inv/NC_037332.1-3.vcf \
# 					--target_bam_file 2 \
# 					--result_dir 1phd/data/bam-depth20-inv-inv-result

