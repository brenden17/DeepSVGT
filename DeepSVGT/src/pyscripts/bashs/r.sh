#!/bin/bash

[[ -d /home/brendan/workspace/phd/1phd/1phd/data/results/$1_$2_$3 ]] || mkdir /home/brendan/workspace/phd/1phd/1phd/data/results/$1_$2_$3

# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
# 					--rm nvcr.io/nvidia/pytorch:22.07-py3-03 \
# 					python ./1phd/pyscripts/run.py \
# 					--chrom $1 \
# 					--query_start $2 \
# 					--query_svtype $3 \
# 					--seq_size 256 \
# 					--bam_files 1phd/data/bam/Hair_44_trimmed.sorted.bam \
# 					--bam_files 1phd/data/bam/Muscle_44_trimmed.sorted.bam \
# 					--bam_files 1phd/data/bam/Hair_09_trimmed.sorted.bam \
# 					--bam_files 1phd/data/bam/Muscle_09_trimmed.sorted.bam \
# 					--vcf_file 1phd/data/cutevcf/Hair_44.vcf.gz \
# 					--result_dir 1phd/data/results/$1_$2_$3 \
# 					--epochs 50 

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
					--rm nvcr.io/nvidia/pytorch:22.07-py3-03 \
					python ./1phd/pyscripts/run.py \
					--chrom $1 \
					--query_start $2 \
					--query_svtype $3 \
					--seq_size 256 \
					--bam_files 1phd/data/bam2/Hair_44_trimmed.filtered.sorted.bam \
					--bam_files 1phd/data/bam2/Muscle_44_trimmed.filtered.sorted.bam \
					--bam_files 1phd/data/bam2/Hair_40_trimmed.filtered.sorted.bam \
					--bam_files 1phd/data/bam2/Muscle_40_trimmed.filtered.sorted.bam \
					--bam_files 1phd/data/bam2/Hair_09_trimmed.filtered.sorted.bam \
					--bam_files 1phd/data/bam2/Muscle_09_trimmed.filtered.sorted.bam \
					--vcf_file 1phd/data/cutevcf/Hair_44.vcf.gz \
					--result_dir 1phd/data/results/$1_$2_$3 \
#					--epochs 50 

#docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/1phd:/workspace/1phd \
#					--rm nvcr.io/nvidia/pytorch:22.07-py3-03 \
#					python ./1phd/pyscripts/run.py \
#					--chrom 1 \
#					--query_start 1378773 \
#					--seq_size 256 \
#					--bam_files 1phd/data/data/Hair_40_Blockchain_v6.0.1.ARS_Btau5.sorted.bam \
#					--bam_files 1phd/data/data/Hair_09_Blockchain_v6.0.1.ARS_Btau5.sorted.bam \
#					--bam_files 1phd/data/data/Muscle_40_Blockchain_v6.0.1.ARS_Btau5.sorted.bam \
#					--bam_files 1phd/data/data/Muscle_09_Blockchain_v6.0.1.ARS_Btau5.sorted.bam \
#					--vcf_file 1phd/data/vcf/cutesv.vcf.gz \
#					--epochs 80 
