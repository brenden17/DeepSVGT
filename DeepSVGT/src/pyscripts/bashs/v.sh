#!/bin/bash

#Set variables
chrom=$1
location=$2
svlen=$3
svtype=$4

gap=50

abssvlen=${svlen/#-/}
startlocation=$((location - gap))
abssvlen=${svlen/#-/}
locationlen=$((location + abssvlen))
endlocationlen=$((location + abssvlen+50))

#tempbatchfile="~/workspace/phd/1phd/1phd/data/results/${chrom}_${location}_${svtype}/temp.batch"
tempbatchfile="./1phd/data/results/${chrom}_${location}_${svtype}/temp.batch"

echo '#!/bin/bash' > ${tempbatchfile}

while read line; do
  eval echo "$line"
done < "./temp.txt" >> ${tempbatchfile}

echo 'exit' >> ${tempbatchfile}

igv --batch=${tempbatchfile}
