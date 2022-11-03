#!/bin/bash

# Set run parameters

scenarios=$1 # 3G
scenariotags=$2 # m1

baseeoss=$3 # SKI5
hybrideoss=$4 # 2009

numversions=$5 # v1

# Create output files and directories

mkdir -p "./batch"
dagfile="./batch/SimulateTwinStarInference.dag"
configfile="./batch/SimulateTwinStarInference.config"

echo "${scenarios}" > $configfile
echo "${scenariotags}" >> $configfile
echo "${baseeoss}" >> $configfile
echo "${hybrideoss}" >> $configfile
echo "${numversions}" >> $configfile

IFS=',' read -r -a scenarios <<< "$scenarios"
IFS=',' read -r -a scenariotags <<< "$scenariotags"

IFS=',' read -r -a baseeoss <<< "$baseeoss"
IFS=',' read -r -a hybrideoss <<< "$hybrideoss"

# Print sub files

binfile="SimulateTwinStarInference.sh"
subfile="./batch/${binfile}.sub"
args="arguments = \"\$(scenario) \$(baseeos) \$(hybrideos) \$(scenariotag) \$(versiontag)\""

echo "universe = vanilla" > $subfile
echo "executable = ./$binfile" >> $subfile
echo $args >> $subfile
echo "output = ./batch/$binfile.out" >> $subfile
echo "error = ./batch/$binfile.err" >> $subfile
echo "log = ./batch/$binfile.log" >> $subfile
echo "getenv = True" >> $subfile
echo "accounting_group = ligo.dev.o4.cbc.extrememmatter.bilby" >> $subfile
echo "accounting_group_user = philippe.landry" >> $subfile
echo "queue 1" >> $subfile

# Print dag file

echo "### Run $binfile batch jobs ###" > $dagfile

job=0

for k in $(seq 0 $((${numversions}-1)))
do

    for j in $(seq 0 $((${#baseeoss[@]}-1)))
    do

        for i in $(seq 0 $((${#scenarios[@]}-1)))
        do
        
            echo "JOB $job $subfile" >> $dagfile
            echo "VARS $job scenario=\"${scenarios[$i]}\" baseeos=\"${baseeoss[$j]}\" hybrideos=\"${hybrideoss[$j]}\" scenariotag=\"${scenariotags[$i]}\" versiontag=\"v${k}\"" >> $dagfile
            echo "RETRY $job 1" >> $dagfile
            
            if [[("$i" > 0) && ("${scenarios[$i]}" == "${scenarios[$(($i-1))]}")]]
            then
                if [[("$i" > 1) && ("${scenarios[$i]}" == "${scenarios[$(($i-2))]}")]]
                then
                    echo "PARENT $(($job-2)) CHILD $job" >> $dagfile
                else
                    echo "PARENT $(($job-1)) CHILD $job" >> $dagfile
                fi
            fi
            
            job=$(($job+1))
            
        done
    done
done