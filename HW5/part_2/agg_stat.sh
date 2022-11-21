#!/bin/bash
count=10
height=10000
echo "cpu_num,width,height,time" >> stat.csv
for cpu_num in 1 2 3 4 5 6 7 8 
do
    for width in 1000 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 
    do
        for i in $(seq $count)
        do
        echo $cpu_num,$width,$height,$(/usr/bin/mpirun -np $cpu_num ./hw5_2 -w $width --height $height -s) >> stat.csv
        done
    done
done