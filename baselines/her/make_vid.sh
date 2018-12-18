#!/bin/sh
echo `which python`
for i in $(seq 0 25 650);
 do
   for e in "HandManipulateBlock-v0" "HandManipulatePen-v0"
     do
        python experiment/play.py $i $e;
   done;
done;
