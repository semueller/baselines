#!/bin/sh
echo `which python`
for i in 0 25 50 100 200;
 do
   for e in "HandManipulateBlock-v0" "HandManipulatePen-v0"
     do
        python experiment/play.py $i $e;
   done;
done;
