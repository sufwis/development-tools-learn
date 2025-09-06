#!/usr/bin/env bash

if [[ -e sout.log  ]]; then
    rm sout.log -f
fi
touch sout.log
if [[ -e serr.log  ]]; then
    rm serr.log -f
fi
touch serr.log

echo "start~"
declare -i jud=0
declare -i tar=1
declare -i cnt=0
while [[ $jud -ne $tar ]];do
    ./test.sh >> ./sout.log 2>> ./serr.log
    jud=$?
    cnt=$(( $cnt + 1 ))
done

echo "counts: $cnt"
