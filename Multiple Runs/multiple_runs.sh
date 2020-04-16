# !/bin/bash

{
    a=100

    echo "The program will run $a time!"
    echo

    while ((a>0)); do

        python3 Assignment\ 2/mood_lyric_generator_Kadakia_111304945.py

        ((--a))
        echo
        echo "$a" runs are left!
        echo

    done
} >& progress.txt &

exit ;
