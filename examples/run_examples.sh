#!/bin/bash

Help()
{
    echo "Run the example scripts in the subfolders of examples/ "
    echo "which are passed as arguements."
    echo
}

verbose=false

while getopts ":hv:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      v) # verbose mode
         verbose=true
         echo "Running examples verbosely.";;
     \?) # Invalid option
         echo "Error: Invalid option. Option must be one of [-h|v]."
         exit;;
   esac
done

# Pattern match for optiomns.
pattern='^\-[h|v]$'

for file in $(find "examples/" -name *.py)
do
    if [ "$verbose" = true ]; then
        python3 $file
    else
        python3 $file > /dev/null
    fi

    if [ $? = 0 ]; then
        echo "success running $file"
    else
        echo "failure: $? when running $file"
        exit 1
    fi
    sleep 2
done

echo "SUCCESS - all examples run without error."
