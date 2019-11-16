#!/bin/bash

for i in "$@"; do
    # check last four characters of filename
    # https://www.cyberciti.biz/faq/bash-get-basename-of-filename-or-directory-name/
    if [ ${i: -4} == ".ele" ] ; then
        # extract filename
        # https://stackoverflow.com/questions/965053/extract-filename-and-extension-in-bash
        filename=$(basename -- "$i")
        filename="${filename%%.*}"
        bin/run ${i} ${filename}
    fi
done

