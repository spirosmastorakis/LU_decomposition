#!/bin/bash

#type the filename of the first argument of diff command
file1=output_block_p2p

#type the filename of the second argument of diff command
file2=output_serial

#print results to the following file
diff_file=diff.txt

diff "${file1}" "${file2}"  > "${diff_file}"
