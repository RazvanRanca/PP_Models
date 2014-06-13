#!/bin/bash

str=$(./texcount.pl -1 -inc acs-dissertation.tex | cut -d" " -f1 | sed 's/+/ + /g')
echo $str
expr $str
