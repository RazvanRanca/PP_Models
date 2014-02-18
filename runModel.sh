#!/bin/bash

if [[ $# < 2 ]]
then
  echo "Need to specify the model name and the language (v/b) to run it in"
  exit
fi

model=$1
language=$2
modelType=$3
outputFile=$3"Samples"

if [[ $language == "v" ]]
then
  cd $model"/Venture"
  ( /usr/bin/time -f "\n%e" python $model"Model.py" $3; ) 2>> $outputFile
fi

if [[ $language == "b" ]]
then
  cd $model"/Bugs"
  ( /usr/bin/time -f "%e" "./"$model"Model.sh" $3; ) 2>> "temp"
  cat "temp" >> $outputFile
  rm "temp"
fi
