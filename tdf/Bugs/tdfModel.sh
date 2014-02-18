#!/bin/bash

modelType=$1
runFile="run""${modelType^}"
codaFile=$modelType"CODAchain1.txt"

OpenBUGS $runFile
mv $codaFile $modelType"Samples"
