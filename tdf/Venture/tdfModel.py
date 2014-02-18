from venture.shortcuts import *
import sys
sys.path.append("../../")
import ventureUtils as vu
from collections import Counter
import numpy as np
import os

def genTdf(v):
  v.assume("y", "(student_t 4)")
  samples = vu.posterior_samples(v, "y", no_samples=1000, no_burns=0, int_mh=100)
  vu.save_samples(samples, os.getcwd(), "gen")

def readData():
  with open('../tdfdata.txt','r') as f:
    raw = f.read()
    ys = map(float, raw.split('\n',2)[-1][:-2].translate(None,'\n\r').split(','))
  return ys

def runModel(v, ys, mType):
  if mType == "cont":
    v.assume("d", "(uniform_continuous 2 100)")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "disc1":
    v.assume("d", "(uniform_discrete 2 50)")
    v.assume("y", "(lambda () (student_t d))")
  elif mType == "disc2":
    v.assume("d", "(uniform_discrete 20 60)")
    v.assume("y", "(lambda () (student_t (/ d 10)))")
  else:
    raise Exception("Unknown model type: " + mType)

  [v.observe("(y)", str(ys[i])) for i in range(len(ys))]
  samples = vu.posterior_samples(v, "d", no_samples=100, no_burns=100, int_mh=1)
  vals = map(lambda x:x[1], samples)
  print "Sample mean: ", np.mean(vals), " Sample Stdev: ", np.std(vals)
  vu.save_samples(samples, os.getcwd(), mType)

if __name__ == "__main__":
  modelType = sys.argv[1]
  v = make_church_prime_ripl()
  ys = readData()
  runModel(v, ys, modelType)


