import sys
from matplotlib import pyplot as plt
import math
import numpy as np

def logLiks(ys, d):
  l = 0
  for y in ys:
    l += math.log(lik(y,d))
  return l

def liks(ys, d, fac=1):
  l = 1
  for y in ys:
    l *= lik(y,d, fac)
  return l

def lik(y,d, fac=1):
  norm = math.gamma((d+1)/2.0) / (math.gamma(d/2.0) * math.sqrt(d*math.pi))
  p = (1 + y*y/d)**((d+1)/(-2.0))
  return norm*p*fac

if __name__ == "__main__":
  with open('../tdfData','r') as f:
    vals = f.read().strip()[:-2].split('\n')[2:]
    vals = map(lambda x: (x.strip().split(',')), vals)
    samples = [float(x) for y in vals for x in y if x!= '']

  ls = []
  ds = []
  tp = "ll"

  if tp == "l":
    norm = 0.0
    fac = 5
    title = "Posterior for tdf"
    for d in np.arange(2,100,0.01):
      #print d
      cur = liks(samples, d, fac)
      norm += cur
      if d <= 6.5:
        ls.append(cur)
        ds.append(d)

    ls = [l/norm for l in ls]
    print sum(ls)

  elif tp == "ll":
    title = "Log Likelihood for tdf"
    for d in np.arange(2,100,0.1):
      ls.append(logLiks(samples, d))
      ds.append(d)

  plt.plot(ds, ls)
  plt.title(title)
  plt.xlabel("degrees of freedom")
  plt.show()

  
