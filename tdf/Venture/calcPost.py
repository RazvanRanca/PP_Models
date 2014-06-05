import sys
from matplotlib import pyplot as plt
import math
import numpy as np
import cPickle
import random

def logLiks(ys, d, base=math.e):
  l = 0
  for y in ys:
    l += math.log(lik(y,d), base)
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

def flipPrior(s,e):
  rez = s + random.random()
  m = int(math.ceil(math.log(e-s-1,2)))
  for n in range(m):
    rez += random.randrange(2)*(2**n)
  return rez

def sumPrior(s, parts):
  rez = s
  for part in parts:
    rez += random.random()*part
  return rez

if __name__ == "__main__":
  with open('../tdfData','r') as f:
    vals = f.read().strip()[:-2].split('\n')[2:]
    vals = map(lambda x: (x.strip().split(',')), vals)
    samples = [float(x) for y in vals for x in y if x!= '']

  ls = []
  ds = []
  try:
    tp = sys.argv[1]
  except:
    tp = "l"

  if tp == "l":
    norm = 0.0
    fac = 5
    title = "Posterior for Tdf21"
    count = 0
    for d in np.arange(2,100,0.01):
      count += 1
      if count % 100 == 0:
        print d
      cur = liks(samples, d, fac)
      norm += cur
      if d < 40:
        ls.append(cur)
        ds.append(d)

    ls = [l/norm for l in ls]
    with open("Posterior21",'w') as f:
      cPickle.dump((ds,ls), f)
    print sum(ls)
    plt.plot(ds,ls)

  elif tp == "lDisc1":
    norm = 0.0
    fac = 5
    title = "Posterior for Tdf - Course Discrete"
    count = 0
    for d in np.arange(2,50,1):
      count += 1
      if count % 100 == 0:
        print d
      cur = liks(samples, d, fac)
      norm += cur
      if d < 10:
        ls.append(cur)
        ds.append(d)

    ls = [l/norm for l in ls]
    plt.plot(ds,ls,'D', markersize=10)

    with open("Posterior4Disc1",'w') as f:
      cPickle.dump((ds,ls), f)
    print sum(ls)

  elif tp == "lDisc2":
    norm = 0.0
    fac = 5
    title = "Posterior for Tdf - Fine Discrete"
    count = 0
    for d in np.arange(2,6.1,0.1):
      count += 1
      if count % 100 == 0:
        print d
      cur = liks(samples, d, fac)
      norm += cur
      if d < 10:
        ls.append(cur)
        ds.append(d)

    ls = [l/norm for l in ls]
    plt.plot(ds,ls,'D', markersize=10)

    with open("Posterior4Disc2",'w') as f:
      cPickle.dump((ds,ls), f)
    print sum(ls)

  elif tp == "rlc": #cumulative
    title = "Cumulative dist. of posterior for tdf"
    with open("Posterior4",'r') as f:
      ds, ls = cPickle.load(f)
      cls = []
      curSum = 0
      for l in ls:
        curSum += l
        cls.append(curSum)
      ls = cls

  elif tp == "rl": 
    title = "Posterior for tdf"
    with open("Posterior4",'r') as f:
      ds, ls = cPickle.load(f)

  elif tp == "rl21": 
    title = "True posterior of Tdf21"
    with open("Posterior21",'r') as f:
      ds, ls = cPickle.load(f)

  elif tp == "rll21": 
    title = "True posterior of Tdf21"
    with open("posteriorDict21",'r') as f:
      print cPickle.load(f)

  elif tp == "ul":
    norm = 0.0
    fac = 5
    title = "Posterior for tdf"
    maxL = -1
    maxD = None
    for d in np.arange(2,100,0.1):
      #print d
      cur = liks(samples, d, fac)
      ls.append(cur)
      ds.append(d)
      if cur > maxL:
        maxL = cur
        maxD = d
    print maxL, maxD

  elif tp == "ll":
    title = "Log Likelihood for Tdf"
    for d in np.arange(2,100,0.1):
      ls.append(logLiks(samples, d))
      ds.append(d)

    with open("PosteriorLL4",'w') as f:
      cPickle.dump((ds,ls), f)
    print sum(ls)

  elif tp == "sll":
    title = "Log Likelihood for tdf"
    dc = {}
    fac = 1000.0
    inc = 1/fac
    d = 2
    while d <= 100.01:
      ll = logLiks(samples, d, base=2)
      ls.append(ll)
      ds.append(d)
      dInd = round(d*fac)
      dc[dInd] = ll
      if dInd % 100 == 0:
        print d
      d += inc
    with open("posteriorDict21", 'w') as f:
      cPickle.dump((fac, dc), f, cPickle.HIGHEST_PROTOCOL)
    #print dc
  elif tp == "testDist1":
    rezs = [flipPrior(2, 100) for i in range(1000000)]
    print min(rezs), max(rezs)
    plt.hist(rezs, min(200, len(set(rezs))))
    plt.show()
  elif tp == "testDist2":
    rezs = [sumPrior(2, [1,2,95]) for i in range(10000000)]
    print min(rezs), max(rezs)
    print plt.hist(rezs, min(200, len(set(rezs))))
    plt.xlabel("Degrees of Freedom", size=20)
    plt.ylabel("Sample Frequency", size=20)
    plt.title("Tdf - [1,2,95] induced prior", size=30)
    plt.show()

  try:
    part = 2** (-1.0*int(sys.argv[2]))
    for p in np.arange(0,1.001,part):
      line = 2 + 98*p
      plt.plot([line,line],[0,max(ls)], 'k')
    plt.xlim([2,10])
    mode = 4.214
    eps = float(sys.argv[3])
    plt.axvspan(mode-eps,mode+eps, facecolor='g', alpha=0.7)
  except:
    pass
  plt.plot(ds, ls)
  plt.title(title, size=30)
  plt.xlabel("Degrees of freedom", size=20)
  plt.ylabel("Probability", size=20)
  plt.show()

  
