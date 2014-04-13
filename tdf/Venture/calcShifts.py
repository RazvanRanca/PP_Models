import numpy as np
import math
import sys
from matplotlib import pyplot as plt

def calcOverlap(n1s, n1e, n2s, n2e):
  if n1e <= n2s or n1s >= n2e or n1e <= n1s or n2e <= n2s:
    return 0

  if n1s <= n2s:
    if n1e <= n2e:
      return n1e - n2s
    else:
      return n2e - n2s
  else:
    if n2e <= n1e:
      return n2e - n1s
    else:
      return n1e - n1s

def getTransProb(n1, n2, rest, mode): # uses distance from mode as proxy for LL
  n1s = abs(n1 - mode)
  n1e = abs(n1 + rest - mode)
  n2s = abs(n2 - mode)
  n2e = abs(n2 + rest - mode)

  n1Min = min(n1s,n1e)
  n1Max = max(n1s,n1e)
  n2Min = min(n2s,n2e)
  n2Max = max(n2s,n2e)

  #print n1,n2,rest,mode,n1Min, n1Max, n2Min, n2Max
  if n1Max <= n2Min:
    return 0
  if n2Max <= n1Min:
    return 1

  if n1Min <= n2Min:
    overlap = n1Max - n2Min
    prob = (overlap / rest) * (1 - (overlap / (2*rest))) # if not in overlap, don't transition, if in on average will be halfway
                                                         # and so will accept a target subinterval of size (rest - overlap/2)
  else:
    overlap = n2Max - n1Min
    prob = (1 - (overlap / rest)) + (overlap / rest) * (1 - (overlap / (2*rest))) # same as before except now always accept if out of overlap

  return prob

def getBoundary(lim, step, tp):
  if tp == "min":
    return math.ceil(lim/step) * step
  elif tp == "max":
    return math.floor(lim/step) * step
  else:
    raise Exception("Unknown type of boundary: " + tp)

def getTransMat(mode, eps, depth, silent = False):
  noNodes = 2**depth
  rest = 2**(-depth)
  intStart = 0
  normTrans = float(1) / (depth + 1)
  trans = {}
  toAbsorb = {}

  for start in range(noNodes):
    intStart = float(start) / noNodes
    binStart = bin(start)[2:].zfill(depth)[::-1]
    #print start, bin(start), binStart

    overlapTrap = float("inf")
    overlapMode = float("inf")
    for bitInd in range(len(binStart)):
      effect = 2**bitInd
      if binStart[bitInd] == "0":
        end = start + effect
      else:
        end = start - effect

      #print start, end, getTransProb(intStart, float(end) / noNodes, rest, mode)
      rez = (end, getTransProb(intStart, float(end) / noNodes, rest, mode) * normTrans)
      #print start, rez

      if rez[1] != 0:
        try:
          trans[start].append(rez)
        except:
          trans[start] = [rez]


      lim = effect / (2.0 * noNodes)
      if binStart[bitInd] == "0":
        minTotTrap = mode - lim
        maxTotTrap = getBoundary(mode-eps, lim*2, "max")
        minTotMode = max(mode - lim, getBoundary(mode-eps, lim*2, "max"))
        maxTotMode = getBoundary(mode+eps, lim*2, "min")
        if not silent:
          print 0, intStart, intStart+rest, rez[1], minTotTrap, maxTotTrap, calcOverlap(intStart, intStart+rest, minTotTrap, maxTotTrap), "--", minTotMode, maxTotMode, calcOverlap(intStart, intStart+rest, minTotMode, maxTotMode)
        overlapMode = min(overlapMode, calcOverlap(intStart, intStart+rest, minTotMode, maxTotMode))
        if rez[1] > 0:
          overlapTrap = min(overlapTrap, calcOverlap(intStart, intStart+rest, minTotTrap, maxTotTrap))
      else:
        minTotTrap = getBoundary(mode+eps, lim*2, "min")
        minTotMode = getBoundary(mode-eps, lim*2, "max")
        maxTotTrap = mode + lim
        maxTotMode = min(mode + lim, getBoundary(mode+eps, lim*2, "min"))
        if not silent:
          print 1, intStart, intStart+rest, rez[1], minTotTrap, maxTotTrap, calcOverlap(intStart, intStart+rest, minTotTrap, maxTotTrap), "--", minTotMode, maxTotMode, calcOverlap(intStart, intStart+rest, minTotMode, maxTotMode)
        overlapMode = min(overlapMode, calcOverlap(intStart, intStart+rest, minTotMode, maxTotMode))
        if rez[1] > 0:
          overlapTrap = min(overlapTrap, calcOverlap(intStart, intStart+rest, minTotTrap, maxTotTrap))
    
    if overlapTrap > 0 and overlapTrap < float("inf"):
      if overlapTrap == rest: # this is a trapped node
        toAbsorb[start] = noNodes+1
      else: # here we might still avoid getting trapped, depending on the uniform
        try:
          trans[start].append((noNodes+1, normTrans * overlapTrap/rest))
        except:
          trans[start] = [(noNodes+1, normTrans * overlapTrap/rest)]

    if overlapMode > 0 and overlapMode < float("inf"):
      if overlapMode == rest: # this is a trapped node
        toAbsorb[start] = noNodes
      else: # here we might still avoid getting trapped, depending on the uniform
        try:
          trans[start].append((noNodes, normTrans * overlapMode/rest))
        except:
          trans[start] = [(noNodes, normTrans * overlapMode/rest)]

  for n in range(noNodes): #trapped node with no valid transitions
    if n not in trans and n not in toAbsorb:
      toAbsorb[n] = noNodes + 1

  noNodes -= len(toAbsorb) # we can fold all nodes representing trapped states into one absorbing node
  mat = np.zeros((noNodes,noNodes+2)) # one mode and one trap absorbing node needed
  
  curStart = 0
  nodeMap = {}
  for start in range(noNodes + len(toAbsorb) + 2):
    if start in toAbsorb:
      nodeMap[start] = toAbsorb[start] - len(toAbsorb)
    else:
      nodeMap[start] = curStart
      curStart += 1

  #print nodeMap
  #print toAbsorb
  #print trans

  for start in range(noNodes + len(toAbsorb)):
    if start not in toAbsorb:
      for end, prob in trans[start]:
        mat[nodeMap[start], nodeMap[end]] = prob

  #mat[0,4] = 0.5
  #mat[1, 5] = 0.6
  rowSums = mat.sum(axis=1)
  for n in range(noNodes):
    mat[n,n] = 1 - rowSums[n]

  if not silent:
    print mat
  return np.split(mat, [noNodes], axis=1), nodeMap

def getAbsorbProbs(mode, eps, depth, mat = None, silent = False):
  (Q, R), nm = getTransMat(mode, eps, depth, silent=silent)
  dim = len(Q)
  Ninv = np.eye(dim) - Q
  if not silent:
    print Ninv.sum(axis=0)
    print Ninv.sum(axis=1)
  N = np.linalg.inv(Ninv)
  #print N
  B = np.dot(N, R)

  if len(nm) > dim + 2:
    Bf = np.zeros((len(nm)-2, 2))
    for start, end in nm.items():
      if start < len(nm)-2:
        if end < dim:
          Bf[start, 0] = B[end, 0]
          Bf[start, 1] = B[end, 1]
        elif end == dim:
          Bf[start, 0] = 1.0
          Bf[start, 1] = 0.0
        elif end == dim+1:
          Bf[start, 0] = 0.0
          Bf[start, 1] = 1.0
        else:
          raise Exception("nodeMap contains node: " + end + " larger than dim: " + dim)
  else:
    Bf = B

  return Bf

def getAbsorbProbs1(mat):
  (Q, R) = np.split(mat, [len(mat)], axis=1)
  N = np.linalg.inv(np.eye(len(Q)) - Q)
  B = np.dot(N, R)
  return B

def getAllAPs():
  eps = 0.005
  for mode in [x/1000.0 for x in range(5,1000,10)]:
    for depth in range(1,6):
      probs = getAbsorbProbs(float(mode), eps, depth, silent=True)
      #print float(mode), eps, depth, getAbsorbProbs(float(mode), eps, depth, silent=True)
      for interv in range(len(probs)):
        print depth, interv, mode, eps, float(probs[interv][0])

def displayDiff(realFn, simFn, tp = None):
  sim = {}
  real = {}
  with open(simFn,'r') as f:
    for line in f:
      depth, interv, mode, eps, prob = map(float, line.strip().split())
      if tp == "depth":
        try:
          sim[depth].append(prob)
        except:
          sim[depth] = [prob]
      elif tp == "mode":
        try:
          sim[mode].append(prob)
        except:
          sim[mode] = [prob]
      else:
        sim[(depth, interv, mode)] = prob
        if depth == 3 and mode == 0.255:
          print (depth, interv, mode), sim[(depth, interv, mode)]

  diffs = []
  with open(realFn,'r') as f:
    for line in f:
      depth, interv, mode, _, prob, _, _ = map(float, line.strip().split())
      if tp == "depth":
        try:
          real[depth].append(prob)
        except:
          real[depth] = [prob]
      elif tp == "mode":
        try:
          real[mode].append(prob)
        except:
          real[mode] = [prob]
      else:
        real[(depth, interv, mode)] = prob
        diffs.append(prob - sim[(depth, interv, mode)])
        if diffs[-1] == -1:
          print depth, interv, mode, prob, sim[(depth, interv, mode)]

  if tp == None:
    plt.hist(diffs, 100)
    plt.xlabel("Difference between simulated and observed prob")
    plt.ylabel("Number of occurrences")
    plt.title("Distribution of (observed prob. - simulated prob.) of succesfull burn-in")
  else:
    simXs = sorted(sim.keys())
    realXs = sorted(real.keys())
    simYs = [sum(sim[k]) / float(len(sim[k])) for k in simXs] 
    realYs = [sum(real[k]) / float(len(real[k])) for k in realXs]

    p1,p2 = plt.plot(simXs, simYs, "b", realXs, realYs, "r")
    plt.legend([p1,p2],["MC Simulated Probabilities","Observed Probabilities with idealized LL"])
    if tp == "mode":
      plt.xlabel("Mode placement")
    if tp == "depth":
      plt.xlabel("Binomial depth")
    plt.ylabel("Average probability of reaching mode +/- " + str(eps))
  plt.show()


if __name__ == "__main__":

  #print getAbsorbProbs(0.255, 0.005, 3, silent=True)
  displayDiff("absorbProbs001_ns", "simAbsorbProbs")
  #print getAbsorbProbs(float(sys.argv[1]), 0.005, int(sys.argv[2]), silent=True)
  #getAllAPs()
  mat1 = np.matrix([ [2.0/3, 1.0/6, 0, 0, 1.0/6, 0],
                     [1.0/6, 2.0/3, 0, 0, 0, 1.0/6],
                     [1.0/3, 0, 2.0/3, 0, 0, 0],
                     [0, 1.0/3, 1.0/3, 1.0/3, 0, 0] ])
  mat2 = np.matrix([ [1.0/2, 0, 0, 0, 1.0/6, 1.0/3],
                     [0, 1.0/2, 0, 0, 1.0/3, 1.0/6],
                     [1.0/6, 0, 2.0/3, 0, 1.0/6, 0],
                     [0, 1.0/6, 1.0/3, 1.0/3, 0, 1.0/6] ])
  #print getAbsorbProbs1(mat2)
