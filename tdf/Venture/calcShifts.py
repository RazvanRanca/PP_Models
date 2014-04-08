import numpy as np
import math

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
  
def getTransMat(mode, eps, depth):
  noNodes = 2**depth
  rest = 2**(-depth)
  intStart = 0
  normTrans = float(1) / depth
  mat = np.zeros((noNodes+2,noNodes+2)) # one mode and one trap absorbing nodes needed

  for start in range(noNodes):
    intStart = float(start) / noNodes
    overlap = calcOverlap(intStart, intStart+rest, mode-eps, mode+eps)
    if overlap > 0:
      mat[start, noNodes] = overlap/rest

    binStart = bin(start)[2:].zfill(depth)[::-1]
    #print start, bin(start), binStart
    overlap = 0
    for bitInd in range(len(binStart)):
      effect = 2**bitInd
      lim = effect / (2.0 * noNodes)
      if binStart[bitInd] == "1":
        effect *= -1
        minTot = getBoundary(mode+eps, lim*2, "min")
        maxTot = mode + lim
        #print 1, intStart, intStart+rest, minTot, maxTot, calcOverlap(intStart, intStart+rest, minTot, maxTot)
        overlap = max(overlap, calcOverlap(intStart, intStart+rest, minTot, maxTot))
      else:
        minTot = mode - lim
        maxTot = getBoundary(mode-eps, lim*2, "max")
        #print 0, intStart, intStart+rest, minTot, maxTot, calcOverlap(intStart, intStart+rest, minTot, maxTot)
        overlap = max(overlap, calcOverlap(intStart, intStart+rest, minTot, maxTot))

      end = start + effect
      #print start, end, getTransProb(intStart, float(end) / noNodes, rest, mode)
      mat[start, end] = getTransProb(intStart, float(end) / noNodes, rest, mode) * normTrans

    if overlap > 0:
      mat[start, noNodes+1] = overlap/rest

  return np.matrix(mat)

if __name__ == "__main__":
  print getTransMat(0.49, 0.01, 2)
