from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
  ps = []
  hs = {}
  with open(sys.argv[1] + "/" + sys.argv[2] + "CODAchain1.txt",'r') as f:
    for line in f:
      val = float(line.strip().split()[-1])
      ps.append(val)
      try:
        hs[val] += 1
      except:
        hs[val] = 1

  #xs = []
  #ys = []
  #for (k,v) in sorted(hs.items()):
  #  xs.append(k)
  #  ys.append(v)

  #plt.plot(xs, ys)

  plt.hist(ps,int(sys.argv[3]))
  plt.show()
