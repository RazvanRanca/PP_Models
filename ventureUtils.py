from venture.shortcuts import *
import numpy as np

def posterior_samples(v, var_name,no_samples,no_burns,int_mh):
    s=[];
    for sample in range(no_burns):
        v.infer(1)
    counter = no_burns
    for sample in range(no_samples): #v.restart()
        v.infer(int_mh)
        counter += int_mh
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s.append((counter,v.report(label)))
    return s

def save_samples(samples, path, model):
  fn = path + "/" + model + "Samples"
  with open(fn, 'w') as f:
    f.write('\n'.join([str(s[0]) + " " + str(s[1]) for s in samples]))
