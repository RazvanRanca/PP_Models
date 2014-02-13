from venture.shortcuts import *
import numpy as np

def posterior_samples(v, var_name,no_samples,int_mh):
    s=[];
    for sample in range(no_samples): #v.restart()
        v.infer(int_mh)
        label = var_name + str(np.random.randint(10**5))+str(sample)
        v.predict(var_name,label)
        s.append(v.report(label))
    return s
