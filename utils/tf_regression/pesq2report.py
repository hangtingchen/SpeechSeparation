# -*- coding: utf-8 -*-
'''
@author : chenhangting 
@data : 2018/12/19
@note : pesq to report
'''
import sys
import numpy as np

if __name__=='__main__':
    if(len(sys.argv)!=3):
        print("Usage : python3 {} <input-pesq-list-file> <output-pesq-report-file>")
        exit(1)
    else:
        print("python3 "+" ".join(sys.argv))
    d=dict()
    with open(sys.argv[1],'r') as f:
        for line in f:
            try:
                name,mix,infer=line.strip().split(" ")
                k=name.split(".")[0].rsplit("_",maxsplit=1)[1]
                if(d.get(k) is None):d[k]=list()
                d[k].append((float(mix),float(infer)))
            except ValueError:
                continue
    d_report=dict()
    ks=list(d.keys());ks.sort()
    for k in ks:
        d_report[k]=np.mean(d[k],0)
    with open(sys.argv[2],'w') as f:
        f.write("SNR\tMixed\tInfer\n")
        for k in ks:
            f.write("{}\t{:.2f}\t{:.2f}\n".format(k,d_report[k][0],d_report[k][1]))
