import numpy as np

leftp = np.array([[ 135.,  755.],
 [ 230.,  764.],
 [ 326.,  773.],
 [ 520.,  791.],
 [ 618.,  800.],
 [ 716.,  809.],
 [ 915.,  828.]])


rightp = np.array([[ 193.,  761.],
 [ 289.,  770.],
 [ 385.,  779.],
 [ 579.,  796.],
 [ 678.,  806.],
 [ 777.,  815.],
 [ 976.,  833.]])
 
dist1 = np.power((leftp[:-1,0]-leftp[1:,0]),2) + np.power((leftp[:-1,1]-leftp[1:,1]),2)
dist1 = np.sqrt(dist1)

dist2 = np.power((rightp[:-1,0]-rightp[1:,0]),2) + np.power((rightp[:-1,1]-rightp[1:,1]),2)
dist2 = np.sqrt(dist2)


d  = np.sort(dist1)
ratio = np.abs(d[1:]-d[:-1])
ratio = np.divide(ratio, d[:-1])
if np.any(ratio>0.2):
    ind = np.argmax(np.abs(d[1:]-d[:-1]))
    m = np.mean(d[:ind+1])
    ngaps = np.round(dist1/m)
    newleft = leftp[0]
    for i in range(len(ngaps)):
        gap = ngaps[i]
        if int(gap)==1:
            newleft = np.append(newleft, leftp[i+1])
        elif int(gap)==2:
            newleft = np.append(newleft, (leftp[i]+leftp[i+1])/2)
            newleft = np.append(newleft, leftp[i+1])
    leftp = np.reshape(newleft, (len(newleft)/2,2))

    
            
    

