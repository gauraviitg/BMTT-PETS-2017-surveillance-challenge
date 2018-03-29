import itertools
import numpy as np
import pickle

with open('08_02_results.txt') as f:
    test = []
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [int(i) for i in line]
            test.append(line)


test = list(itertools.chain.from_iterable(test))

seq = 10
threshold1 = 5
threshold2 = 50
newseq = []
start = test[0]

for ind,i in enumerate(test):
	cnt = 0
	for j in test[ind:]:
		if (i+seq >=j):
			cnt+=1
	if cnt > threshold1:
		# print(i)
		newseq.append(i)

print(newseq)

# initial = newseq[0]
# for k in newseq:
#     if (k-initial) < threshold2:


start = newseq[0] 
end = newseq[len(newseq)-1] 

fin = []
for k in range(start,end+1):
	fin.append(k)

# print(fin)

pred  = [0]*1056
gt = []
for i in fin:
	pred[i] = 1

print(pred)


with open("/home/amit/Desktop/vignesh/allmerge2.pickle",'rb') as f1:
    spatial_test_data=pickle.load(f1)

key1 = spatial_test_data.keys()
keylist1=[]
for i in key1:
    newkey1 = int(i)
    keylist1.append(newkey1)
    

key2 = sorted(keylist1)
keylist2=[]
for i in key2:
    if (i > 4000):                # change limits of indices for other cases
        newkey2 = str(i)
        keylist2.append(newkey2)    

for k in keylist2:
	gt.append(spatial_test_data[k])

# print(gt)

accuracy0 = 0
cnt0 = 0		
for i,j in zip(pred,gt):
    if i==j:
        accuracy0+=1

    cnt0+=1    
print(accuracy0)
print(cnt0)
print(accuracy0/float(cnt0))


	



