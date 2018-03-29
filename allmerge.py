import pickle
import os

with open("/home/amit/Desktop/vignesh/11_04_backsub.pickle",'rb') as f1:
    imgdict1 = pickle.load(f1)

with open("/home/amit/Desktop/vignesh/11_03_backsub.pickle",'rb') as f2:
    imgdict2 = pickle.load(f2)  

with open("/home/amit/Desktop/vignesh/08_02_backsub.pickle",'rb') as f3:
    imgdict3 = pickle.load(f3)      

# with open("/home/amit/Desktop/vignesh/11_04_backsub2.pickle",'rb') as f4:
#     imgdict4 = pickle.load(f4)      

# with open("/home/amit/Desktop/vignesh/11_03_backsub2.pickle",'rb') as f5:
#     imgdict5 = pickle.load(f5)      

with open("/home/amit/Desktop/vignesh/08_02_backsub2.pickle",'rb') as f6:
    imgdict6 = pickle.load(f6)   


dict1 = dict(imgdict1, **imgdict2)
dict2 = dict(dict1, **imgdict3)
# dict3 = dict(dict2, **imgdict4)
# dict4 = dict(dict3, **imgdict5)
dict5 = dict(dict2, **imgdict6)

print(len(dict5.keys()))

with open("/home/amit/Desktop/vignesh/allmerge3.pickle", 'r+') as f:
    pickle.dump(dict5, f)

