import numpy as np
import sys
from skimage import io as sio
import os
from matplotlib import pyplot
def hitRates(testMap,gtMap):
    eps = sys.float_info.epsilon

    neg_gtMap = 1-gtMap
    #sio.imshow(neg_gtMap)
    #sio.show()
    neg_testMap = 1-testMap
    #sio.imshow(neg_testMap)
    #sio.show()
    #sio.imshow(testMap*gtMap)
    #sio.show()
    #sio.imshow(testMap*neg_gtMap)
    #sio.show()
    hitCount = np.sum(testMap*gtMap)#Tp
    trueAvoidCount = np.sum(neg_gtMap*neg_testMap)#Tn
    missCount = np.sum(testMap*neg_gtMap)#Fp
    falseAvoidCount = np.sum(neg_testMap*gtMap)#Fn
    Precision = hitCount/(eps+hitCount+missCount)
    hitRate = hitCount / (eps+ hitCount + falseAvoidCount);#Recall
    falseAlarm = 1 - trueAvoidCount / (eps+trueAvoidCount+missCount)
    return Precision, hitRate, falseAlarm

def thresholdBased_HR_FR(sMap,thresholds,gtMap):
    numOfThreshs = thresholds.__len__()
    hitRate = np.zeros([numOfThreshs])
    falseAlarm = np.zeros([numOfThreshs])
    Precision = np.zeros([numOfThreshs])

    for threshIdx in range (0,numOfThreshs):
        cThrsh=thresholds[threshIdx]
        Precision[threshIdx] , hitRate[threshIdx] , falseAlarm[threshIdx]= hitRates((sMap>=cThrsh),gtMap)
    return Precision,hitRate, falseAlarm


salMappath = ['/home/nyarbel/Python_Gaze_Prediction/MSRA10K_MDF_MAPS/','/home/nyarbel/PCA_Saliency/PCA_Saliency_CVPR2013/OUT/']
opt = ['g-','g^']
alg_nam = ['MDF','PCA_Saliency']
thresholds = np.linspace(1,0,20)
gtpath = '/home/nyarbel/Python_Gaze_Prediction/MSRA10K_Imgs_GT/Imgs/'
gtext = 'png'
outpath = '/home/nyarbel/Python_Gaze_Prediction/Benchmarks/'
subgt = '/home/nyarbel/Python_Gaze_Prediction/ground_truth_experiment/'
mHitRate= np.zeros([2,thresholds.__len__()])
mFalseAlarm= np.zeros([2,thresholds.__len__()])
mPrecision = np.zeros([2,thresholds.__len__()])
line = []
for j in range(0,2):
    if j==0 :
        salMapdatafiles = [fn for fn in os.listdir(salMappath[j]) if fn.endswith('npy')]
    else:
        salMapdatafiles = [fn for fn in os.listdir(salMappath[j]) if fn.endswith('png')]

    salMapdatafiles.sort()
    numOfFiles = salMapdatafiles.__len__()
    #outFiles = [fn for fn in os.listdir(salMappath) if fn.endswith('npy')]
    for imIndx in range(0,numOfFiles):
        print("Processing image %d out of %d\n" % (imIndx,numOfFiles))
        gtMap = sio.imread(gtpath+salMapdatafiles[imIndx][0:-3]+gtext)/255;
        img = sio.imread(gtpath+salMapdatafiles[imIndx][0:-3]+'jpg');
        sio.imsave('/home/nyarbel/Python_Gaze_Prediction/experiment_picture/'+salMapdatafiles[imIndx][0:-3]+'jpg',img)
        #sio.imsave(salMapdatafiles[imIndx][0:-3]+'png',np.load(salMappath + salMapdatafiles[imIndx]).item()['sal_map'])
        #sio.imsave(subgt+salMapdatafiles[imIndx][0:-3]+gtext,gtMap)
        if j == 0:
            sMap = np.load(salMappath[j] + salMapdatafiles[imIndx]).item()['sal_map']
        else :
            sMap = sio.imread(salMappath[j] + salMapdatafiles[imIndx])
        sMap =sMap/np.max(sMap)
        sMap[sMap<0]=0
        gtSize = gtMap.shape;
        Precision,hitRate, falseAlarm = thresholdBased_HR_FR(sMap, thresholds, gtMap)
        data = {}
        data['hitRate']=hitRate
        mHitRate[j]= np.add(mHitRate[j],hitRate)
        data['falseAlarm']=falseAlarm
        mPrecision[j] = np.add(mPrecision[j],Precision)
        mFalseAlarm[j]= np.add(mFalseAlarm[j],falseAlarm)
        #np.save(outpath+salMapdatafiles[imIndx],data)


    mHitRate[j]=mHitRate[j]/numOfFiles
    mFalseAlarm[j]=mFalseAlarm[j]/numOfFiles
    mPrecision[j] = mPrecision[j]/numOfFiles
    AUC = np.trapz(mPrecision[j],mFalseAlarm[j])
    print("%s AUC: %d\n" % (alg_nam[j],AUC))


line1 =pyplot.plot(mHitRate[0],mPrecision[0],'ro')
line2 =pyplot.plot(mHitRate[1],mPrecision[1],'go')
data = {}
data['Precision'] = mPrecision
data['HitRate']=mHitRate
data['FalseAlarm']=mFalseAlarm
np.save('ROC_results',data)
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend(['MDF','PCA'], loc='lower right')
pyplot.axis((0,1,0,1))
pyplot.grid()
#pyplot.plot(mHitRate,mPrecision)
pyplot.show()



