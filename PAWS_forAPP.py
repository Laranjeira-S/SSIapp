#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 12:48:59 2021

@author: Simao
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:11:53 2020

@author: Simao

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk,ImageDraw
from PIL import Image as IM
import glob
import random
import cv2  
from random import shuffle
import os
import sys
import matplotlib.pyplot as plt
from skimage  import measure as region 
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unet import unet
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate
import imutils
import joblib
import keras_preprocessing
import tensorflow
import math
import xlsxwriter
from tkinter import filedialog
from tkinter import *
import matplotlib.patches as patches
import time


                
# def Measurement(filename,CROP,FRAMES,file):
def Measurement(filename,CROP,file,VIDEO_rotate, model, TSmodel, ITSmodel):

    Folder='Results_ROTATION'

    # model=unet()
    # model.load_weights('PAWS900.h5')
    
    # TSmodel=unet()
    # TSmodel.load_weights('TS10000.h5')
    
    # ITSmodel=unet()
    # ITSmodel.load_weights('ITS10000.h5')
 
    cap=cv2.VideoCapture(filename)
    length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    TSF=np.ones(length)*np.nan
    ITSF=np.ones(length)*np.nan
    
    TS_l=np.ones(length)*np.nan
    TS_r=np.ones(length)*np.nan
    ITS_l=np.ones(length)*np.nan
    ITS_r=np.ones(length)*np.nan
    Ratio_area=np.ones(length)*np.nan
    
    
    Y=[r[0] for r in CROP]
    X=[r[1] for r in CROP]
       
    img = IM.new('L', (256, 256), 0)
    ImageDraw.Draw(img).polygon(CROP, outline=1, fill=1)
    BOX = np.array(img)
    j=0
    rt,frame=cap.read()
    
    SAVING_MK=np.zeros([256,256,length])
    # Experiment labelling all the slides
    # i=780
    i=0
    
    while rt:  
        start=time.time()

        cap.set(1,i)
        rt,frame=cap.read()
        MK=np.zeros([frame.shape[0],frame.shape[1]])
        if VIDEO_rotate==1:            
            frame=imutils.rotate_bound(np.uint8(frame),90)
            BOX=imutils.rotate_bound(np.uint8(BOX),90)
        
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(frame,(256, 256))/255
        raw[BOX==0,:]=0
        raw = raw[:,:,0:3]
    # plt.imshow(raw)
    # plt.pause(0.1)
    #predict the mask 
        pred = model.predict(np.expand_dims(raw, 0))
    
    #         #mask post-processing 
        msk  = pred.squeeze()
        msk = np.stack((msk,)*3, axis=-1)
        msk[msk >= 0.5] = 1 
        msk[msk < 0.5] = 0 
    
    # #        raw[msk>0]=255
    #         plt.imshow(raw)
    #         plt.pause(0.1)
    #     #    
        MASK=msk[:,:,0]
        # ResLMASK=cv2.resize(MASK,(frame.shape[1],frame.shape[0]))
        LMASK=label(MASK>0)
        REGIONS=regionprops(LMASK)
        Area=[r.area for r in REGIONS]
        SArea=np.argsort(Area)
        Positions=SArea[-2:]
        
        
        # ResLMASK=cv2.resize(np.uint8(LMASK),(frame.shape[1],frame.shape[0]))
    
        number=1
        ts=np.empty((2))
        ts[:]=-999
        its=np.empty((2))
        its[:]=-999
        
        
        if Positions.size==2:
            Length=np.zeros(2)
            idx=0
            
            IMAGEI=np.uint(LMASK==Positions[0]+1)
            ResLMASKI=cv2.resize(np.uint8(IMAGEI),(frame.shape[1],frame.shape[0]))
        
            RI=regionprops(label(ResLMASKI>0))

            # y0, x0 = props.centroid
            minrI,mincI,maxrI,maxcI=RI[0].bbox
            
        
            I=frame[minrI:maxrI,mincI:maxcI]
            OI = RI[0].orientation
            
            if OI<0:
                AngleI=np.degrees(OI)-90 

            else: 
                AngleI=np.degrees(OI)+90 

                
            RotatedI=imutils.rotate_bound(np.uint8(I), AngleI)
            digitsTOlabelI = cv2.resize(RotatedI,(256, 256))/255
            digitsTOlabelI = digitsTOlabelI[:,:,0:3]
        
            predTS = TSmodel.predict(np.expand_dims(digitsTOlabelI, 0))
            PI_TSmsk  = predTS.squeeze()
            PI_TSmsk = np.stack((PI_TSmsk,)*3, axis=-1)
            PI_TSmsk[PI_TSmsk >= 0.5] = 1 
            PI_TSmsk[PI_TSmsk < 0.5] = 0 
            Original_TSmsk=cv2.resize(PI_TSmsk,(RotatedI.shape[1],RotatedI.shape[0]))
        
            
            TSLPI=label(PI_TSmsk>0)
            TSCLOTS=regionprops(TSLPI)
            TSCLOTSArea=np.array([r.area for r in TSCLOTS])
        
            # plt.imshow(ResLMASK)
            # plt.pause(0.1)
            
            Original_TSLPI=label(Original_TSmsk>0)
            Original_TSCLOTS=regionprops(Original_TSLPI)
            Original_TSCLOTSArea=np.array([r.area for r in Original_TSCLOTS])
        
            
            
            # plt.imshow(LMASK)
            # plt.pause(0.1)
            
            predITS = ITSmodel.predict(np.expand_dims(digitsTOlabelI, 0))
            PI_ITSmsk  = predITS.squeeze()
            PI_ITSmsk = np.stack((PI_ITSmsk,)*3, axis=-1)
            PI_ITSmsk[PI_ITSmsk >= 0.5] = 1 
            PI_ITSmsk[PI_ITSmsk < 0.5] = 0 
            Original_ITSmsk=cv2.resize(PI_ITSmsk,(RotatedI.shape[1],RotatedI.shape[0]))

        
            ITSLPI=label(PI_ITSmsk>0)
            ITSCLOTS=regionprops(ITSLPI)
            ITSCLOTSArea=np.array([r.area for r in ITSCLOTS])
            
            
            Original_ITSLPI=label(Original_ITSmsk>0)
            Original_ITSCLOTS=regionprops(Original_ITSLPI)
            Original_ITSCLOTSArea=np.array([r.area for r in Original_ITSCLOTS])
        
                        
            
            
            ######## OTHER PAW
            IMAGEII=np.uint(LMASK==Positions[1]+1)
            ResLMASKII=cv2.resize(np.uint8(IMAGEII),(frame.shape[1],frame.shape[0]))
        
            RII=regionprops(label(ResLMASKII>0))
            # y0, x0 = props.centroid
            minrII,mincII,maxrII,maxcII=RII[0].bbox
            
        
            II=frame[minrII:maxrII,mincII:maxcII]
            OII = RII[0].orientation
            
            if OII<0:
                AngleII=np.degrees(OII)-90 

            else: 
                AngleII=np.degrees(OII)+90 

            
            RotatedII=imutils.rotate_bound(np.uint8(II), AngleII)
            digitsTOlabelII = cv2.resize(RotatedII,(256, 256))/255
            digitsTOlabelII = digitsTOlabelII[:,:,0:3]
        
            predTSII = TSmodel.predict(np.expand_dims(digitsTOlabelII, 0))
            PII_TSmsk  = predTSII.squeeze()
            PII_TSmsk = np.stack((PII_TSmsk,)*3, axis=-1)
            PII_TSmsk[PII_TSmsk >= 0.5] = 1 
            PII_TSmsk[PII_TSmsk < 0.5] = 0 
            OriginalII_TSmsk=cv2.resize(PII_TSmsk,(RotatedII.shape[1],RotatedII.shape[0]))
            
            OriginalII_TSLP=label(OriginalII_TSmsk>0)
            OriginalII_TSCLOTS=regionprops(OriginalII_TSLP)
            OriginalII_TSCLOTSArea=np.array([r.area for r in OriginalII_TSCLOTS])
        
                                
            
            
            TSLPII=label(PII_TSmsk>0)
            TSCLOTSII=regionprops(TSLPII)
            TSCLOTSAreaII=np.array([r.area for r in TSCLOTSII])
        

            predITS = ITSmodel.predict(np.expand_dims(digitsTOlabelII, 0))
            PII_ITSmsk  = predITS.squeeze()
            PII_ITSmsk = np.stack((PII_ITSmsk,)*3, axis=-1)
            PII_ITSmsk[PII_ITSmsk >= 0.5] = 1 
            PII_ITSmsk[PII_ITSmsk < 0.5] = 0 
            OriginalII_ITSmsk=cv2.resize(PII_ITSmsk,(RotatedII.shape[1],RotatedII.shape[0]))
           
            OriginalII_ITSLP=label(OriginalII_ITSmsk>0)
            OriginalII_ITSCLOTS=regionprops(OriginalII_ITSLP)
            OriginalII_ITSCLOTSArea=np.array([r.area for r in OriginalII_ITSCLOTS])
        
                

        
        
            ITSLPII=label(PII_ITSmsk>0)
            ITSCLOTSII=regionprops(ITSLPII)
            ITSCLOTSAreaII=np.array([r.area for r in ITSCLOTSII])
           
            
            # Centroid=R[0].centroid
            # fig, ax=plt.subplots()
                        
            # ax.imshow(frame)
            
            # rect=patches.Rectangle((maxrI,maxcI),maxcII-mincII,maxrII-minrII,angle=np.degrees(O),linewidth=1, edgecolor='r',facecolor='none')
            # ax.add_patch(rect)
            # plt.show()
            ##   
            if Original_TSCLOTSArea.size>=2  and Original_ITSCLOTSArea.size>=2 and OriginalII_TSCLOTSArea.size>=2  and OriginalII_ITSCLOTSArea.size>=2 and TSCLOTSArea.size>=2  and ITSCLOTSArea.size>=2 and TSCLOTSAreaII.size>=2  and ITSCLOTSAreaII.size>=2: 
                # Find the centre of the box
                Cy=256/2
                
                # Finding the X coordinate of the different blobs 
                TSWHRE=np.argsort(TSCLOTSArea)
                TSPOTS=TSWHRE[-2:]                
                        
                XTS=np.array([r.centroid[1] for r in TSCLOTS])
                X_TS=XTS[TSPOTS]
                
                ITSWHRE=np.argsort(ITSCLOTSArea)
                ITSPOTS=ITSWHRE[-2:]                
                        
                XITS=np.array([r.centroid[1] for r in ITSCLOTS])
                X_ITS=XITS[ITSPOTS]
                
                TSWHREII=np.argsort(TSCLOTSAreaII)
                TSPOTSII=TSWHREII[-2:]                 
                                                
                XTSII=np.array([r.centroid[1] for r in TSCLOTSII])
                X_TSII=XTSII[TSPOTSII]
                
                ITSWHREII=np.argsort(ITSCLOTSAreaII)
                ITSPOTSII=ITSWHREII[-2:]                
        
                XITSII=np.array([r.centroid[1] for r in ITSCLOTSII])
                X_ITSII=XITSII[ITSPOTSII]

                Original_TSWHRE=np.argsort(Original_TSCLOTSArea)
                Original_TSPOTS=Original_TSWHRE[-2:]  
                
                Original_ITSWHRE=np.argsort(Original_ITSCLOTSArea)
                Original_ITSPOTS=Original_ITSWHRE[-2:]  
                                
                OriginalII_TSWHRE=np.argsort(OriginalII_TSCLOTSArea)
                OriginalII_TSPOTS=OriginalII_TSWHRE[-2:]  
                
                OriginalII_ITSWHRE=np.argsort(OriginalII_ITSCLOTSArea)
                OriginalII_ITSPOTS=OriginalII_ITSWHRE[-2:]  
                
                Original_XTS=np.array([r.centroid[1] for r in Original_TSCLOTS])
                Original_X_TS=Original_XTS[Original_TSPOTS]
                
                Original_XITS=np.array([r.centroid[1] for r in Original_ITSCLOTS])
                Original_X_ITS=Original_XITS[Original_ITSPOTS]
                
                Original_XTSII=np.array([r.centroid[1] for r in OriginalII_TSCLOTS])
                Original_X_TSII=Original_XTSII[OriginalII_TSPOTS]
                
                Original_XITSII=np.array([r.centroid[1] for r in OriginalII_ITSCLOTS])
                Original_X_ITSII=Original_XITSII[OriginalII_ITSPOTS]     
                
                Original_YTS=np.array([r.centroid[0] for r in Original_TSCLOTS])
                Original_Y_TS=Original_YTS[Original_TSPOTS]
                
                Original_YITS=np.array([r.centroid[0] for r in Original_ITSCLOTS])
                Original_Y_ITS=Original_YITS[Original_ITSPOTS]
                
                Original_YTSII=np.array([r.centroid[0] for r in OriginalII_TSCLOTS])
                Original_Y_TSII=Original_YTSII[OriginalII_TSPOTS]
                
                Original_YITSII=np.array([r.centroid[0] for r in OriginalII_ITSCLOTS])
                Original_Y_ITSII=Original_YITSII[OriginalII_ITSPOTS]  
                
                
                # If the X coordinates of the blobs is larger than the centre of the image it means that the rotated rat is oriented towards the Y axis.
                if np.sum(np.uint(X_TS>128)+np.uint(X_ITS>128)+np.uint(X_TSII>128)+np.uint(X_ITSII>128))==0:
                   
                    ROTATE_LMASK=imutils.rotate_bound(np.float32(LMASK), AngleII)
                    ROTATE_LMASKI=ROTATE_LMASK==Positions[0]+1
                    ROTATE_LMASKII=ROTATE_LMASK==Positions[1]+1
                    
                    if np.sum(ROTATE_LMASKII)>0 and np.sum(ROTATE_LMASKI)>0:
                        # Error_ROTATE_LMASK=ROTATE_LMASK>0
                        # The blob with largest Y coordinate will be the left paw which is normally the control paw
                        Blobs_ROTED_LMASKI= regionprops(label(ROTATE_LMASKI))
                        Blobs_ROTED_LMASK_AreaI=[r.area for r in Blobs_ROTED_LMASKI]
                        Blobs_ROTED_LMASK_SAreaI=np.argsort(Blobs_ROTED_LMASK_AreaI)
         
                        
                        YCOORDINATES_of_Blobs=np.array([y.centroid[0] for y in Blobs_ROTED_LMASKI])
                        YCOORDINATES_of_BlobsI=YCOORDINATES_of_Blobs[Blobs_ROTED_LMASK_SAreaI[-1]]
                        
                        Blobs_ROTED_LMASKII= regionprops(label(ROTATE_LMASKII))
                        Blobs_ROTED_LMASK_AreaII=[r.area for r in Blobs_ROTED_LMASKII]
                        Blobs_ROTED_LMASK_SAreaII=np.argsort(Blobs_ROTED_LMASK_AreaII)
         
                        
                        YCOORDINATES_of_Blobs=np.array([y.centroid[0] for y in Blobs_ROTED_LMASKII])
    
                        YCOORDINATES_of_BlobsII=YCOORDINATES_of_Blobs[Blobs_ROTED_LMASK_SAreaII[-1]]                   
                        
                            
                        if YCOORDINATES_of_BlobsI<YCOORDINATES_of_BlobsII:
                            LeftPaw=cv2.cvtColor(RotatedII, cv2.COLOR_BGR2RGB)
                            LEFT_TSmsk=cv2.resize(PII_TSmsk,(RotatedII.shape[1],RotatedII.shape[0]))                        
                            LEFT_ITSmsk=cv2.resize(PII_ITSmsk,(RotatedII.shape[1],RotatedII.shape[0]))                       
                            RightPaw=cv2.cvtColor(RotatedI, cv2.COLOR_BGR2RGB)                       
                            RIGHT_TSmsk=cv2.resize(PI_TSmsk,(RotatedI.shape[1],RotatedI.shape[0]))                        
                            RIGHT_ITSmsk=cv2.resize(PI_ITSmsk,(RotatedI.shape[1],RotatedI.shape[0]))
                            
                            DTSI=np.sqrt(np.diff(Original_X_TS)*np.diff(Original_X_TS)+np.diff(Original_Y_TS)*np.diff(Original_Y_TS))
                            DTSII=np.sqrt(np.diff(Original_X_TSII)*np.diff(Original_X_TSII)+np.diff(Original_Y_TSII)*np.diff(Original_Y_TSII))
                            DITSI=np.sqrt(np.diff(Original_X_ITS)*np.diff(Original_X_ITS)+np.diff(Original_Y_ITS)*np.diff(Original_Y_ITS))
                            DITSII=np.sqrt(np.diff(Original_X_ITSII)*np.diff(Original_X_ITSII)+np.diff(Original_Y_ITSII)*np.diff(Original_Y_ITSII))
                            
                            
                            TS_l[i]=DTSII
                            ITS_l[i]=DITSII
                            
                            TS_r[i]=DTSI
                            ITS_r[i]=DITSI
                            
    
                            SAVING_MK[:,:,i]=LMASK
                            
                            
                            AreaI=RI[0].area
                            AreaII=RII[0].area
                            
                            
                            
                            Ratio_area[i]=1-(np.abs(AreaII-AreaI)/AreaI)
                            
                            
                            FRAME=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
                            #  Left Blue
                            FRAME[ResLMASKII>0,:]=[200,0,0]
                            # Right Red
                            FRAME[ResLMASKI>0,:]=[0,0,200]
                            
                            minrLTS,mincLTS,ii,maxrLTS,maxcLTS,ii=TSCLOTSII[TSPOTSII[0]].bbox
                            minrLIITS,mincLIITS,ii,maxrLIITS,maxcLIITS,ii=TSCLOTSII[TSPOTSII[1]].bbox
                            
                            L_TS=np.zeros((256,256))
                            L_TS[minrLTS:maxrLTS,mincLTS:maxcLTS]=TSCLOTSII[TSPOTSII[0]].image[:,:,0]
                            L_TS[minrLIITS:maxrLIITS,mincLIITS:maxcLIITS]=TSCLOTSII[TSPOTSII[1]].image[:,:,0]
                            
                            LEFT_TS=cv2.resize(L_TS,(LEFT_TSmsk.shape[1],LEFT_TSmsk.shape[0]))
                            
                            minrLITS,mincLITS,ii,maxrLITS,maxcLITS,ii=ITSCLOTSII[ITSPOTSII[0]].bbox
                            minrLII_ITS,mincLII_ITS,ii,maxrLII_ITS,maxcLII_ITS,ii=ITSCLOTSII[ITSPOTSII[1]].bbox
                            
                            L_ITS=np.zeros((256,256))
                            L_ITS[minrLITS:maxrLITS,mincLITS:maxcLITS]=ITSCLOTSII[ITSPOTSII[0]].image[:,:,0]
                            L_ITS[minrLII_ITS:maxrLII_ITS,mincLII_ITS:maxcLII_ITS]=ITSCLOTSII[ITSPOTSII[1]].image[:,:,0]
                            
                            LEFT_ITS=cv2.resize(L_ITS,(LEFT_ITSmsk.shape[1],LEFT_ITSmsk.shape[0]))

                           
                            
                            minrRTS,mincRTS,ii,maxrRTS,maxcRTS,ii=TSCLOTS[TSPOTS[0]].bbox
                            minrRIITS,mincRIITS,ii,maxrRIITS,maxcRIITS,ii=TSCLOTS[TSPOTS[1]].bbox
                           
                            R_TS=np.zeros((256,256))
                            R_TS[minrRTS:maxrRTS,mincRTS:maxcRTS]=TSCLOTS[TSPOTS[0]].image[:,:,0]
                            R_TS[minrRIITS:maxrRIITS,mincRIITS:maxcRIITS]=TSCLOTS[TSPOTS[1]].image[:,:,0]
                           
                            RIGHT_TS=cv2.resize(R_TS,(RIGHT_TSmsk.shape[1],RIGHT_TSmsk.shape[0]))
                           
                            minrRITS,mincRITS,ii,maxrRITS,maxcRITS,ii=ITSCLOTS[ITSPOTS[0]].bbox
                            minrRII_ITS,mincRII_ITS,ii,maxrRII_ITS,maxcRII_ITS,ii=ITSCLOTS[ITSPOTS[1]].bbox
                           
                            R_ITS=np.zeros((256,256))
                            R_ITS[minrRITS:maxrRITS,mincRITS:maxcRITS]=ITSCLOTS[ITSPOTS[0]].image[:,:,0]
                            R_ITS[minrRII_ITS:maxrRII_ITS,mincRII_ITS:maxcRII_ITS]=ITSCLOTS[ITSPOTS[1]].image[:,:,0]
                           
                            RIGHT_ITS=cv2.resize(R_ITS,(RIGHT_ITSmsk.shape[1],RIGHT_ITSmsk.shape[0]))


                            
                            
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'Colours'+'/'+file+'_'+str(i)+'.png',np.uint8(FRAME))
     
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(LeftPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_ITS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(RightPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_ITS))

                        else:
            
                            LeftPaw=cv2.cvtColor(RotatedI, cv2.COLOR_BGR2RGB)                        
                            LEFT_TSmsk=cv2.resize(PI_TSmsk,(RotatedI.shape[1],RotatedI.shape[0]))                        
                            LEFT_ITSmsk=cv2.resize(PI_ITSmsk,(RotatedI.shape[1],RotatedI.shape[0]))                        
                            RightPaw=cv2.cvtColor(RotatedII, cv2.COLOR_BGR2RGB)                                              
                            RIGHT_TSmsk=cv2.resize(PII_TSmsk,(RotatedII.shape[1],RotatedII.shape[0]))                        
                            RIGHT_ITSmsk=cv2.resize(PII_ITSmsk,(RotatedII.shape[1],RotatedII.shape[0]))
                            
                            DTSI=np.sqrt(np.diff(Original_X_TS)*np.diff(Original_X_TS)+np.diff(Original_Y_TS)*np.diff(Original_Y_TS))
                            DTSII=np.sqrt(np.diff(Original_X_TSII)*np.diff(Original_X_TSII)+np.diff(Original_Y_TSII)*np.diff(Original_Y_TSII))
                            DITSI=np.sqrt(np.diff(Original_X_ITS)*np.diff(Original_X_ITS)+np.diff(Original_Y_ITS)*np.diff(Original_Y_ITS))
                            DITSII=np.sqrt(np.diff(Original_X_ITSII)*np.diff(Original_X_ITSII)+np.diff(Original_Y_ITSII)*np.diff(Original_Y_ITSII))
                            
                            
                            
                            TS_l[i]=DTSI
                            ITS_l[i]=DITSI
                            
                            TS_r[i]=DTSII
                            ITS_r[i]=DITSII
                            
    
                            
    
                            SAVING_MK[:,:,i]=LMASK
                            
                            AreaI=RI[0].area
                            AreaII=RII[0].area
                            
                            Ratio_area[i]=1-(np.abs(AreaII-AreaI)/AreaI)
                            
                            
                            # if np.isnan(Ratio_area[i]):
                            #     plt.pause(0.1)
                            FRAME=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
                            # Left Blue
                            FRAME[ResLMASKI>0,:]=[200,0,0]
                            # Right Red
                            FRAME[ResLMASKII>0,:]=[0,0,200]
                            minrLTS,mincLTS,ii,maxrLTS,maxcLTS,ii=TSCLOTS[TSPOTS[0]].bbox
                            minrLIITS,mincLIITS,ii,maxrLIITS,maxcLIITS,ii=TSCLOTS[TSPOTS[1]].bbox
                             
                            L_TS=np.zeros((256,256))
                            L_TS[minrLTS:maxrLTS,mincLTS:maxcLTS]=TSCLOTS[TSPOTS[0]].image[:,:,0]
                            L_TS[minrLIITS:maxrLIITS,mincLIITS:maxcLIITS]=TSCLOTS[TSPOTS[1]].image[:,:,0]
                             
                            LEFT_TS=cv2.resize(L_TS,(LEFT_TSmsk.shape[1],LEFT_TSmsk.shape[0]))
                             
                            minrLITS,mincLITS,ii,maxrLITS,maxcLITS,ii=ITSCLOTS[ITSPOTS[0]].bbox
                            minrLII_ITS,mincLII_ITS,ii,maxrLII_ITS,maxcLII_ITS,ii=ITSCLOTS[ITSPOTS[1]].bbox
                             
                            L_ITS=np.zeros((256,256))
                            L_ITS[minrLITS:maxrLITS,mincLITS:maxcLITS]=ITSCLOTS[ITSPOTS[0]].image[:,:,0]
                            L_ITS[minrLII_ITS:maxrLII_ITS,mincLII_ITS:maxcLII_ITS]=ITSCLOTS[ITSPOTS[1]].image[:,:,0]
                             
                            LEFT_ITS=cv2.resize(L_ITS,(LEFT_ITSmsk.shape[1],LEFT_ITSmsk.shape[0]))
                      
                            
                             
                            minrRTS,mincRTS,ii,maxrRTS,maxcRTS,ii=TSCLOTSII[TSPOTSII[0]].bbox
                            minrRIITS,mincRIITS,ii,maxrRIITS,maxcRIITS,ii=TSCLOTSII[TSPOTSII[1]].bbox
                            
                            R_TS=np.zeros((256,256))
                            R_TS[minrRTS:maxrRTS,mincRTS:maxcRTS]=TSCLOTSII[TSPOTSII[0]].image[:,:,0]
                            R_TS[minrRIITS:maxrRIITS,mincRIITS:maxcRIITS]=TSCLOTSII[TSPOTSII[1]].image[:,:,0]
                            
                            RIGHT_TS=cv2.resize(R_TS,(RIGHT_TSmsk.shape[1],RIGHT_TSmsk.shape[0]))
                            
                            minrRITS,mincRITS,ii,maxrRITS,maxcRITS,ii=ITSCLOTSII[ITSPOTSII[0]].bbox
                            minrRII_ITS,mincRII_ITS,ii,maxrRII_ITS,maxcRII_ITS,ii=ITSCLOTSII[ITSPOTSII[1]].bbox
                            
                            R_ITS=np.zeros((256,256))
                            R_ITS[minrRITS:maxrRITS,mincRITS:maxcRITS]=ITSCLOTSII[ITSPOTSII[0]].image[:,:,0]
                            R_ITS[minrRII_ITS:maxrRII_ITS,mincRII_ITS:maxcRII_ITS]=ITSCLOTSII[ITSPOTSII[1]].image[:,:,0]
                            
                            RIGHT_ITS=cv2.resize(R_ITS,(RIGHT_ITSmsk.shape[1],RIGHT_ITSmsk.shape[0]))
                      
                           
                             
                            
                            
                            # plt.imshow(FRAME)
                            # plt.pause(0.1)
                      
                            # out.write(FRAME)
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'Colours'+'/'+file+'_'+str(i)+'.png',np.uint8(FRAME))
                      
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(LeftPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_ITS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(RightPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_ITS))


                
                # If the X coordinates of the blobs is smaller than the centre of the image it means that the rotated rat is oriented away from the Y axis.
                elif np.sum(np.uint(X_TS<128)+np.uint(X_ITS<128)+np.uint(X_TSII<128)+np.uint(X_ITSII<128))==0:
                    ROTATE_LMASK=imutils.rotate_bound(np.float32(LMASK), AngleII)
                    ROTATE_LMASKI=ROTATE_LMASK==Positions[0]+1
                    ROTATE_LMASKII=ROTATE_LMASK==Positions[1]+1
                    if np.sum(ROTATE_LMASKII)>0 and np.sum(ROTATE_LMASKI)>0:

                        # Error_ROTATE_LMASK=ROTATE_LMASK>0
                        # The blob with largest Y coordinate will be the left paw which is normally the control paw
                        Blobs_ROTED_LMASKI= regionprops(label(ROTATE_LMASKI))
                        Blobs_ROTED_LMASK_AreaI=[r.area for r in Blobs_ROTED_LMASKI]
                        Blobs_ROTED_LMASK_SAreaI=np.argsort(Blobs_ROTED_LMASK_AreaI)
         
                        
                        YCOORDINATES_of_Blobs=np.array([y.centroid[0] for y in Blobs_ROTED_LMASKI])
                        YCOORDINATES_of_BlobsI=YCOORDINATES_of_Blobs[Blobs_ROTED_LMASK_SAreaI[-1]]
                        
                        Blobs_ROTED_LMASKII= regionprops(label(ROTATE_LMASKII))
                        Blobs_ROTED_LMASK_AreaII=[r.area for r in Blobs_ROTED_LMASKII]
                        Blobs_ROTED_LMASK_SAreaII=np.argsort(Blobs_ROTED_LMASK_AreaII)
         
                        
                        YCOORDINATES_of_Blobs=np.array([y.centroid[0] for y in Blobs_ROTED_LMASKII])
    
                        YCOORDINATES_of_BlobsII=YCOORDINATES_of_Blobs[Blobs_ROTED_LMASK_SAreaII[-1]]                   
                        
                        
                        try: 
                            YCOORDINATES_of_Blobs[1]
                        except:
                            plt.pause(0.1)
            
                        if YCOORDINATES_of_BlobsI<YCOORDINATES_of_BlobsII:
                            LeftPaw=cv2.cvtColor(RotatedI, cv2.COLOR_BGR2RGB)
            
                            
                            LEFT_TSmsk=cv2.resize(PI_TSmsk,(RotatedI.shape[1],RotatedI.shape[0]))                        
                            LEFT_ITSmsk=cv2.resize(PI_ITSmsk,(RotatedI.shape[1],RotatedI.shape[0]))                        
                            RightPaw=cv2.cvtColor(RotatedII, cv2.COLOR_BGR2RGB)                                                
                            RIGHT_TSmsk=cv2.resize(PII_TSmsk,(RotatedII.shape[1],RotatedII.shape[0]))                        
                            RIGHT_ITSmsk=cv2.resize(PII_ITSmsk,(RotatedII.shape[1],RotatedII.shape[0]))
    
                            DTSI=np.sqrt(np.diff(Original_X_TS)*np.diff(Original_X_TS)+np.diff(Original_Y_TS)*np.diff(Original_Y_TS))
                            DTSII=np.sqrt(np.diff(Original_X_TSII)*np.diff(Original_X_TSII)+np.diff(Original_Y_TSII)*np.diff(Original_Y_TSII))
                            DITSI=np.sqrt(np.diff(Original_X_ITS)*np.diff(Original_X_ITS)+np.diff(Original_Y_ITS)*np.diff(Original_Y_ITS))
                            DITSII=np.sqrt(np.diff(Original_X_ITSII)*np.diff(Original_X_ITSII)+np.diff(Original_Y_ITSII)*np.diff(Original_Y_ITSII))
                            
                            TS_l[i]=DTSI
                            ITS_l[i]=DITSI
                            
                            TS_r[i]=DTSII
                            ITS_r[i]=DITSII
                            
    
                            SAVING_MK[:,:,i]=LMASK
    
                            AreaI=RI[0].area
                            AreaII=RII[0].area
                            
                            Ratio_area[i]=1-(np.abs(AreaII-AreaI)/AreaI)
                            
                            
                            
                            # if np.isnan(Ratio_area[i]):
                            #     plt.pause(0.1)
                            FRAME=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
                            #  Left Blue
                            FRAME[ResLMASKI>0,:]=[200,0,0]
                            # Right Red
                            FRAME[ResLMASKII>0,:]=[0,0,200]
                         
                            minrLTS,mincLTS,ii,maxrLTS,maxcLTS,ii=TSCLOTS[TSPOTS[0]].bbox
                            minrLIITS,mincLIITS,ii,maxrLIITS,maxcLIITS,ii=TSCLOTS[TSPOTS[1]].bbox
                          
                            L_TS=np.zeros((256,256))
                            L_TS[minrLTS:maxrLTS,mincLTS:maxcLTS]=TSCLOTS[TSPOTS[0]].image[:,:,0]
                            L_TS[minrLIITS:maxrLIITS,mincLIITS:maxcLIITS]=TSCLOTS[TSPOTS[1]].image[:,:,0]
                          
                            LEFT_TS=cv2.resize(L_TS,(LEFT_TSmsk.shape[1],LEFT_TSmsk.shape[0]))
                          
                            minrLITS,mincLITS,ii,maxrLITS,maxcLITS,ii=ITSCLOTS[ITSPOTS[0]].bbox
                            minrLII_ITS,mincLII_ITS,ii,maxrLII_ITS,maxcLII_ITS,ii=ITSCLOTS[ITSPOTS[1]].bbox
                          
                            L_ITS=np.zeros((256,256))
                            L_ITS[minrLITS:maxrLITS,mincLITS:maxcLITS]=ITSCLOTS[ITSPOTS[0]].image[:,:,0]
                            L_ITS[minrLII_ITS:maxrLII_ITS,mincLII_ITS:maxcLII_ITS]=ITSCLOTS[ITSPOTS[1]].image[:,:,0]
                          
                            LEFT_ITS=cv2.resize(L_ITS,(LEFT_ITSmsk.shape[1],LEFT_ITSmsk.shape[0]))
                     
                         
                          
                            minrRTS,mincRTS,ii,maxrRTS,maxcRTS,ii=TSCLOTSII[TSPOTSII[0]].bbox
                            minrRIITS,mincRIITS,ii,maxrRIITS,maxcRIITS,ii=TSCLOTSII[TSPOTSII[1]].bbox
                         
                            R_TS=np.zeros((256,256))
                            R_TS[minrRTS:maxrRTS,mincRTS:maxcRTS]=TSCLOTSII[TSPOTSII[0]].image[:,:,0]
                            R_TS[minrRIITS:maxrRIITS,mincRIITS:maxcRIITS]=TSCLOTSII[TSPOTSII[1]].image[:,:,0]
                         
                            RIGHT_TS=cv2.resize(R_TS,(RIGHT_TSmsk.shape[1],RIGHT_TSmsk.shape[0]))
                         
                            minrRITS,mincRITS,ii,maxrRITS,maxcRITS,ii=ITSCLOTSII[ITSPOTSII[0]].bbox
                            minrRII_ITS,mincRII_ITS,ii,maxrRII_ITS,maxcRII_ITS,ii=ITSCLOTSII[ITSPOTSII[1]].bbox
                         
                            R_ITS=np.zeros((256,256))
                            R_ITS[minrRITS:maxrRITS,mincRITS:maxcRITS]=ITSCLOTSII[ITSPOTSII[0]].image[:,:,0]
                            R_ITS[minrRII_ITS:maxrRII_ITS,mincRII_ITS:maxcRII_ITS]=ITSCLOTSII[ITSPOTSII[1]].image[:,:,0]
                         
                            RIGHT_ITS=cv2.resize(R_ITS,(RIGHT_ITSmsk.shape[1],RIGHT_ITSmsk.shape[0]))
                     
                        
                          
                         

                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'Colours'+'/'+file+'_'+str(i)+'.png',np.uint8(FRAME))
                     
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(LeftPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_ITS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(RightPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_ITS))

                        else:
                            
                            LeftPaw=cv2.cvtColor(RotatedII, cv2.COLOR_BGR2RGB)                        
                            LEFT_TSmsk=cv2.resize(PII_TSmsk,(RotatedII.shape[1],RotatedII.shape[0]))                        
                            LEFT_ITSmsk=cv2.resize(PII_ITSmsk,(RotatedII.shape[1],RotatedII.shape[0]))                        
                            RightPaw=cv2.cvtColor(RotatedI, cv2.COLOR_BGR2RGB)                                                
                            RIGHT_TSmsk=cv2.resize(PI_TSmsk,(RotatedI.shape[1],RotatedI.shape[0]))                        
                            RIGHT_ITSmsk=cv2.resize(PI_ITSmsk,(RotatedI.shape[1],RotatedI.shape[0]))
                            
                            DTSI=np.sqrt(np.diff(Original_X_TS)*np.diff(Original_X_TS)+np.diff(Original_Y_TS)*np.diff(Original_Y_TS))
                            DTSII=np.sqrt(np.diff(Original_X_TSII)*np.diff(Original_X_TSII)+np.diff(Original_Y_TSII)*np.diff(Original_Y_TSII))
                            DITSI=np.sqrt(np.diff(Original_X_ITS)*np.diff(Original_X_ITS)+np.diff(Original_Y_ITS)*np.diff(Original_Y_ITS))
                            DITSII=np.sqrt(np.diff(Original_X_ITSII)*np.diff(Original_X_ITSII)+np.diff(Original_Y_ITSII)*np.diff(Original_Y_ITSII))
                            
                                            
                            
                            TS_l[i]=DTSII
                            ITS_l[i]=DITSII
                            
                            TS_r[i]=DTSI
                            ITS_r[i]=DITSI
                                               
    
                            SAVING_MK[:,:,i]=LMASK
                    
                            AreaI=RI[0].area
                            AreaII=RII[0].area
                            
                            Ratio_area[i]=1-(np.abs(AreaII-AreaI)/AreaI)                        
    
                            if np.isnan(Ratio_area[i]):
                                plt.pause(0.1)                          
                            FRAME=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            #  Left Blue
                            FRAME[ResLMASKII>0,:]=[200,0,0]
                            # Right Red
                            FRAME[ResLMASKI>0,:]=[0,0,200]
                            minrLTS,mincLTS,ii,maxrLTS,maxcLTS,ii=TSCLOTSII[TSPOTSII[0]].bbox
                            minrLIITS,mincLIITS,ii,maxrLIITS,maxcLIITS,ii=TSCLOTSII[TSPOTSII[1]].bbox
                            
                            L_TS=np.zeros((256,256))
                            L_TS[minrLTS:maxrLTS,mincLTS:maxcLTS]=TSCLOTSII[TSPOTSII[0]].image[:,:,0]
                            L_TS[minrLIITS:maxrLIITS,mincLIITS:maxcLIITS]=TSCLOTSII[TSPOTSII[1]].image[:,:,0]
                            
                            LEFT_TS=cv2.resize(L_TS,(LEFT_TSmsk.shape[1],LEFT_TSmsk.shape[0]))
                            
                            minrLITS,mincLITS,ii,maxrLITS,maxcLITS,ii=ITSCLOTSII[ITSPOTSII[0]].bbox
                            minrLII_ITS,mincLII_ITS,ii,maxrLII_ITS,maxcLII_ITS,ii=ITSCLOTSII[ITSPOTSII[1]].bbox
                            
                            L_ITS=np.zeros((256,256))
                            L_ITS[minrLITS:maxrLITS,mincLITS:maxcLITS]=ITSCLOTSII[ITSPOTSII[0]].image[:,:,0]
                            L_ITS[minrLII_ITS:maxrLII_ITS,mincLII_ITS:maxcLII_ITS]=ITSCLOTSII[ITSPOTSII[1]].image[:,:,0]
                            
                            LEFT_ITS=cv2.resize(L_ITS,(LEFT_ITSmsk.shape[1],LEFT_ITSmsk.shape[0]))

                           
                            try:
                                minrRTS,mincRTS,ii,maxrRTS,maxcRTS,ii=TSCLOTS[TSPOTS[0]].bbox
                                minrRIITS,mincRIITS,ii,maxrRIITS,maxcRIITS,ii=TSCLOTS[TSPOTS[1]].bbox
                                R_TS=np.zeros((256,256))
                                R_TS[minrRTS:maxrRTS,mincRTS:maxcRTS]=TSCLOTS[TSPOTS[0]].image[:,:,0]
                                R_TS[minrRIITS:maxrRIITS,mincRIITS:maxcRIITS]=TSCLOTS[TSPOTS[1]].image[:,:,0]
                            except: 
                                plt.pause(0.1)
                                
                           
                           
                            RIGHT_TS=cv2.resize(R_TS,(RIGHT_TSmsk.shape[1],RIGHT_TSmsk.shape[0]))
                           
                            minrRITS,mincRITS,ii,maxrRITS,maxcRITS,ii=ITSCLOTS[ITSPOTS[0]].bbox
                            minrRII_ITS,mincRII_ITS,ii,maxrRII_ITS,maxcRII_ITS,ii=ITSCLOTS[ITSPOTS[1]].bbox
                           
                            R_ITS=np.zeros((256,256))
                            R_ITS[minrRITS:maxrRITS,mincRITS:maxcRITS]=ITSCLOTS[ITSPOTS[0]].image[:,:,0]
                            R_ITS[minrRII_ITS:maxrRII_ITS,mincRII_ITS:maxcRII_ITS]=ITSCLOTS[ITSPOTS[1]].image[:,:,0]
                           
                            RIGHT_ITS=cv2.resize(R_ITS,(RIGHT_ITSmsk.shape[1],RIGHT_ITSmsk.shape[0]))


                           
                           # plt.imshow(FRAME)
                           # plt.pause(0.1)

                           # out.write(FRAME)
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'Colours'+'/'+file+'_'+str(i)+'.png',np.uint8(FRAME))

                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(LeftPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'LEFT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*LEFT_ITS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'RAW'+'/'+file+'_'+str(i)+'.png',np.uint8(RightPaw))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'TSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_TS))
                            cv2.imwrite('CHECKING_RESULTS'+'/'+file[:-4]+'/'+'RIGHT'+'/'+'ITSMASKS'+'/'+file+'_'+str(i)+'.png',np.uint8(255*RIGHT_ITS))
        
        
        rt,frame=cap.read()
        i+=1  
        Break_clause=np.sum(np.uint(~np.isnan(TS_r)))
        end=time.time()
        # print(end-start)
        
        if Break_clause>=1000:
            break
        
        # if i>=1000:
        #     break
        
        
         
    return TS_l, ITS_l, TS_r, ITS_r, SAVING_MK, Ratio_area
            

            




def main(CROP,folder_selected):
    
    LOG=[]
    
    Folder='CHECKING_RESULTS'
    
    Main_folder_all_files = os.listdir(folder_selected)
    
    
    for f in Main_folder_all_files:
        if f.startswith('.'):
          Main_folder_all_files.remove(f)
          
    # for f in Main_folder_all_files:
    #     if (f[0]!='D'):
    #       Main_folder_all_files.remove(f)
    try:
            os.mkdir(Folder)
    except OSError:
            print('Already exists')
            

    

 
    
    
    
    model=unet()
    model.load_weights('PAWS900.h5')
    
    TSmodel=unet()
    TSmodel.load_weights('TS10000.h5')
    
    ITSmodel=unet()
    ITSmodel.load_weights('ITS10000.h5')
    
    # CROP=joblib.load('CropALL_vic.joblib')
    # CROP=dict()
    FRAMES=dict()
    TS_LEFT=dict()
    TS_RIGHT=dict()
    ITS_LEFT=dict()
    ITS_RIGHT=dict()
    TSF=dict()
    ITSF=dict()
    MASK=dict()
    Ratio_area=dict()
    OUTPUTS=Folder+'/'+'OUTPUTS'
    
    
    try:
        os.mkdir(OUTPUTS)
    except OSError:
        print('Already exists')
    

    
    for f in Main_folder_all_files:
        if f.startswith('.'):
          Main_folder_all_files.remove(f)
        if f=='cropping.joblib':
          Main_folder_all_files.remove(f)

    All_DONE = os.listdir(Folder+'/'+'OUTPUTS')
    
    for g in All_DONE:
        for f in Main_folder_all_files:
            if g[:-5]==f[:-4]:
                Main_folder_all_files.remove(f)
    
    
    # for f in All_DONES:
    #     Main_folder_all_files.remove(f)
    

    i=0
    for f in Main_folder_all_files:
    
        # MASK=dict()
        # OUT=dict()

    
           try:
                os.mkdir(Folder+'/'+f[:-4])
           except OSError:
                print('Already exists')
            
        
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'RIGHT')
           except OSError:
                print('Already exists')    
                
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'RIGHT'+'/'+'TSMASKS')
           except OSError:
                print('Already exists')
                
            
            
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'RIGHT'+'/'+'ITSMASKS')
           except OSError:
                print('Already exists')
            
        
            
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'RIGHT'+'/'+'RAW')
           except OSError:
                print('Already exists')
                
                
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'LEFT')
           except OSError:
                print('Already exists')    
                
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'LEFT'+'/'+'TSMASKS')
           except OSError:
                print('Already exists')
                
            
            
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'LEFT'+'/'+'ITSMASKS')
           except OSError:
                print('Already exists')
            
        
            
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'LEFT'+'/'+'RAW')
           except OSError:
                print('Already exists')
    
           try:
                os.mkdir(Folder+'/'+f[:-4]+'/'+'Colours')
           except OSError:
                print('Already exists')
           VIDEO_rotate=0
           file=f
           print(file)
           filename=folder_selected+'/'+file
           TS_LEFT[file[:-4]], ITS_LEFT[file[:-4]], TS_RIGHT[file[:-4]], ITS_RIGHT[file[:-4]],MASK[file[:-4]],Ratio_area[file[:-4]]=Measurement(filename,CROP[filename],file,VIDEO_rotate,model, TSmodel,ITSmodel)
           joblib.dump(TS_LEFT, Folder+'/'+'TS_LEFT_VIC.joblib')
           joblib.dump(TS_RIGHT, Folder+'/'+'TS_RIGHT_VIC.joblib')
           joblib.dump(ITS_LEFT, Folder+'/'+'ITS_LEFT_VIC.joblib')
           joblib.dump(ITS_RIGHT, Folder+'/'+'ITS_RIGHT_VIC.joblib')
            
            
           TSr=TS_RIGHT[file[:-4]]
           ITSr=ITS_RIGHT[file[:-4]]
           TSl=TS_LEFT[file[:-4]]
           ITSl=ITS_LEFT[file[:-4]]
        
            
           OTSr=TS_RIGHT[file[:-4]]
           OITSr=ITS_RIGHT[file[:-4]]
           OTSl=TS_LEFT[file[:-4]]
           OITSl=ITS_LEFT[file[:-4]]
            
        
            
           TSr=TSr[~np.isnan(TSr)]
           ITSr=ITSr[~np.isnan(ITSr)]
           TSl=TSl[~np.isnan(TSl)]
           ITSl=ITSl[~np.isnan(ITSl)]
            
            
            
           Filter_r=np.uint(TSr>ITSr) 
           Filter_l=np.uint(TSl>ITSl)
            
           TSr=TSr[Filter_r>0]
           ITSr=ITSr[Filter_r>0]
           TSl=TSl[Filter_l>0]
           ITSl=ITSl[Filter_l>0]
            
           Q_TSr=np.percentile(TSr,[75,25])
           Q_ITSr=np.percentile(ITSr,[75,25])
           Q_TSl=np.percentile(TSl,[75,25])
           Q_ITSl=np.percentile(ITSl,[75,25])
            
           TSr=TSr[(np.uint(TSr<Q_TSr[0])*np.uint(TSr>Q_TSr[1]))>0]
           ITSr=ITSr[(np.uint(ITSr<Q_ITSr[0])*np.uint(ITSr>Q_ITSr[1]))>0]
           TSl=TSl[(np.uint(TSl<Q_TSl[0])*np.uint(TSl>Q_TSl[1]))>0]
           ITSl=ITSl[(np.uint(ITSl<Q_ITSl[0])*np.uint(ITSl>Q_ITSl[1]))>0]
            
           TSF=np.ones(100)*np.nan
           ITSF=np.ones(100)*np.nan
           SSI=np.ones(100)*np.nan
            
           for sample in range(0,100):
                Per_tsr=np.random.permutation(TSr)
                Per_itsr=np.random.permutation(ITSr)
                Per_tsl=np.random.permutation(TSl)
                Per_itsl=np.random.permutation(ITSl)
                
                M_tsr=np.mean(Per_tsr[:15])
                M_tsl=np.mean(Per_tsl[:15])
                M_itsr=np.mean(Per_itsr[:15])
                M_itsl=np.mean(Per_itsl[:15])
                
                TSF[sample]=(M_tsl-M_tsr)/M_tsr
                ITSF[sample]=(M_itsl-M_itsr)/M_itsr
                SSI[sample]= 108.44*TSF[sample]+31.85*ITSF[sample]-5.49
                
                
            
           workbook = xlsxwriter.Workbook(OUTPUTS+'/'+file[:-4]+'.xlsx')
           worksheet = workbook.add_worksheet()
             
              # Add a bold format to use to highlight cells.
           bold = workbook.add_format({'bold': True})
             
             
              # Write some data headers.
           worksheet.write('B3', 'TS control', bold)
           worksheet.write('C3', 'TS injury', bold)
           worksheet.write('D3', 'TSF', bold)
           worksheet.write('E3', 'ITS control', bold)
           worksheet.write('F3', 'ITS injury', bold)
           worksheet.write('G3', 'ITSF', bold)
           worksheet.write('H3', 'SSI', bold)
            
            
           col=1
           row=3
            
            # DATA=[]
            
            
           worksheet.write_column(3,1,TSr)
           worksheet.write_column(3,2,TSl)
           worksheet.write_column(3,3,TSF)
            
           worksheet.write_column(3,4,ITSr)
           worksheet.write_column(3,5,ITSl)
           worksheet.write_column(3,6,ITSF)
            
           worksheet.write_column(3,7,SSI)
                
                
           workbook.close()
            
           Filter=np.sum(np.uint(~np.isnan(TS_RIGHT[file[:-4]])))

           if  Filter>= 50:
               Doc="Everything went well with" + file[:-4]
               LOG.append(Doc)
           else:
               Doc="Check if everything is OK with" + file[:-4]
               LOG.append(Doc)
        
           CHECKING=pd.DataFrame(LOG, columns=["colummn"])
           CHECKING.to_csv("LOGGING.csv",index=False)

           i+=1    