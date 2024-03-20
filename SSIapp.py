#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:06:23 2022

@author: Simao
"""
from tkinter import (ttk,Tk,PhotoImage,Canvas, filedialog, colorchooser,RIDGE,
                     GROOVE,ROUND,Scale,HORIZONTAL)
import tkinter as tk
import cv2
from PIL import ImageTk, Image,ImageDraw
import numpy as np
import glob
# import imutils
# import joblib
# import keras_preprocessing
# import tensorflow
# import xlsxwriter
import os
import matplotlib.pyplot as plt
from PAWS_forAPP import main
import joblib
class FrontEnd:
    def __init__(self, master):
        self.master = master
        self.menu_initialisation()
        
    def menu_initialisation(self):
        self.master.geometry('750x360+250+10')
        self.master.title('SSI App Windowing')
        
        self.frame_header = ttk.Frame(self.master)
        self.frame_header.pack()        
        
        self.frame_menu = ttk.Frame(self.master)
        self.frame_menu.pack()
        self.frame_menu.config(relief=RIDGE, padding=(50, 15))
        
        
        self.frame_menu = ttk.Frame(self.master)
        self.frame_menu.pack()
        self.frame_menu.config(relief=RIDGE, padding=(50, 15))
        
        
        ttk.Label(
            self.frame_header, text="File Name").grid(row=0,column=0)
        
        
        
        self.ImageII=dict()
        self.ImageI=dict()
        self.CROP=dict()
        
        ttk.Button(
            self.frame_menu, text="Folder", command=self.upload_action).grid(
            row=0, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

                


        self.FRAME=tk.Entry(self.frame_menu, bg='grey')
        self.FRAME.place(x=0,y=145, width=100,height=30)
        
        
        ttk.Button(
            self.frame_menu, text="Frame #", command=self.Reading).grid(
            row=8, column=0, columnspan=2, padx=5, pady=5, sticky='sw')
                
        ttk.Button(
            self.frame_menu, text="All done", command=self.CNN).grid(
            row=9, column=0, columnspan=2, padx=5, pady=5, sticky='sw')
        
        self.canvas = Canvas(self.frame_menu, bg="gray", width=256, height=256)
        self.canvas.grid(row=0, column=3, rowspan=10)
        
        self.canvasII = Canvas(self.frame_menu, bg="gray", width=256, height=256)
        self.canvasII.grid(row=0, column=4, rowspan=10)
        
        self.side_frame = ttk.Frame(self.frame_menu)
        self.side_frame.grid(row=0, column=4, rowspan=10)
        self.side_frame.config(relief=GROOVE, padding=(50,15))
        
        
        self.apply_and_cancel = ttk.Frame(self.master)
        self.apply_and_cancel.pack()
        self.apply = ttk.Button(self.apply_and_cancel, text="<<", command=self.backward).grid(
            row=0, column=3, columnspan=1, padx=5, pady=5)

        ttk.Button(
            self.apply_and_cancel, text="         ", command=self.revert_action).grid(
                row=0, column=4, columnspan=1,padx=5, pady=5)

        ttk.Button(
            self.apply_and_cancel, text=">>", command=self.foward).grid(
                row=0, column=5, columnspan=1,padx=0, pady=5)
    
    def Reading(self):
        frame_number=float(self.FRAME.get())
                  
        cap = cv2.VideoCapture(self.files[self.count])
        length = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(0, frame_number)
        # Take first non-null frame and find corners within it
        ret, frame = cap.read()
        frame = cv2.resize(frame,(256, 256))
        self.filtered_image=frame
        self.ImageI[self.files[self.count]]=frame.copy()
        self.display_image(self.ImageI[self.files[self.count]])
        return
        
        
        
    def upload_action(self):
        self.canvas.delete("all")
        self.folder_selected = filedialog.askdirectory()
        self.list_of_points=0
        types=('*.MP4','*.mp4','*.MOV','*.mov')
        i=0
        self.count=0
        for TYPE in types:
        
            FILES=glob.glob(self.folder_selected+'/'+TYPE)
            CROPEXIST=glob.glob(self.folder_selected+'/'+'*.joblib')
            if CROPEXIST:
                self.CROP=joblib.load(self.folder_selected+'/'+'cropping.joblib')
            if FILES:
                self.files=glob.glob(self.folder_selected+'/'+TYPE)
                for f in FILES:
                   cap = cv2.VideoCapture(f)
                   length = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                   cap.set(0, np.uint(length/2))
               
               
                   # Take first non-null frame and find corners within it
                   ret, frame = cap.read()
                   frame = cv2.resize(frame,(256, 256))
                   self.ImageI[f]=frame
                   
                   if CROPEXIST:
                       img = Image.new('L', (256, 256), 0)
                       ImageDraw.Draw(img).polygon(self.CROP[f], outline=1, fill=1)
                       mask = np.array(img)
                       cropping=self.ImageI[f].copy()
                       cropping[mask==0,:]=0
                       # self.edited_image=image
                       self.ImageII[f]= cropping.copy()
                       
                   else:
                       self.ImageII[f]=[]
                       self.CROP[f]=[]
                   

         
                # self.do_not_touch=img1
                # my_label.grid_forget()
        self.original_image=self.ImageI[self.files[self.count]].copy()
        self.edited_image =self.original_image.copy()
        self.filtered_image = self.original_image.copy()
        self.display_image(self.filtered_image) 
        if CROPEXIST:
            self.display_image_crop(self.ImageII[self.files[self.count]]) 

        self.all_files = os.listdir(self.folder_selected)
        
        for f in self.all_files:
           if f.startswith('.'):
             self.all_files.remove(f)
        ttk.Label(
        self.frame_header, text=self.all_files[self.count]).grid(row=0,column=0)
        
        
        
        self.list_of_points=[]
        self.poly = None
        self.canvas.bind("<ButtonPress>", self.start_crop)
        self.canvas.bind("<Motion>", self.crop)

    
    def foward(self):
        self.canvas.delete('all')
        self.canvasII.delete('all')
        if self.count>=0 and self.count<len(self.files):
            self.count+=1
        self.list_of_points=[]
        cap = cv2.VideoCapture(self.files[self.count])
        length = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(0, np.uint(length/2))        

        # Take first non-null frame and find corners within it
        if self.ImageI[self.files[self.count]].all():
            self.filtered_image=self.ImageI[self.files[self.count]]
        else:
            ret, frame = cap.read()
            frame = cv2.resize(frame,(256, 256))
            self.filtered_image=frame
            self.ImageI[self.files[self.count]]=frame
            
        self.display_image(self.filtered_image)
        ttk.Label(
                self.frame_header, text=self.all_files[self.count]).grid(row=0,column=0)
        if len(self.ImageII[self.files[self.count]])>0:
            self.display_image_crop(self.ImageII[self.files[self.count]])
        if len(self.CROP[self.files[self.count]])>0:
            self.list_of_points=self.CROP[self.files[self.count]]
        return
    
    def  backward(self):
        self.canvas.delete('all')
        
        self.canvasII.delete('all')
        if self.count>0 and self.count<=len(self.files):
            self.count-=1
        self.list_of_points=[]
        cap = cv2.VideoCapture(self.files[self.count])
        length = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(0, np.uint(length/2))        

    
        if self.ImageI[self.files[self.count]].any():
            self.filtered_image=self.ImageI[self.files[self.count]]
        else:
            ret, frame = cap.read()
            frame = cv2.resize(frame,(256, 256))
            self.filtered_image=frame
        self.display_image(self.filtered_image)
        
        ttk.Label(
                self.frame_header, text=self.all_files[self.count]).grid(row=0,column=0)
        if len(self.ImageII[self.files[self.count]])>0:
            self.display_image_crop(self.ImageII[self.files[self.count]])
        if len(self.CROP[self.files[self.count]])>0:
                self.list_of_points=self.CROP[self.files[self.count]]
        return
   
    def revert_action(self):
        
        
        
     return
    
    def display_image(self, image=None):
        self.canvas.delete("all")
        if image is None:
            image = self.ImageI[self.files[self.count]]
        else:
            image = image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.height, self.width, channels = image.shape
        ratio = self.height / self.width

        new_width = self.width
        new_height = self.height

        # if height > 400 or width > 300:
        #     if ratio < 1:
        #         new_width = 300
        #         new_height = int(new_width * ratio)
        #     else:
        #         new_height = 400
        #         new_width = int(new_height * (width / height))

        self.ratio = self.height / new_height
        self.new_image = cv2.resize(image, (new_width, new_height))

        self.new_image = ImageTk.PhotoImage(
            Image.fromarray(self.new_image))

        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(
            new_width / 2, new_height / 2,  image=self.new_image)
    
    def display_image_crop(self, image=None):
        self.canvasII.delete("all")
        if image is None:
            image = self.ImageII[self.files[self.count]]
        else:
            image = image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        ratio = height / width

        new_width = self.width
        new_height = self.height

        self.ratio_crop = new_width / new_height
        self.new_image_crop = cv2.resize(image, (new_width, new_height))
        self.ImageII[self.files[self.count]]= self.new_image_crop.copy()

        self.new_image_crop = ImageTk.PhotoImage(
            Image.fromarray(self.ImageII[self.files[self.count]]))
        
        self.canvasII.config(width=new_width, height=new_height)
        self.canvasII.create_image(
            new_width / 2, new_height / 2,  image=self.new_image_crop)
        return
        # if len(self.list_of_points>=5):
            


        
    def start_crop(self, event):
        # self.canvas.delete('all')
        # self.display_image(self.ImageI[self.files[self.count]])
        # self.rootine_desplay(self.filtered_image)
        mouse_xy = (event.x, event.y)
        center_x, center_y = mouse_xy
        self.list_of_points.append((center_x, center_y))
    
        for pt in self.list_of_points:
            x, y =  pt
            #draw dot over position which is clicked
            x1, y1 = (x - 1), (y - 1)
            x2, y2 = (x + 1), (y + 1)
            self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
        # add clicked positions to list
    
    
        numberofPoint=len(self.list_of_points)
        
        if numberofPoint>=2 and numberofPoint<5:
              mouse_xy = (event.x, event.y)
              center_x, center_y = mouse_xy
          
              for pt in self.list_of_points:
                  x, y =  pt
                  #draw dot over position which is clicked
                  x1, y1 = (x - 1), (y - 1)
                  x2, y2 = (x + 1), (y + 1)
                  self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
              x0,y0=self.list_of_points[0]
              x1,y1=self.list_of_points[1]
              
              poly=self.canvas.create_polygon(self.list_of_points, fill='', outline='green', width=2)
        elif numberofPoint>=5:
            self.canvas.delete('all')
            self.canvasII.delete('all')
            # cap = cv2.VideoCapture(self.files[self.count])
            # length = np.int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # cap.set(0, np.uint(length/2))
            # # Take first non-null frame and find corners within it
            # ret, frame = cap.read()
            # frame = cv2.resize(frame,(256, 256))
            self.display_image(self.ImageI[self.files[self.count]])
            self.filtered_image=self.ImageI[self.files[self.count]]
            self.list_of_points=[]
            


    def crop(self, event):
        numberofPoint=len(self.list_of_points)
    
        if numberofPoint==1:
            self.display_image(self.ImageI[self.files[self.count]])


            mouse_xy = (event.x, event.y)
            center_x, center_y = mouse_xy
            # list_of_points[1]=((center_x, center_y))
        
            for pt in self.list_of_points:
                x, y =  pt
                #draw dot over position which is clicked
                x1, y1 = (x - 1), (y - 1)
                x2, y2 = (x + 1), (y + 1)
                self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
            x0,y0=self.list_of_points[0]
            self.canvas.create_line(x0,y0,mouse_xy[0],mouse_xy[1])
            
    
     
        if numberofPoint==2:
              self.display_image(self.ImageI[self.files[self.count]])

              mouse_xy = (event.x, event.y)
              center_x, center_y = mouse_xy
          
              for pt in self.list_of_points:
                  x, y =  pt
                  #draw dot over position which is clicked
                  x1, y1 = (x - 1), (y - 1)
                  x2, y2 = (x + 1), (y + 1)
                  self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
              x0,y0=self.list_of_points[0]
              x1,y1=self.list_of_points[1]
              
              poly=self.canvas.create_polygon(x0,y0,x1,y1,mouse_xy[0],mouse_xy[1], fill='', outline='green', width=2)
        
        
        if numberofPoint==3:
            self.display_image(self.ImageI[self.files[self.count]])
       
            mouse_xy = (event.x, event.y)
            center_x, center_y = mouse_xy
        
            for pt in self.list_of_points:
                x, y =  pt
                #draw dot over position which is clicked
                x1, y1 = (x - 1), (y - 1)
                x2, y2 = (x + 1), (y + 1)
                self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
            x0,y0=self.list_of_points[0]
            x1,y1=self.list_of_points[1]
            x2,y2=self.list_of_points[2]
            poly=self.canvas.create_polygon(x0,y0,x1,y1,x2,y2,mouse_xy[0],mouse_xy[1], fill='', outline='green', width=2)
       
        if numberofPoint==4:
            # self.display_image(self.ImageI[self.files[self.count]])

            mouse_xy = (event.x, event.y)
            center_x, center_y = mouse_xy
        
            for pt in self.list_of_points:
                x, y =  pt
                #draw dot over position which is clicked
                x1, y1 = (x - 1), (y - 1)
                x2, y2 = (x + 1), (y + 1)
                self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
            x0,y0=self.list_of_points[0]
            x1,y1=self.list_of_points[1]
            x2,y2=self.list_of_points[2]
            x3,y3=self.list_of_points[3]
            poly=self.canvas.create_polygon(x0,y0,x1,y1,x2,y2,x3,y3, fill='', outline='green', width=2)
            self.end_crop()
            

                # self.canvas.create_oval(x1, y1, x2, y2, fill='green', outline='green', width=5)
                # self.list_of_points.append((center_x, center_y))
        
    def end_crop(self):
            img = Image.new('L', (256, 256), 0)
            ImageDraw.Draw(img).polygon(self.list_of_points, outline=1, fill=1)
            mask = np.array(img)
            
            self.CROP[self.files[self.count]]=self.list_of_points
            
            # self.filtered_image = self.edited_image[y, x]
            # self.polygon = self.canvas.create_polygon(start_x, start_y, start_x + (end_x-start_x), start_y + (end_y-start_y),\
            #     outline='red')
            cropping=self.ImageI[self.files[self.count]].copy()
            
            
            cropping[mask==0,:]=0
            # self.edited_image=image
            self.ImageII[self.files[self.count]]= cropping
            self.display_image_crop(self.ImageII[self.files[self.count]])
  
    def CNN(self):
        mainWindow.quit()
        joblib.dump(self.CROP,self.folder_selected+'/'+'cropping.joblib')
        main(self.CROP, self.folder_selected)
        something=1
        
        
mainWindow = Tk()
FrontEnd(mainWindow)
mainWindow.mainloop()