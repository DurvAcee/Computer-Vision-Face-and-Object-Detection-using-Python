from tkinter import ttk
import tkinter as tk
from tkinter import font
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
from tkinter.messagebox import showinfo
from imageai import Detection
from imageai.Detection import ObjectDetection
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# Setting the Window
root = tk.Tk()
root.geometry("620x685")  
root.title('Computer Vision')

boldFont = font.Font(family='Helvetica', size=12, weight='bold')
boldFont1 = font.Font(family='Helvetica', size=10, weight='bold')

# Styling notebook
style = ttk.Style()
# style.configure("BW.TLabel", foreground="white", background="black")
style.theme_create('pastel', settings={
    ".": {
        "configure": {
            "background": 'black', # All except tabs
        }
    },
    "TNotebook": {
        "configure": {
            "background":'white', # Your margin color
            "tabmargins": [5, 4, 5, 1], # margins: left, top, right, separator
        }
    },
    "TNotebook.Tab": {
        "configure": {
            "background": 'white', # tab color when not selected
        },
        "map": {
            "background": [("selected", '#ccffff')], # Tab color when selected
            "expand": [("selected", [1, 1, 1, 0])] # text margins
        }
    }
})
 
style.theme_use('pastel')


tab_parent = ttk.Notebook(root)
tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)
tab_parent.add(tab1, text="Face Detection")
tab_parent.add(tab2, text="Object Detection")
tab_parent.pack(expand=1, fill='both')


# Widgets for Tab 1
firstLabelTabOne = tk.Label(tab1, anchor=CENTER, font = boldFont, text="Welcome To Face Detection! ",bg='#000000',fg='white')
secondLabelTabOne = tk.Label(tab1, anchor=CENTER, font = boldFont, text="Select any image to Detect Faces",bg='#000000',fg='white')


button1 = tk.Button(tab1, text="Choose File", width = 20, command = lambda:open_file(),bg="blue",fg="white",activebackground="white",font = boldFont1)
button2 = tk.Button(tab1, text="Detect Now", width = 25, command = lambda:detect(),bg="blue",fg="white",activebackground="white",font = boldFont1)

tab_parent.pack(expand=1, fill='both')

img = Image.open("img/fd1.jpg")
load = img.resize((500, 450), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = tk.Label(tab1,image=render)
img.image = render


# Functions
def open_file():
    global file 
    file =  filedialog.askopenfilename(initialdir = "/", title = "Select file")
    if len(file)!=0 and file[-3:] == "jpg" :
        popup_showinfo()

    elif len(file)!=0 and file[-3:] != "jpg":
        popup_showinfo3()
    else:
        popup_showinfo2()   

def popup_showinfo():
    showinfo("Windows", "File Selected!\n Press Ok to continue")

def popup_showinfo2():
    showinfo("Windows", "File not Selected!\n Please select a file to continue")

def popup_showinfo3():
    showinfo("Windows", "The Selected File is not compatible \n Please select a .jpg file")


def detect():

    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    orgImg = cv2.imread(file)
    img = cv2.imread(file)
    gray_img = cv2.imread(file,0)
    faces = face_cascade.detectMultiScale(gray_img,
    scaleFactor=1.25,
    minNeighbors=4)
    for x,y,w,h in faces:
        img=cv2.rectangle(img, (x,y),(x+w,y+h),	(255,200,120),5)

    resized_img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
    resized_orImg = cv2.resize(orgImg,(int(orgImg.shape[1]/3),int(orgImg.shape[0]/3)))
    cv2.imshow("Face Detection",resized_img)
    cv2.imshow("Original Image",resized_orImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectObj():

    detector = ObjectDetection()

    model_path = "models/yolo-tiny.h5"
    input_path = file
    output_path = "output/newimage.jpg"

    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path)
    detector.loadModel()
    global detection
    detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

    org_img = cv2.imread(file)
    new_img = cv2.imread("output/newimage.jpg")
    resized_img = cv2.resize(new_img,(int(new_img.shape[1]/1.15),int(new_img.shape[0]/1.15)))
    resized_orgImg = cv2.resize(org_img,(int(org_img.shape[1]/1.15),int(org_img.shape[0]/1.15)))
    cv2.imshow("Original Image",resized_orgImg)
    cv2.imshow("Object Detection",resized_img)

    print("Detected Objects from the Image :\n\n")
    for eachItem in detection:  
        print(eachItem["name"] , " : ", eachItem["percentage_probability"])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

# Add Widgets for Tab 1 
firstLabelTabOne.grid(row=0, column=0, padx=15, pady=5)
img.grid(row=1,column=0,padx=60, pady=10)
secondLabelTabOne.grid(row=2, column=0, padx=10, pady=5)
button1.grid(row=3, column=0, padx=10, pady=5)
button2.grid(row=4, column=0, padx=10, pady=5)


# Widgets for Tab 2
firstLabelTabOne = tk.Label(tab2, anchor=CENTER, font = boldFont, text="Welcome to Object Detection! ",bg='#000000',fg='white')
secondLabelTabOne = tk.Label(tab2, anchor=CENTER, font = boldFont, text="Select any image to Detect Objects",bg='#000000',fg='white')


button1 = tk.Button(tab2, text="Choose File", width = 20, command = lambda:open_file(),bg="blue",fg="white",activebackground="white",font = boldFont1)
button2 = tk.Button(tab2, text="Detect Now", width = 25, command = lambda:detectObj(),bg="blue",fg="white",activebackground="white",font = boldFont1)

tab_parent.pack(expand=1, fill='both')

img = Image.open("img/od.jpg")
load = img.resize((450, 450), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = tk.Label(tab2,image=render)
img.image = render

# Add Widgets for Tab 2 
firstLabelTabOne.grid(row=0, column=0, padx=15, pady=5)
img.grid(row=1,column=0,padx=85, pady=10)
secondLabelTabOne.grid(row=2, column=0, padx=10, pady=5)
button1.grid(row=3, column=0, padx=10, pady=5)
button2.grid(row=4, column=0, padx=10, pady=5)


root.mainloop()

