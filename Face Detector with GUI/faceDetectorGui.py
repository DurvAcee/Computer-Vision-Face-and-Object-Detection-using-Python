import cv2
from tkinter import *
from tkinter import filedialog
from tkinter import font

from PIL import Image, ImageTk

def faceDetGui():

    root = Tk()
    root.geometry("620x705")  
    root.title('Face Detector')
    root.configure(bg="black")

    boldFont = font.Font(family='Helvetica', size=12, weight='bold')
    boldFont1 = font.Font(family='Helvetica', size=10, weight='bold')

    var = StringVar()
    var2 = StringVar()
    label = Label(root, anchor=CENTER, textvariable = var, relief = RAISED, width=240,font = boldFont)
    label2 = Label(root, anchor=CENTER, textvariable = var2, relief = RAISED, width=240,font = boldFont)
    var.set(" Welcome To Face Detector! ")
    label.pack(pady=10)

    img = Image.open("fd.jpg")
    load = img.resize((450, 450), Image.ANTIALIAS)
    render = ImageTk.PhotoImage(load)
    img = Label(image=render)
    img.image = render
    img.pack()
    # img.place(x=1, y=1)

    var2.set("Select any image to detect faces")
    label2.pack(pady=10)

    # filename = Entry(root, width=75)
    # filename.pack()
    
    def open_file():
        global file 
        file =  filedialog.askopenfilename(initialdir = "/", title = "Select file")
        # print(file)

    def detect():
        face_cascade = cv2.CascadeClassifier("C:/Users/DURVESH/Desktop/Voice Assistant/OpenCV2/haarcascade_frontalface_default.xml")
        orgImg = cv2.imread(file)
        img = cv2.imread(file)
        gray_img = cv2.imread(file,0)

        faces = face_cascade.detectMultiScale(gray_img,
        scaleFactor=1.15,
        minNeighbors=5)

        for x,y,w,h in faces:
            img=cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),5)

        resized_img = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))
        resized_orImg = cv2.resize(orgImg,(int(orgImg.shape[1]/3),int(orgImg.shape[0]/3)))
        cv2.imshow("Face Detection",resized_img)
        cv2.imshow("Original Image",resized_orImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

     
    btn1 = Button(root, text ='Choose File', width = 20, command = lambda:open_file(),bg="blue",fg="white",activebackground="white",font = boldFont1)
    btn1.pack(side = TOP, pady = 20)

    btn2 = Button(root, text ='Detect Now', width=25, command = lambda:detect(),bg="blue",fg="white",activebackground="white",font = boldFont1)
    btn2.pack(side = TOP)

    root.mainloop()

faceDetGui()







