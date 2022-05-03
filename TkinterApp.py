from turtle import left, pos
import imageio
import numpy as np

from skimage.transform import resize
import warnings
import sys
import cv2
import time
import pyautogui
import os,shutil
import torch
from scipy.spatial import ConvexHull
import tkinter as tk
from PIL import Image,ImageTk
from tkinter import ttk
from datetime import datetime as dt
import pytz
from tkinter import filedialog

ist=pytz.timezone('Asia/Kolkata')

# from tqdm import tqdm
warnings.filterwarnings("ignore")


root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
root.title("FakeFace")
root.iconbitmap("./buttons/myicon.ico")
root.minsize(640,480)
lmain = ttk.Label(root)
lmain.place(relheight=1.0,relwidth=1.0)
show=True


# style=ttk.Style()
# style.theme_use('alt')
# style.configure('TButton',background='black',foreground='white',borderwidth=0,)

########## setup ###########
stream =True
media_path="./media"
model_path='model/'

webcam_id=0
gpu_id=0
stream_id=1
system='win'

webcam_width=640
webcam_height=480
screen_width,screen_height=pyautogui.size()
img_shape=[256,256,0]

reset=True
previous=None



img_list = []
print("showing the available images.....")
for filename in os.listdir(media_path):
    if filename.endswith(".jpg") or filename.endswith('.jpeg') or filename.endswith('.png'):
        img_list.append(os.path.join(media_path,filename))
        print(filename)


######### end setup #######
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):       #removed tqdm(range())
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def process_image(source,base,curr,generator,kp_detector,relative):
    predictions = make_animation(source,[base,curr],generator,kp_detector,relative,adapt_movement_scale=False)
    return predictions[1]

def load_face_model():
    modelFile= f"{model_path}/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = f"{model_path}./deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile,modelFile)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)     ######## model is not built with cuda ##########
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def cut_face_window(x1,y1,x2,y2,frame):
    frame = frame.copy()
    frame = frame[y1:y2,x1:x2]
    face = resize(frame,(256,256))[...,:3]
    return face

# find the face in the webcam stream and center a 256x256 window
def find_face_cut(net,face):
    blob = cv2.dnn.blobFromImage(face,1.0,(300,300),[104,117,123],False,False)
    frameWidth = 640
    frameHeight = 480
    net.setInput(blob)
    detections = net.forward()
    face_found = False
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>0.9:
            x1 = (int(detections[0,0,i,3]*frameWidth)//2)*2
            y1 = (int(detections[0,0,i,4]*frameHeight)//2)*2
            x2 = (int(detections[0,0,i,5]*frameWidth)//2)*2
            y2 = (int(detections[0,0,i,6]*frameHeight)//2)*2
        
            face_margin_w = int(256-(abs(x1-x2)))
            face_margin_h = int(256 - (abs(y1-y2)))
            
            cut_x1 = x1-int(face_margin_w//2)
            cut_y1 = y1-int(2*face_margin_h//3)
            
            cut_x2 = x2+int(face_margin_w//2)
            cut_y2 = y2+face_margin_h-int(2*face_margin_h//3)
            
            face_found=True
            break
    
    if not face_found:
        print("No face detected in video")
        cut_x1,cut_y1,cut_x2,cut_y2 = 112,192,368,448
        # center of the image
    else:
        print(f'Found face at: ({x1},{y1}) ({x2},{y2}) width: {x2-x2} height :{y2-y1}')
        print(f'Cutting at: ({cut_x1,cut_y1}) ({cut_x2,cut_y2}) width: {(cut_x2-cut_x1)} height : {(cut_y2-cut_y1)}')
        
    return cut_x1,cut_y1,cut_x2,cut_y2

def readimage():
    global img_list,img_shape,source_image,reset
    img = imageio.imread(img_list[pos])
    img = resize(img,(256,256))[...,:3]
    source_image=img
    reset=True
    return img

def readpreviousimage():
    global pos
    if pos<len(img_list)-1:
        pos=pos-1
    else:
        pos=0
    return readimage()

def readnextimage(position=-1):
    global pos
    if (position!=-1):
        pos=position
    else:
        if pos<len(img_list)-1:
            pos=pos+1
        else:
            pos=0
    return readimage()

def showCam():
    global show,reset
    reset=True
    show=not show
    
def capture():
    global take,output,reset
    reset=True
    take=not take
    if take:
        now=dt.now(ist)
        filename = "FakeFace_{}.mp4".format(now.strftime("%Y%m%dT%H%M%S"))
        print(filename)
        output = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MP4V'),3, (640,480))
    else:
        output.release()
        
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    shutil.copy(filename,media_path)
      
def show_frame():
    global previous,source_image,reset,x1,x2,y1,y2,show,take,output
    
    res,frame = cap.read()
    frame =cv2.resize(frame,(640,480))
    frame = cv2.flip(frame,1)
    
    if (previous is None or reset is True):
        x1,y1,x2,y2 = find_face_cut(net,frame)
        previous = cut_face_window(x1,y1,x2,y2,frame)
        reset = False
        
    curr_face=cut_face_window(x1,y1,x2,y2,frame.copy())
    
    deep_fake = process_image(source_image,previous,curr_face,generator,kp_detector,relative=True)
    deep_fake = cv2.cvtColor(deep_fake, cv2.COLOR_RGB2BGR)
    #print(deep_fake.shape,frame.shape)
    rgb = cv2.resize(deep_fake,(int(source_image.shape[0]//source_image.shape[1]*480),480))
    rgb=(rgb*255).astype(np.uint8)
    small=cv2.resize(frame,(150,150))
    
    x_border = int((640-(img_shape[1]//img_shape[0]*480))//2)
    y_border = int((480-(img_shape[0] // img_shape[1] * 640))//2)
    
    stream_v = cv2.copyMakeBorder(rgb,0,0,x_border if x_border >=0 else 0,x_border if x_border>=0 else 0, cv2.BORDER_CONSTANT)
    if show:
        stream_v[:150,-150:,:]=small
        
    destination = stream_v[top_y:bottom_y, left_x:right_x]
    stream_v = cv2.addWeighted(destination, 1, logo, 0.1, 0)
    # stream_v[top_y:bottom_y, left_x:right_x] = result
    
    if take:
        output.write(stream_v)
    cv2image = cv2.cvtColor(stream_v, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    
    
    
source_image=readnextimage(0)   
cap = cv2.VideoCapture(webcam_id)

waterMark=cv2.imread("buttons/watermark.jpg")
logo=cv2.resize(waterMark,(640,480))
h_logo, w_logo, _ = logo.shape

h_img, w_img, = 480,640
center_y = int(h_img/2)
center_x = int(w_img/2)

top_y = center_y - int(h_logo/2)
left_x = center_x - int(w_logo/2)
bottom_y = top_y + h_logo
right_x = left_x + w_logo
take=False

time.sleep(1)
width = cap.get(3)
height = cap.get(4)
print(f"webcam dimensions = {width} X {height}")
x1,x2,y1,y2=200,200,200,200
#load models
net = load_face_model()
# generator,kp_detector = demo.load_checkpoints(config_path=f'{first_order_path}config/vox-adv-256.yaml',checkpoint_path=f'{model_path}/vox-adv-cpk.pth.tar')
generator=torch.load(f"{model_path}/generator.pth")
generator.to(torch.device("cuda"))
generator.eval()
kp_detector=torch.load(f"{model_path}/kp_detector.pth")
kp_detector.to(torch.device("cuda"))
kp_detector.eval()

cam=tk.Button(root,text="C",command=showCam,bg="black",fg="white",borderwidth=0)
cam.place(relx=0.9,rely=0.5,height=30,width=30,bordermode="ignore")

LeftImage=ImageTk.PhotoImage(file="./buttons/lt.jpg")
Left_Button=ttk.Button(root,image=LeftImage,command=readpreviousimage)
Left_Button.place(relx=0.2,rely=0.85,bordermode='ignore')

centerImage=ImageTk.PhotoImage(file="./buttons/cap.png")
Center_Button=ttk.Button(root,image=centerImage,command=capture)
Center_Button.place(relx=0.5,rely=0.85,bordermode='ignore')

RightImage=ImageTk.PhotoImage(file="./buttons/gt.jpg")
Right_Button=ttk.Button(root,image=RightImage,command=readnextimage)
Right_Button.place(relx=0.8,rely=0.85,bordermode='ignore')

uploadImage=ImageTk.PhotoImage(file="./buttons/upload.png")
upload_Button=ttk.Button(root,image=uploadImage,command=UploadAction)
upload_Button.place(relx=0.01,rely=0.01,bordermode='ignore')

show_frame()
root.mainloop()
