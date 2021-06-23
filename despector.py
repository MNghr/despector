import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
from torchvision import transforms
import copy
from collections import OrderedDict
from PIL import Image

import PyQt5
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGraphicsScene,QApplication, QMainWindow, QLabel, QWidget, QGraphicsView
from hlmobilenetv2 import hlmobilenetv2

RESTORE_FROM = './indexnet_matting.pth.tar'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = hlmobilenetv2(
    pretrained=False,
    freeze_bn=True, 
    output_stride=32,
    apply_aspp=True,
    conv_operator='std_conv',
    decoder='indexnet',
    decoder_kernel_size=5,
    indexnet='depthwise',
    index_mode='m2o',
    use_nonlinear=True,
    use_context=True
)

try:
    checkpoint = torch.load(RESTORE_FROM, map_location=device)
    pretrained_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module' in key:
            key = key[7:]
        pretrained_dict[key] = value
except:
    raise Exception('Please download the pretrained model!')
net.load_state_dict(pretrained_dict)
net.to(device)
if torch.cuda.is_available():
    net = nn.DataParallel(net)
net.eval()

class ProcessImage(QObject):
    def __init__(self):
        super().__init__()
    
    def processimage():
        pass
    



class Main(QWidget):

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.title = '素材作成支援ツール'

        self.width = 1920
        self.height = 1080
        eimg = cv2.imread('/Users/nagahara/kadai/black.png')
        
        

        self.targetImage = targetImageWidget('/Users/nagahara/kadai/black.png',self)
        self.targetImage.setVisible(False)
        self.outcomes = []
        for i in range(1):
            self.outcomes.append(outcomeWidget(QtGui.QImage(eimg.flatten(), eimg.shape[1], eimg.shape[0], QtGui.QImage.Format_RGB888),self))
            self.outcomes[i].move(800,250*i+50)
            self.outcomes[i].setVisible(False)

        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height)

        d = PyQt5.QtWidgets.qApp
        desktop = d.desktop()
        framesize = self.frameSize()
        geometry = desktop.screenGeometry()
        self.labelA = QLabel(self)
        self.labelA.setText('Drop your image')
        self.labelA.move(520,220)
        font = QtGui.QFont("Courier",74,50)
        #font.setPointSize(96)
        self.labelA.setFont(font)
        self.show()
        self.outcomeGraphics = outcomeGraphics(QtGui.QImage(eimg.flatten(), eimg.shape[1], eimg.shape[0], QtGui.QImage.Format_RGB888),self)
    
    def dropEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dropEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print(type(mimetype))
            print('Data:', mimeData.data(mimetype))
            print()
        print()

        if mimeData.formats()[0] == "text/uri-list":
            print(type(mimeData.data(mimeData.formats()[0])))
            print(str(mimeData.data(mimeData.formats()[0])))
            print(type(mimeData.data(mimeData.urls()[0].toString())))
            print(mimeData.data(mimeData.formats()[0]))
            print(type(mimeData.urls()[0].toString()))

            self.labelA.setText('Now Processing...')
            
            
            imageSrc = mimeData.urls()[0].toString().replace('file://',"")
            self.targetImage.updateImage(imageSrc)
            self.targetImage.setVisible(True)

            img = cv2.imread(imageSrc)
            img = img[...,::-1]
            #water = watershed(copy.deepcopy(img))
            indexnet = indexNetMatting(copy.deepcopy(img))
            print("indexnet:"+str(indexnet[(indexnet != 0) & (indexnet != 255)]))
            print(indexnet.shape)
            #self.outcomes[0].updateImage(QtGui.QImage(water.flatten(), water.shape[1], water.shape[0], QtGui.QImage.Format_RGB888))
            self.outcomes[0].updateImage(QtGui.QImage(indexnet.flatten(), indexnet.shape[1], indexnet.shape[0], QtGui.QImage.Format_RGB888))
            self.outcomes[0].setVisible(True)
            self.outcomeGraphics.updateImage(QtGui.QImage(indexnet.flatten(), indexnet.shape[1], indexnet.shape[0], QtGui.QImage.Format_RGB888))
            #self.outcomes[1].setVisible(True)
            self.labelA.setVisible(False)
            
            


    def dragEnterEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dragEnterEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        print()

#セグメント結果用のウィジェット
class outcomeGraphics(QGraphicsView):
    def __init__(self,image=None,parent=None):
        super().__init__(parent)
        
        self.width = 300
        self.height = 400
        self.resize(self.width,self.height)
        pixmap = QtGui.QPixmap(image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setScene(scene)
        self.show()


    def updateImage(self,image=None):
        #print("imageSrc:"+str(imageSrc))
        pixmap = QtGui.QPixmap(image)
        scene = QGraphicsScene()
        scene.addPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setScene(scene)
        self.show()

    
    def MousePressEvent():
        pass
    
class outcomeWidget(QLabel):
    def __init__(self,image=None,parent=None):
        super().__init__(parent)
        
        self.width = 300
        self.height = 400
        self.resize(self.width,self.height)
        pixmap = QtGui.QPixmap(image)
        self.setPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        


    def updateImage(self,image=None):
        #print("imageSrc:"+str(imageSrc))
        self.update()
        self.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
    
    def MousePressEvent():
        pass

class targetImageWidget(QLabel):
    def __init__(self,imageSrc=None,parent=None):
        super().__init__(parent)
        
        self.width = 300
        self.height = 400
        self.resize(self.width,self.height)
        self.image = QtGui.QImage(imageSrc)
        pixmap = QtGui.QPixmap(imageSrc)
        self.setPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setWindowOpacity = 1.0
    
    #def paintEvent(self,event):
    #     print("")
    #    painter = QtGui.QPainter(self.image)
    #    #painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
    #    painter.drawImage(0,0, self.image)
    #    painterLabel = QtGui.QPainter(self)
    #    painterLabel.drawImage(0,0, self.image)
    #    
    #    painter.end()
        

    
    def updateImage(self,imageSrc=None):
        print("imageSrc:"+str(imageSrc))
        self.image = QtGui.QImage(imageSrc)
        self.update()
        self.setPixmap(QtGui.QPixmap.fromImage(self.image).scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        

       

    def MousePressEvent():
        pass

def indexNetMatting(img):
    ret = img[:,:,:]
    h,w,_ =  img.shape
    img = cv2.resize(img,(320,320))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask,(w,h))
    img = cv2.resize(img,(w,h))
    print(mask.shape)
    plt.close()
    plt.gray()
    plt.figure(figsize = (20,20))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.savefig("mask.png")
    plt.close()
    
    
    trimap = generateTrimap(mask,kernelSize=(10,10),iterate = 5)
    plt.close()
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.imshow(trimap)
    
    plt.savefig("trimap.png")

    #return trimap




    with torch.no_grad():
        #img = np.array(Image.open('./examples/images/beach-747750_1280_2.png'))
        #ret = img[:,:,:]
        #trimap = np.array(Image.open('./examples/trimaps/beach-747750_1280_2.png'))
        #h,w,_ =  img.shape

        trimap = np.expand_dims(trimap,axis=2)
        print(trimap.shape)
        img = np.concatenate((img,trimap),axis=2)
        print(img.shape)
        h,w = img.shape[:2]

        img = img.astype('float32')
        img = (1./255*img-np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))) / np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))
        img = img.astype('float32')

        img = image_alignment(img,32)
        print(img.shape)
        print(np.expand_dims(img.transpose(2,0,1),axis=0).shape)
        inputs = torch.from_numpy(np.expand_dims(img.transpose(2,0,1),axis=0))
        print(inputs.size)
        inputs = inputs.to(device)

        outputs = net(inputs)

        outputs = outputs.squeeze().cpu().numpy()
        alpha = cv2.resize(outputs,dsize=(w,h),interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha,0,1) * 255.0
        trimap = trimap.squeeze()
        mask = np.equal(trimap,128).astype(np.float32)
        alpha = (1-mask)*trimap+mask*alpha
        
        
       
        plt.close()
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(trimap)
        plt.savefig("alpha.png")

        print(type(alpha))
        print(type(ret))

        print(alpha.shape)
        print(ret.shape)


        print(ret.shape)
        plt.close()
        plt.imshow(ret)
        plt.savefig("xx.png")
        ret = ret.astype(float)
        plt.close()
        plt.imshow(ret.astype(np.uint8))
        plt.savefig("x.png")

        bg = np.full_like(ret,255)
        bg[:,:,0] = 0
        bg[:,:,2] = 0
        print(alpha[(alpha != 0) & (alpha != 255)])
        alpha = alpha.astype(float)/255.0
        

        ret[:,:,0] = cv2.multiply(ret[:,:,0], alpha)
        ret[:,:,1] = cv2.multiply(ret[:,:,1], alpha)
        ret[:,:,2] = cv2.multiply(ret[:,:,2], alpha)

        plt.close()
        plt.imshow(ret.astype(np.uint8))
        plt.savefig("xxxx.png")

        bg[:,:,0] = cv2.multiply(bg[:,:,0], 1.0-alpha)
        bg[:,:,1] = cv2.multiply(bg[:,:,1], 1.0-alpha)
        bg[:,:,2] = cv2.multiply(bg[:,:,2], 1.0-alpha)



        print(bg[(bg != 255) & (bg != 0)])

        plt.close()
        plt.imshow(bg.astype(np.uint8))
        plt.savefig("bg.png")
        
        outImage = ret+bg

        print(ret.shape)
        plt.close()
        plt.imshow(np.clip(outImage,0,255))
        plt.savefig("output.png")

    return outImage.astype(np.uint8)

def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x


def generateTrimap(mask,kernelSize=(5,5),iterate=1):
    kernel = np.ones(kernelSize,np.uint8)
    eroded = cv2.erode(mask,kernel,iterations = iterate)
    plt.close()
    plt.subplot(1,2,1)
    plt.imshow(mask)
    plt.subplot(1,2,2)
    plt.imshow(eroded)
    plt.savefig("eroded.png")
    dilated = cv2.dilate(mask,kernel,iterations = iterate)

    plt.close()
    plt.subplot(1,2,1)
    plt.imshow(mask)
    plt.subplot(1,2,2)
    plt.imshow(dilated)
    plt.savefig("dilated.png")

    trimap = np.full(mask.shape,128)
    print(trimap)
    plt.close()
    plt.subplot(1,2,1)
    plt.imshow(mask)
    plt.subplot(1,2,2)
    plt.imshow(trimap)
    
    trimap[eroded != 0] = 255
    trimap[dilated == 0] = 0
    print("eroded:"+str(eroded[eroded != 0]))
    plt.savefig("tri.png")
    print(trimap[trimap == 128].shape)
    cv2.imwrite('./examples/trimaps/11.png',trimap)
    return trimap



if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Main()
    
    
    sys.exit(app.exec_())