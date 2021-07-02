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
import qimage2ndarray

import PyQt5
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGraphicsScene,QApplication, QMainWindow, QLabel, QWidget, QGraphicsView, QScrollArea, QVBoxLayout
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




class Main(QScrollArea):
    startSegmentation = pyqtSignal()
    def __init__(self):
    
        super().__init__()
        self.setWidgetResizable(True)
        self.setAcceptDrops(True)
        self.title = 'despector'
        #self.startSegmentation.connect(self.doSegmentation)
        self.segmentationProcess = doSegmentationProcess()
        
        self.startSegmentation.connect(self.segmentationProcess.start)
        self.segmentationProcess.finished.connect(self.onSegmentationFinished)


        self.width = 1920
        self.height = 1080
        eimg = cv2.imread('/Users/nagahara/kadai/black.png')
        #self.targetImage = targetImageWidget('/Users/nagahara/kadai/black.png',self)
        self.targetImage = targetGraphics(QtGui.QImage(eimg.flatten(), eimg.shape[1], eimg.shape[0], QtGui.QImage.Format_RGB888),self)
        self.targetImage.move(0,0)
        self.targetImage.setVisible(False)
        self.outcomes = []
        self.image = None
        
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
        self.outcomeGraphics.setVisible(False)

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
            img = cv2.imread(imageSrc)
            self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

            print("accept Drop.")
            self.segmentationProcess.setImg(self.img)
            self.targetImage.setVisible(True)
            self.targetImage.updateImage(qimage2ndarray.array2qimage(self.img))
            self.segmentationProcess.rect = None


            self.startSegmentation.emit()
            
            #indexnet = indexNetMatting(copy.deepcopy(self.img))
            #print("indexnet:"+str(indexnet[(indexnet < 0) & (indexnet > 255)]))
            #print(indexnet.shape)
            #self.outcomes[0].updateImage(cv2pixmap(indexnet))
            #self.outcomes[0].setVisible(True)
            #print(indexnet.dtype)
            #self.outcomeGraphics.updateImage(cv2pixmap(indexnet))
            #self.outcomes[1].setVisible(True)
            #self.labelA.setVisible(False)
            
    
    def dragEnterEvent(self, event):
        event.accept()
        mimeData = event.mimeData()
        print('dragEnterEvent')
        for mimetype in mimeData.formats():
            print('MIMEType:', mimetype)
            print('Data:', mimeData.data(mimetype))
            print()
        print()
        #self.startSegmentation.emit()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            #indexnet = indexNetMatting(copy.deepcopy(self.img))
            #self.outcomes[0].updateImage(QtGui.QImage(indexnet.flatten(), indexnet.shape[1], indexnet.shape[0], QtGui.QImage.Format_RGB888))
            #self.outcomes[0].setVisible(True)
            self.segmentationProcess.setRect(self.targetImage.scene.rect)
            self.segmentationProcess.setImg(self.img)
            self.startSegmentation.emit()

    def onSegmentationFinished(self):
        print("segmentation Finished")
        self.labelA.setVisible(False)
        self.outcomeGraphics.updateImage(self.segmentationProcess.resultQImage)
        
        
    
"""        
    @pyqtSlot()
    def doSegmentation(self):
        if self.targetImage.scene.rect is not None:
            x,y,height,width = int(self.targetImage.scene.rect.x()),int(self.targetImage.scene.rect.y()),int(self.targetImage.scene.rect.height()),int(self.targetImage.scene.rect.width())

            print(self.targetImage.scene.rect)
    
            
            print(x)
            print(y)
            print(height)
            print(width)
            print(self.img.dtype)
            z = qimage2cv(self.targetImage.image)
            
            z = self.img[y:min(height+y,self.img.shape[0]),x:min(width+x,self.img.shape[1])]
            print(z.shape)
            print(z.dtype)
            plt.close()
            plt.imshow(z)
            plt.savefig("img.png")
            #q = qimage2ndarray.array2qimage(z)
            #self.outcomeGraphics.updateImage(q)
            indexnet = indexNetMatting(z)
            self.outcomeGraphics.updateImage(qimage2ndarray.array2qimage(indexnet))

        else:
            self.targetImage.updateImage(qimage2ndarray.array2qimage(self.img))

            self.targetImage.setVisible(True)

            
            indexnet = indexNetMatting(copy.deepcopy(self.img))
            self.outcomeGraphics.updateImage(qimage2ndarray.array2qimage(indexnet))
            
            self.labelA.setVisible(False)
"""


#セグメンテーションの処理がクソ重いのでスレッド化
class doSegmentationProcess(QThread):
    isFinished = pyqtSignal()
    def __init__(self, parent=None, rect = None, imgNdarray = None):
        QtCore.QThread.__init__(self, parent)

        self.mutex = QtCore.QMutex()
        self.rect = rect
        self.img = imgNdarray
        self.stopped = False
        self.resultQImage = None

    def setImg(self,imgNdarray):
        self.img = imgNdarray
    
    def setRect(self,rect):
        self.rect = rect

    def __del__(self):
        self.stop()
        self.wait()

    def stop(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = True

    def restart(self):
        with QtCore.QMutexLocker(self.mutex):
            self.stopped = False

    def run(self):
        print("do Segmentation")
        if self.stopped == True:
            return

        if self.rect is not None:
            x,y,height,width = int(self.rect.x()),int(self.rect.y()),int(self.rect.height()),int(self.rect.width())

            print(self.rect)
    
            
            print(x)
            print(y)
            print(height)
            print(width)
            print(self.img.dtype)
            
            z = self.img[y:min(height+y,self.img.shape[0]),x:min(width+x,self.img.shape[1])]
            print(z.shape)
            print(z.dtype)
            indexnet = indexNetMatting(z)
            self.resultQImage = qimage2ndarray.array2qimage(indexnet)

        else:
            indexnet = indexNetMatting(copy.deepcopy(self.img))
            print("indexnet Finished")
            self.resultQImage = qimage2ndarray.array2qimage(indexnet)
    



#セグメント結果用のGraphics
class outcomeGraphics(QGraphicsView):
    def __init__(self,image=None,parent=None):
        super().__init__(parent)
        self.image = image
        self.setStyleSheet("background-color: transparent;")
        self.height,self.width = image.size().height(), image.size().width()
        #self.resize(self.width,self.height)
        pixmap = QtGui.QPixmap(image)
        self.move(700,0)
        
        self.scene = Scene(self)
        self.scene.addPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setScene(self.scene)
        #self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.show()


    def updateImage(self,image=None):
        #print("imageSrc:"+str(imageSrc))
        self.height, self.width = image.size().height(), image.size().width()
        self.image = image
        
        pixmap = QtGui.QPixmap(image)
        self.scene = Scene(self)
        self.scene.addPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setScene(self.scene)
        #self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.show()


class targetGraphics(QGraphicsView):
    def __init__(self,image=None,parent=None):
        super().__init__(parent)
        self.image = image
        self.setStyleSheet("background-color: transparent;")
        self.height,self.width = image.size().height(), image.size().width()
        #self.resize(self.width,self.height)
        pixmap = QtGui.QPixmap(image)
        
        self.scene = targetScene(self)
        self.scene.addPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setScene(self.scene)
        #self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.show()


    def updateImage(self,image=None):
        #print("imageSrc:"+str(imageSrc))
        self.height, self.width = image.size().height(), image.size().width()
        self.image = image
        
        pixmap = QtGui.QPixmap(image)
        self.scene = targetScene(self)
        self.scene.addPixmap(pixmap.scaled(self.width,self.height,Qt.KeepAspectRatio,Qt.FastTransformation))
        self.setScene(self.scene)
        #self.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        print(self.scene.sceneRect())
        self.show()

    
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

class Scene(QGraphicsScene):

    mouse_pos = None
    sel_item = None
    sel_mode = False
    rect = None
    startX = None
    stratY = None
    endX = None
    endY = None

    def __init__(self, *args, **kwargs):
 
        super(Scene, self).__init__(*args, **kwargs)
 
class targetScene(Scene):
    
    def mousePressEvent(self, e):
        self.x = self.width()
        self.y = self.height()
        self.rect = None
        if e.button() == QtCore.Qt.LeftButton:
            self.mouse_pos = e.scenePos()
            if(self.sel_item is not None):
                self.removeItem(self.sel_item)

            self.sel_item = self.addRect(QtCore.QRectF(self.mouse_pos.toPoint(), self.mouse_pos.toPoint()),QtGui.QPen(QtCore.Qt.green,4,QtCore.Qt.DotLine))
 
            if e.modifiers() != QtCore.Qt.ShiftModifier:
                # 通常モードの場合は、色つきのRectを作成
                #self.sel_item.setBrush(QtGui.QColor("pink"))
                pass
            else:
                # Shiftを押しながらの時は、選択まとめて選択モード(撤廃)
                self.sel_mode = False
        print("いいいいいいいいい")


    def mouseMoveEvent(self, e):
 
        if self.sel_item is not None:
            cur = e.scenePos()
            if self.mouse_pos is not None and isBounded(self.mouse_pos.x(),self.mouse_pos.y(),self.x,self.y):
                st_x = self.mouse_pos.x()
                st_y = self.mouse_pos.y()
                cur_x = cur.x() 
                cur_y = cur.y()
                print("posision:"+str(cur_x)+","+str(cur_y))
                self.startX = st_x
                self.startY = st_y
                self.endX = cur_x
                self.endY = cur_y
            else:
                return
                
            rect = None
 
            # カーソル位置に応じて、■の生成を変える
            # スタート位置から右上
            if st_x < cur_x and st_y > cur_y:
                rect = QtCore.QRectF(QtCore.QPoint(st_x, cur_y), QtCore.QPoint(cur_x, st_y))
             # スタート位置から右下
            if st_x < cur_x and st_y < cur_y:
                rect = QtCore.QRectF(self.mouse_pos.toPoint(), cur.toPoint())
            # スタート位置から左上
            if st_x > cur_x and st_y < cur_y:
                rect = QtCore.QRectF(QtCore.QPoint(cur_x, st_y), QtCore.QPoint(st_x, cur_y))
 
            if st_x > cur_x and st_y > cur_y:
                rect = QtCore.QRectF(cur.toPoint(), self.mouse_pos.toPoint())
 
            if rect is not None:
                self.sel_item.setRect(rect)
                self.rect = rect
                 
    def mouseReleaseEvent(self, e):
        self.mouse_pos = None
        # 選択用の矩形は削除する
        if self.sel_mode is True:
            self.removeItem(self.sel_item)
            print (str(len(self.items(self.sel_item.sceneBoundingRect()))))
            self.sel_mode = False
        #self.sel_item = None
      

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
    
    trimap = generateTrimap(mask,kernelSize=(10,10),iterate = 5)

    with torch.no_grad():
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
        
        print(type(alpha))
        print(type(ret))

        print(alpha.shape)
        print(ret.shape)


        print(ret.shape)
        ret = ret.astype(float)
        
        bg = np.full_like(ret,255)
        bg[:,:,0] = 0
        bg[:,:,2] = 0
        print(alpha[(alpha != 0) & (alpha != 255)])
        alpha = alpha.astype(float)/255.0
        

        ret[:,:,0] = cv2.multiply(ret[:,:,0], alpha)
        ret[:,:,1] = cv2.multiply(ret[:,:,1], alpha)
        ret[:,:,2] = cv2.multiply(ret[:,:,2], alpha)


        bg[:,:] = chooseBackColor(img)
        bg[:,:,0] = cv2.multiply(bg[:,:,0], 1.0-alpha)
        bg[:,:,1] = cv2.multiply(bg[:,:,1], 1.0-alpha)
        bg[:,:,2] = cv2.multiply(bg[:,:,2], 1.0-alpha)



        print(bg[(bg != 255) & (bg != 0)])

        
        outImage = ret+bg

        print(ret.shape)

        outImage = np.clip(outImage,0,255)
        
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
    dilated = cv2.dilate(mask,kernel,iterations = iterate)

    trimap = np.full(mask.shape,128)
    print(trimap)
    
    trimap[eroded != 0] = 255
    trimap[dilated == 0] = 0
    print("eroded:"+str(eroded[eroded != 0]))
    #plt.savefig("tri.png")
    print(trimap[trimap == 128].shape)
    cv2.imwrite('./examples/trimaps/11.png',trimap)
    return trimap

def chooseBackColor(img):
    print("backcolor")
    includedColorList = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            includedColorList.append(tuple(img[i][j]))

    includedColorSet = set(includedColorList)

    for r in range(256):
        for g in range(255,0,-1):
            for b in range(256): 
                if not ((r,g,b) in includedColorSet):
                    
                    return np.array((r,g,b))

    print("backcolor end")

    return np.array((0,0,0))

def qimage2cv(qimage):
    w, h, d = qimage.size().width(), qimage.size().height(), qimage.depth()
    bytes_ = qimage.bits().asstring(w * h * d // 8)
    arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
    return arr

def cv2pixmap(cvImage,convertColor = False):
    height, width, bytesPerComponent = cvImage.shape
    bytesPerLine = cvImage.strides[0] 
    X = None
    if convertColor :
        X = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB )
    else:
        X = cvImage[:,:,:]
    
    print(X.shape)
    print(type(width))
    print(type(height))
    print(type(bytesPerLine))
    print(X.data)

    return QtGui.QImage(X.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    

def isBounded(x,y,boundX,boundY):
    return x >= 0 and y >= 0 and x <= boundX and y <= boundY

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Main()
    
    
    sys.exit(app.exec_())