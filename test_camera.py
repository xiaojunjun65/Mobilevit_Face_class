import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

def cv2ImgAddText(img, label, left, top, textColor=(0, 255, 0)):
    img = Image.fromarray(np.uint8(img))
    # 设置字体
    font = ImageFont.truetype(font='simhei.ttf', size=20)

    draw = ImageDraw.Draw(img)
    label = label.encode('utf-8')
    draw.text((left, top), str(label,'UTF-8'), fill=textColor, font=font)
    return np.asarray(img)







classes = ["魏赫男","js","hly"]
threshold = 0.89
model = torch.load("/data/guojun/vit/Moblievit_class/output1/model_70--0.774193525314331.pth", map_location=torch.device('cpu'))
model.eval()


camera = cv2.VideoCapture('test_video/y5.mp4')
# camera = cv2.VideoCapture(0)
# 检查视频是否打开成功
if not camera.isOpened():
    print("Error opening video file")
    exit()
while (True):
    # 调取摄像头，读取一帧图像
    # flag 如果一直播放返回ture
    # frame 当前一帧对应的图片
    flag, frame = camera.read()
    # 判断图片读取成功

    transform = transforms.Compose([transforms.Resize(int(256 * 1.14)),
         transforms.CenterCrop(256),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    face_data = Image.fromarray(frame)
    input_image = transform(face_data)
    input_image = torch.reshape(input_image,(1, input_image.shape[0], input_image.shape[1], input_image.shape[2]))
    with torch.no_grad():
        output = model(input_image)
        # _, predicted = torch.max(output.data, 1)
        # name = 'unknown' if predicted.item() == 0 else 'person'
        # 在图像上显示人脸名字
        value = torch.softmax(output, 1).numpy().max()  # 预测概率
        if value <= threshold:
            org_list = [(0, 20), (0, 50)]
            name = "未录入人脸信息"
            img = cv2ImgAddText(frame, name, 100, 30,(0,0,255))
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:

            result = output.argmax(1)
            # name =[classes[result]]
            name = classes[result]
            img = cv2ImgAddText(frame, name, 100, 30)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img =cv2ImgAddText(img,"当前人员",0,30)
        img = cv2ImgAddText(img, "准确率：{}".format(value), 0, 60)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('face_camera', img)
        # 如果按下q键则退出
        if cv2.waitKey(100) & 0xff == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()