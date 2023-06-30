import socket
from video_emotion import *
import threading
import time
import json
import base64
import hashlib
import os
import struct
import queue
import base64
from utils.inference import detect_faces
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
# 创建目录
def Mkdir(path): # path是指定文件夹路径
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

imgsQueue=queue.Queue(maxsize=0)
connected=False
HOST = 'localhost'
PORT = 9000
MAGIC_STRING = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
HANDSHAKE_STRING = "HTTP/1.1 101 Switching Protocols\r\n" \
                   "Upgrade:websocket\r\n" \
                   "Connection: Upgrade\r\n" \
                   "Sec-WebSocket-Accept: {1}\r\n" \
                   "WebSocket-Location: ws://{2}/chat\r\n" \
                   "WebSocket-Protocol:chat\r\n\r\n"

username=""
videoname=""

def send_data(con,data):
    if data:
        pass
    else:
        return False
    token = b'\x81'
    length = len(data)
    if length < 126:
        token += struct.pack("B", length)
    elif length <= 0xFFFF:
        token += struct.pack("!BH", 126, length)
    else:
        token += struct.pack("!BQ", 127, length)
    #struct为Python中处理二进制数的模块，二进制流为C，或网络流的形式。
    data = token+data
    con.send(data)
    return True

def handshake(con):
    #con为用socket，accept()得到的socket
    headers = {}
    shake = con.recv(1024)

    if not len(shake):
        return False
    shake_str=shake.decode()
    header, data = shake_str.split('\r\n\r\n', 1)
    for line in header.split('\r\n')[1:]:
        key, val = line.split(': ', 1)
        headers[key] = val

    if 'Sec-WebSocket-Key' not in headers:
        print ('This socket is not websocket, client close.')
        con.close()
        return False

    sec_key = headers['Sec-WebSocket-Key']
    res_key = base64.b64encode(hashlib.sha1((sec_key + MAGIC_STRING).encode()).digest())

    str_handshake = HANDSHAKE_STRING.replace('{1}', res_key.decode()).replace('{2}', HOST + ':' + str(PORT))
    print(str_handshake)
    con.send(str_handshake.encode())
    return True

def unpack_data(info):
    payload_len = info[1] & 127
    if payload_len == 126:
            extend_payload_len = info[2:4]
            mask = info[4:8]
            decoded = info[8:]
    elif payload_len == 127:
        extend_payload_len = info[2:10]
        mask = info[10:14]
        decoded = info[14:]
    else:
        extend_payload_len = None
        mask = info[2:6]
        decoded = info[6:]
    bytes_list = bytearray()
    for i in range(len(decoded)):
        chunk = decoded[i] ^ mask[i % 4]
        bytes_list.append(chunk)
    body = str(bytes_list, encoding='utf-8')
    return body



def subThreadIn(myconnection):
    global start
    global running
    global pause
    global connected
    print("成功连接,开始握手")
    flag=handshake(myconnection)
    if(flag):
        print("握手成功,开始监听")
        connected=True
    else:
        print("握手失败,关闭连接")
        connected=False
        return
    while True:
        try:
            recvedMsg = myconnection.recv(1024)
            recvedMsg=unpack_data(recvedMsg)
            if recvedMsg:
                print("收到消息",recvedMsg)
                if(recvedMsg=="start"):
                    start=True
                    running=True
                    pause=False
                    #接收用户与视频名称消息
                    recvedMsg = myconnection.recv(1024)
                    recvedMsg=unpack_data(recvedMsg)
                    msgs=recvedMsg.split(' ')
                    global username
                    global videoname
                    username=msgs[0]
                    videoname=msgs[1]
                elif(recvedMsg=="pause"):
                    pause=not pause
                elif(recvedMsg=="stop"):
                    start=False
                    running=False
                    pause=False

        except Exception as e:
            print(e)
            print("连接关闭")
            connected=False
            myconnection.close()
            return



# 初始化
detection_model_path = 'F:\\homework1\\2023-spring\\visual-perception\\visual-perception-final-project\\backend\\models\\haarcascade_frontalface_default.xml'
emotion_model_path = 'F:\\homework1\\2023-spring\\visual-perception\\visual-perception-final-project\\backend\\models\\fer2013_big_XCEPTION.54-0.66.hdf5'
emotion_label = 'fer2013'
face_detection,emotion_classifier=loadModel(detection_model_path,emotion_model_path)
emotion_labels=getLabels("fer2013")

start=False
running=True
pause=False

server = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
server.bind(("localhost",9000))
print("启动")
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.listen(1)

while True:
    if(not connected):
        print("等待连接...")
        connection,addr=server.accept()
        connected=True
        print('Accept a new connection', connection.getsockname(), connection.fileno())
        mythread = threading.Thread(target=subThreadIn, args=[connection])
        mythread.setDaemon(True)
        mythread.start()
    if not start:
        continue
    video_capture = cv2.VideoCapture(1)
    emotion_offsets = (20, 40)
    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]
    # starting video streaming
    res={
        'emotions':[],
        'emotion_texts':[],
        'emotion_probs':[],
        'times_second':[]
    }
    # 初始化完成
    start_time = time.perf_counter()
    firstEnterPause=True
    pauseTime=0.0
    while (running):
        while pause:
            if firstEnterPause:
                pauseStartTime=time.perf_counter()
                firstEnterPause=False
            
        if not firstEnterPause:
            firstEnterPause=True
            pauseTime+=time.perf_counter()-pauseStartTime
            print(pauseTime)
        ret,bgr_image = video_capture.read()

        # 获取当前时间
        current_time = time.perf_counter()
        delta_time=current_time-start_time-pauseTime

        print(delta_time)
        if not ret:
            break
        #将摄像头画面化为base64编码字符串然后放进发送队列
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        retval, buffer = cv2.imencode('.jpeg', bgr_image,encode_param)
        jpg_as_byte = base64.b64encode(buffer)
        send_data(connection,jpg_as_byte)
        gray_image=cv2.cvtColor(bgr_image,cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_detection,gray_image)
        for face in faces:
            x1, x2, y1, y2 = apply_offsets(face, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            break
        # 出现多个脸的情况可能是噪声,忽略
        if(len(faces)>0): #检测到人脸
            emotionText=getEmotionText(emotion_labels,emotion_prediction)
            emotionProb=getEmotionProbability(emotion_prediction)
            res['emotions'].append(emotion_prediction.tolist())
            res['emotion_texts'].append(emotionText)
            res['emotion_probs'].append(emotionProb.tolist())
            res['times_second'].append(delta_time)

    print(res)
    json_data=json.dumps(res)
    Mkdir('F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/'+username)
    with open('F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/'+username+"/"+videoname+".json",'w') as f:
        f.write(json_data)
        print("写入成功"+'F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/'+username+"/"+videoname+".json")



    video_capture.release()
    start=False
    running=True
    pause=False


