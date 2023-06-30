from flask import Flask,jsonify,request
import json
import numpy as np
import math
import os

def list_users():
    startpath="F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas"
    res=[]
    for root, dirs, files in os.walk(startpath):
        if root == startpath:
            continue
        temp={"user":'',"videos":[]}
        temp['user']= os.path.basename(root)
        for f in files:
            temp['videos'].append(os.path.splitext(f)[0])
        res.append(temp)
    json_data = json.dumps(res, ensure_ascii=False)
    return json_data

def list_videos():
    startpath="F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas"
    res=[]
    for root, dirs, files in os.walk(startpath):
        for f in files:
            res.append(os.path.splitext(f)[0])
    json_data = json.dumps(res, ensure_ascii=False)
    return json_data

def getTwoEmoBarData(data):
    emotions=data['emotions']
    good=0
    bad=0
    for a in emotions:
        e=a[0]
        good+=e[0]+e[1]+e[2]+e[4]
        bad+=e[3]+e[5]
    total=good+bad
    good=round(good/total,2)
    bad=round(bad/total,2)
    return [good,bad]

def getMulEmoRadarData(data):
    emotions=data['emotions']
    res=np.zeros(7)
    for a in emotions:
        e=np.array(a[0])
        res+=e
    norm = np.linalg.norm(res)
    res=res/norm
    for i in range (len(res)):
        res[i]=round(res[i],2)
    res=res.tolist()
    print(res)
    return res

def getRiverData(data):
    emotions=data['emotions']
    probs=data['emotion_probs']
    times=data['times_second']
    t=0
    res=[]
    vec=np.zeros(7)
    flag=False
    for i in range(len(times)):
        time=math.floor(times[i])
        if time==t:
            flag=True
            vec+=np.array(emotions[i][0])*probs[i]
        else:
            if(flag):
                norm = np.linalg.norm(vec)
                vec=vec/norm
                vec=vec.tolist()
                res.append([t,round(vec[0],3),'angry'])
                res.append([t,round(vec[1],3),'disgust'])
                res.append([t,round(vec[2],3),'fear'])
                res.append([t,round(vec[3],3),'happy'])
                res.append([t,round(vec[4],3),'sad'])
                res.append([t,round(vec[5],3),'surprise'])
                res.append([t,round(vec[6],3),'neutral'])
                vec=np.zeros(7)
            t+=1
            flag=False
    #print(res)
    return res

def getDiffRadarData(username):
    startpath="F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username
    videos=[]
    datas=[]
    for root, dirs, files in os.walk(startpath):
        for f in files:
            name=os.path.splitext(f)[0]
            videos.append(name)
            with open("F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username+'/'+f) as file:
                data=json.load(file)
                data=getMulEmoRadarData(data)
                datas.append({'value':data,'name':name})
    res={'videos':videos,'datas':datas}
    print(res)
    return res

def getDiffBarData(username):
    startpath="F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username
    videos=[]
    happy=[]
    surprise=[]
    neutral=[]
    angry=[]
    disgust=[]
    fear=[]
    sad=[]
    for root, dirs, files in os.walk(startpath):
        for f in files:
            name=os.path.splitext(f)[0]
            videos.append(name)
            with open("F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username+'/'+f) as file:
                data=json.load(file)
                data=getMulEmoRadarData(data)
                happy.append(data[3])
                surprise.append(data[5])
                neutral.append(data[6])
                angry.append(data[0])
                disgust.append(data[1])
                fear.append(data[2])
                sad.append(data[4])
    res={'videos':videos,'happy':happy,'surprise':surprise,'neutral':neutral,'angry':angry,'disgust':disgust,'fear':fear,'sad':sad}
    print(res)
    return res

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/getAllVideos/')
def getAllVideos():
    try:
        return list_videos()
    except Exception:
        return jsonify([])

@app.route('/getUserList/')
def userList():
    try:
        return list_users()
    except Exception:
        return jsonify([])

@app.route('/getData/<username>/<videoname>/')
def getData(username,videoname):
    try:
        with open("F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username+'/'+videoname+'.json') as f:
            data = json.load(f)
            return jsonify(data)
    except Exception:
        return jsonify({})

@app.route('/getTwoEmoBar/<username>/<videoname>/')
def getTwoEmoBar(username,videoname):
    try:
        with open("F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username+'/'+videoname+'.json') as f:
            data = json.load(f)
            return jsonify(getTwoEmoBarData(data))
    except Exception as e:
        #print(e)
        return jsonify([1,1])

@app.route('/getMulEmoRadar/<username>/<videoname>/')
def getMulEmoRadar(username,videoname):
    try:
        with open("F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username+'/'+videoname+'.json') as f:
            data = json.load(f)
            return jsonify(getMulEmoRadarData(data))
    except Exception as e:
        #print(e)
        return jsonify([1,1,1,1,1,1,1])

@app.route('/getRiverData/')
def getRiver():
    username=request.args.get('username')
    videoname=request.args.get('videoname')
    try:
        with open("F://homework1/2023-spring/visual-perception/visual-perception-final-project/backend/userdatas/"+username+'/'+videoname+'.json') as f:
            data = json.load(f)
            return jsonify(getRiverData(data))
    except Exception as e:
        print(e)
        return jsonify([])

@app.route('/getDiffRadar/')
def getDiffRadar():
    username=request.args.get('username')
    try:
        return jsonify(getDiffRadarData(username))
    except Exception as e:
        print(e)
        return jsonify({'videos':[],'datas':[]})

@app.route('/getDiffBar/')
def getDiffBar():
    username=request.args.get('username')
    try:
        return jsonify(getDiffBarData(username))
    except Exception as e:
        print(e)
        return jsonify({'videos':[],'happy':[],'surprise':[],'neutral':[],'angry':[],'disgust':[],'fear':[],'sad':[]})

if __name__=="__main__":
    app.run()