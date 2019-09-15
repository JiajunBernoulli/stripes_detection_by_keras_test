import subprocess
import time
import requests
import pandas as pd
import time
import os, shutil
import re
import numpy as np
from PIL import Image
from tensorflow.tensorboard.backend.event_processing import event_accumulator
# from skimage import io
from config import WIDTH, HEIGHT
def download_data():
    for networkName in os.listdir('./logs'):
        # 先获取logs目录下不同神经网络的名字
        print(networkName)
        # if networkName == 'ResNet34' or networkName == 'DenseNet121':
        #     continue
        # 根据获得的名字打开tensorboard
        p = subprocess.Popen('tensorboard --logdir ./logs/' + networkName, shell=True)
        # 等待数据加载，数据越多，设置时长越久
        time.sleep(50)
        # 列出想要获得的验证集数据名字
        for title in ['acc', 'f1_score', 'precision', 'recall']:
            # 请求数据，获得结果相当于python中打印出的字符串
            # try:
            response = requests.get(
                'http://localhost:6006/data/plugin/scalars/scalars?tag=epoch_val_' + title + '&run=.&experiment=')
            if 'error' in response.text:
            # except:
                response = requests.get(
                    'http://localhost:6006/data/plugin/scalars/scalars?tag=val_' + title + '&run=.&experiment=')
            datas = response.text
            # 将获得结果存入DataFrame，注意eval的使用；dtype设置为字符串避免Wall time的时间戳被用科学计算法表示
            df = pd.DataFrame(eval(datas), dtype=str, columns=['Wall time', 'Step', 'Value'])
            # 检查目录存在与否，否则新建
            if not os.path.exists('./download/' + networkName):
                os.mkdir('./download/' + networkName)
            # 直接将获得的结果存为csv文件
            df.to_csv('./download/' + networkName + '/' + title + '.csv', index=False)
            # 打印行数，可以帮助检查下载数据是否完整
            print(df.shape[0])
        # 调用命令获得pid并杀死进程
        result = os.popen('tasklist | findstr "tensorboard"')
        res = result.read()
        for line in res.splitlines():
            pid = re.findall(' ?([0-9]*) Con', line)[0]
            print(pid)
            os.system('taskkill /f /pid ' + str(pid))
        p.kill()

def dist_log_data():
    srcPath = './download/'
    dirs = os.listdir(srcPath)
    for dir in dirs:
        for file in os.listdir(srcPath+dir):
            title = file[:-4]
            if not os.path.exists('./csvs/'+title):
              os.mkdir('./csvs/'+title)
            shutil.move(srcPath+dir+'/'+file, './csvs/'+title+'/'+dir+'.csv')


def alter_brightness():
    def Illumi_adjust(alpha, img):
        if alpha > 0:
            img_out = img * (1 - alpha) + alpha * 255.0
        else:
            img_out = img * (1 + alpha)
        return img_out / 255.0
    for file_name in os.listdir('./brightness/007'):
        img=io.imread('./brightness/007/'+file_name)
        print(img.shape)
        # -255.0 - 255.0 alpha -1.0 - 1.0
        Increment = -105.0;
        alpha = Increment/255.0;
        img_out = Illumi_adjust(alpha, img)
        io.imsave('./cutPics/test/007/'+file_name, img_out)

def get_data_from_log(log_path, tag_name='val_acc'):
	#加载日志数据
    ea=event_accumulator.EventAccumulator(log_path)
    ea.Reload()
    val_acc = ea.Scalars('val_acc')
    items = []
    for item in val_acc:
        items.append(item.value)
    return items
    # print("wall_time","step","value")
    # for item in val_acc:
	 #    print(item.wall_time, item.step,item.value)


if __name__ == '__main__':
    # download_data()
    # dist_log_data()
    # predict_all()
    get_data_from_log()