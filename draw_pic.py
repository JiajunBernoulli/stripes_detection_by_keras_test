import os, shutil
from config import NUM_CLASS
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

from data_handle import get_data_from_log


def draw_confusion_matrix(networkName, idx=None, realNum=[60, 70, 55, 55, 60, 60]):
    # 生成相应的真实标签
    realLabels = np.zeros(shape=(1, realNum[0]))
    for i in range(1, len(realNum)):
        realLabel = np.ones(shape=(1, realNum[i]))*i
        realLabels = np.hstack((realLabels, realLabel))
    print(realLabels)
    # 加载预测的结果
    results = np.load('./predicts/'+networkName+'.npy')
    count = 0
    # 空矩阵用于填充为混淆矩阵
    resMat = np.zeros(shape=(NUM_CLASS, NUM_CLASS))
    # 每行结果为一个样本属于某一类的可能性列表
    for i in range(0, len(results)):
        realLabel = int(realLabels[0,i])
        print(results[i])
        print(results[i].argmax() ,realLabel)
        resMat[results[i].argmax(), realLabel] += 1
        count += 1
    # i不为空，绘制为子图
    if idx != None:
        sns.heatmap(resMat, annot=True, cmap='rainbow', ax=plt.subplot(121 + i))
    # i为空，直接绘制
    else:
        sns.heatmap(resMat, annot=True, cmap='rainbow')
    plt.title(networkName+' ConfusionMatrix')
    plt.xlabel('real')
    plt.ylabel('predict')
    plt.show()
    # plt.savefig('./pics/'+networkName+' Confusion Matrix.png')
def draw_many_matrix():
    i = 0
    for networkNames in os.listdir('./models/ResNet34'):
        print(networkNames)
        draw_confusion_matrix(networkNames, i)
        i += 1
    plt.show()

def draw_property_curve(dir, i):
    files = os.listdir(dir)
    titles = []
    for file in files:
        titles.append(file[:-4])
    print(titles)
    df = pd.DataFrame(columns=titles)
    for file in files:
        data = pd.read_csv(dir+'/'+file, nrows=20)
        df[file[:-4]] = data['Value']
    df.plot(ax=plt.subplot(221 + i))
    title = os.path.split(dir)[1]
    plt.title(title+' performance curve')
    plt.ylabel(title)
    plt.xlabel('epochs')
    plt.xticks(np.arange(0, 20, step=4))
def draw_many_curve_pic():
    i = 0
    for dir in os.listdir('./csvs'):
        print('./csvs/'+dir)
        draw_property_curve('./csvs/'+dir, i)
        plt.tight_layout()
        i += 1
    plt.show()

def evalu_topk(networkName,k):
    results = np.load('./predict/'+networkName+'.npy')
    count = 0
    rightCount = 0
    for result in results:
        temp = sorted(range(len(result)), key=lambda i:result[i], reverse=True)
        toplist = temp[0:k]
        print(toplist)
        if count // 25 in toplist:
            rightCount += 1
        count += 1
    print(networkName, rightCount / count)
    return rightCount / count
def evalu_many_topk(K):
    networkNames = os.listdir('./models')
    n = len(networkNames)
    matrix = np.zeros(shape=(K, n))
    for k in range(1, K+1):
        for j in range(0, n):
            topk_acc = evalu_topk(networkNames[j], k)
            matrix[k-1, j] = topk_acc
    draw_many_topk_bar(networkNames, matrix, K)
def draw_many_topk_bar(networks, num_list, K):
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / float(5), 1.01 * height, '%s' % round(float(height), 2))
    # num_list = np.random.rand(k, len(networks))
    x = list(range(len(networks)))
    total_width, n = 0.7, len(networks)
    width = total_width / n
    for count in range(1, K+1):
        rects=plt.bar(x, num_list[count-1, :], width=width, label='top-'+str(count), tick_label=networks)
        autolabel(rects)
        for i in range(len(x)):
            x[i] = x[i] + width
    plt.legend()
    plt.title('top-k')
    plt.xlabel('network')
    plt.ylabel('acc')
    plt.show()

def compare_step_epoch():
    i = 0
    for dir in os.listdir('./step_epoch'):
        df = pd.DataFrame()
        for file in os.listdir('./step_epoch/'+dir):
            data = pd.read_csv('./step_epoch/'+dir+'/'+file)
            df[file[:-4]] = data['Value']
        df.plot(ax=plt.subplot(221+i))
        plt.title(dir)
        plt.ylabel('acc')
        plt.xlabel('epoch')
        i += 1
    plt.show()

def draw_many_acc_curve(exp_typ):
    i = 0
    for exp_name in ['left', 'right', 'face']:
        exp_group = {}
        logs_path = './'+exp_name+'/history/'+exp_typ+'/logs/'
        for dir in os.listdir(logs_path):
            for file in os.listdir(logs_path+dir):
                val_acc =get_data_from_log(log_path=logs_path+dir+'/'+file)
                exp_group[dir] = val_acc
        df = pd.DataFrame(exp_group)
        df.to_csv(logs_path+exp_name+'.csv')
        ax = plt.subplot(311 + i)
        ax.set_title(exp_name, fontsize=25)
        ax.set_ylabel('acc', fontsize=20)
        df.plot(ax = ax)
        i += 1
    plt.xlabel('step', fontsize=20)
    plt.suptitle(exp_typ, fontsize=30)
    plt.show()

    # print(df.head(5))
if __name__ == '__main__':
    # draw_many_matrix()
    # draw_many_curve_pic()
    # evalu_many_topk(3)
    # draw_confusion_matrix('ResNet34_0_2')
    # evalu_topk('experimental group2', 1)
    # plt.show()
    draw_many_acc_curve('dropout')
