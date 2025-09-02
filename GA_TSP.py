# 导入常用库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from time import perf_counter
from sko.GA import GA_TSP
import math

# 存储路径 save path
path = 'G:\竞赛学术\双智竞赛\智慧城市竞赛\成果\论文数据\区域六_GATSP'
time_cost = 0
cost2 = 0
# 数据路径 data path
datas = pd.read_excel("G:\竞赛学术\双智竞赛\智慧城市竞赛\区域六"
                      ".xls")
points1 = np.array(datas[['X坐标', "Y坐标"]])

def get_xyx(treelist):
    list_x1 = [];list_y1 = [];list_x2 = [];list_y2 = [];
    list_z1 = []; list_z2 = []
    # 将路径转化为xy坐标形式
    for j in range(0, len(treelist) - 1):
        list_x1.append(treelist[j][0])
        list_y1.append(treelist[j][1])
        list_z1.append(0)
    for j in range(1, len(treelist)):
        list_x2.append(treelist[j][0])
        list_y2.append(treelist[j][1])
        list_z2.append(0)
    return list_x1, list_y1, list_z1, list_x2, list_y2, list_z2

def recost(cost):
    return 2 / (1 + math.exp(-cost)) - 1

def is_interstance(x1,y1,x2,y2,x3,y3,x4,y4):
    u = [x2 - x1, y2 - y1]
    v = [x4 - x3, y4 - y3]
    w = [x1 - x3, y1 - y3]
    #计算向量
    cross_1 = u[0]*v[1]-u[1]*v[0]
    cross_2 = u[0]*w[1]-u[1]*w[0]
    cross_3 = v[0]*w[1]-v[1]*w[0]
    if cross_1 == 0:
        if cross_2 != 0 or cross_3 != 0:
            return False
        else:
            if x1 < x2:
                if x2 < x3 or x4 < x1:
                    return False
            else:
                if x1 < x3 or x4 < x2:
                        return False
            return True
    else:
        s = cross_2/cross_1
        t = cross_3/cross_1
        if 0 <= s <= 1 and 0 <= t <= 1:
            return True
        else:
            return False

def Tree_interstance(points,routine,linex1,liney1,linex2,liney2,cost,num):
    for i in range(0, (len(routine) - 1)):
        n = 0
        for j in range(0, len(linex1)):
            if is_interstance(points[routine[i]][0],points[routine[i]][1],points[routine[i+1]][0],points[routine[i+1]][1],linex1[j],liney1[j],linex2[j],liney2[j]):
                cost += cost/num
                n = 1
            if n == 0:
                cost -= 0.02*cost/num
    return cost

def findway(points_coordinate, num_points):
    start = perf_counter()
    treelist = []
    def cal_total_distance(routine):
        '''计算总距离：输入路线,返回总距离.
        cal_total_distance(np.arange(num_points))
        '''
        list_x1 = [];list_y1 = [];list_x2 = [];list_y2 = []
        num_points, = routine.shape
        cost = 0
        for i in range(1, num_points):
            cost += math.sqrt(math.pow(abs(points_coordinate[i][0] - points_coordinate[i - 1][0]), 2)) \
                    + math.sqrt(math.pow(abs(points_coordinate[i][1] - points_coordinate[i - 1][1]), 2))
            """if i == 1:
                list_x1.append(points_coordinate[i][0])
                list_x2.append(points_coordinate[i - 1][0])
                list_y1.append(points_coordinate[i][1])
                list_y2.append(points_coordinate[i - 1][1])
            else:
                cost = Tree_interstance(points_coordinate, routine, list_x1, list_y1, list_y2, list_y2, cost,
                                        num_points)
                list_x1.append(points_coordinate[i][0])
                list_x2.append(points_coordinate[i - 1][0])
                list_y1.append(points_coordinate[i][1])
                list_y2.append(points_coordinate[i - 1][1])
                recost(1 / (1 + math.exp(cost * -2)))"""
        return cost
    # 执行遗传(GA)算法
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=20, max_iter=100, prob_mut=0.6)  # 调用工具箱

    # 结果输出
    best_points, best_distance = ga_tsp.run()
    dist = best_distance[0]
    print("运行时间是: {:.5f}s".format(perf_counter() - start))  # 计时结束
    print("最优距离为:", recost(1 / (1 + math.exp(dist * -2))))
    fig2, ax2 = plt.subplots(1, 1)
    ax2.set_title('Optimization process', loc='center')  # 优化过程
    ax2.plot(ga_tsp.generation_best_Y)
    ax2.set_xlabel("代数")
    ax2.set_ylabel("最优值")
    plt.show()
    #将结果转化为坐标放如treelist当中
    for i in range(0, len(best_points)):
        treelist.append(points_coordinate[best_points[i]])
    return treelist, dist

if __name__ == "__main__":
    waylist, dist = findway(points1, points1.shape[0])
    list_x1, list_y1, list_z1, list_x2, list_y2, list_z2 = get_xyx(waylist)
    Datawrite = {
        'X1坐标': list_x1,
        'Y1坐标': list_y1,
        'X2坐标': list_x2,
        'Y2坐标': list_y2,
        'Z1坐标': list_z1,
        'Z2坐标': list_z2,
    }
    fwrite = pd.DataFrame(Datawrite)
    fwrite.to_excel(path + "\\Route.xlsx", index=False)