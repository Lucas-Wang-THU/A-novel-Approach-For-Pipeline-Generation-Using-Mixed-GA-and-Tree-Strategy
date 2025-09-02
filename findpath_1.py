import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from time import perf_counter
from sko.GA import GA_TSP
import sko
import math
import os

path = 'G:\竞赛学术\双智竞赛\智慧城市竞赛\成果\论文数据/区域七_分支点4_一次扩充'
time_cost = 0
cost2 = 0
datas = pd.read_excel("G:\竞赛学术\双智竞赛\智慧城市竞赛\区域七.xls")
points1 = np.array(datas[['X坐标', "Y坐标"]])
pass_points = points1[0:-2]
num = 0
list_x1 = []; list_y1 =[]; list_x2 = []; list_y2 = []

def intinal_treepoint(poingtlist, n):
    #初始化随机生成分支点
    if isinstance(n, int):
        num = random.sample(range(0,len(poingtlist)), n)
    else:
        num = []
        for i in range(0,len(n)):
            if int(n[i]) not in num:
                num.append(i)
            else:
                for j in range(0, len(poingtlist)):
                    if j not in num:
                        num.append(j)
                        break
    treepoints  = []
    for i in range(0, len(num)):
        treepoints.append(poingtlist[num[i]])
        poingtlist = np.delete(poingtlist, num[i], axis=0)
    return treepoints, poingtlist

def get_distance_point2line(point, line):
    """求点到直线的距离
    Args:
        point: [x0, y0]
        line: [x1, y1, x2, y2]
    """
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance

def find_mainline(start, treepoints, points, end, dist):
    #找到主干线
    list_1 = [];
    list_1.append(start)
    for i in range(0, len(treepoints)):
        if i == 0:
            line = np.concatenate([start,treepoints[i]])
        elif i == len(treepoints) + 1:
            line = np.concatenate([treepoints[i],end])
        else:
            line = np.concatenate([treepoints[i - 1],treepoints[i]])
        j = 0
        while j < len(points):
            distance = get_distance_point2line(points[j], line.tolist())
            if distance <= dist:
                list_1.append(points[j])
                points = np.delete(points, j, axis=0)
            j += 1
        list_1.append(treepoints[i])
    list_1.append(end)
    return list_1, points

def find_mainline_none(start, treepoints, points, end):
    #找到主干线
    list_1 = [];
    list_1.append(start)
    for i in range(0, len(treepoints)):
        list_1.append(treepoints[i])
    list_1.append(end)
    return list_1, points

def find_mainline_again(mainpoints, points, dist):
    #重新找到主干线
    list_1 = [];
    list_1.append(mainpoints[0])
    for i in range(1, len(mainpoints)):
        line = np.concatenate([mainpoints[i - 1], mainpoints[i]])
        j = 0
        while j < len(points):
            distance = get_distance_point2line(points[j], line.tolist())
            if distance <= dist:
                list_1.append(points[j])
                points = np.delete(points, j, axis=0)
            j += 1
        list_1.append(mainpoints[i])
    list_1.append(mainpoints[-1])
    return list_1, points

def find_tr_passpoint(treepoints, points):
    #找到每一个点对应的分支
    list_2 = []
    for j in range(0, len(points)):
        n = 0
        distance = abs(points[j][0] - treepoints[0][0]) + abs(points[j][1] - treepoints[0][1])
        for i in range(0, len(treepoints)):
            dist = abs(points[j][0] - treepoints[i][0]) + abs(points[j][1] - treepoints[i][1])
            if dist <= distance:
                n = i
                distance = dist
        list_2.append(n)
    return list_2

def release_tr_passpoints(points, pointlist, treepoints, n):
    #获得实际坐标
    list_l = [];list_r = [];list_H = [];list_L = [];
    for i in range(0, len(points)):
        if pointlist[i] == n:
            if treepoints[n][1] < points[i][1] and treepoints[n][0] < points[i][0]:
                list_l.append(points[i])
            elif treepoints[n][1] > points[i][1] and treepoints[n][0] < points[i][0]:
                list_r.append(points[i])
            elif treepoints[n][1] > points[i][1] and treepoints[n][0] > points[i][0]:
                list_H.append(points[i])
            else:
                list_L.append(points[i])

    return list_l, list_r, list_H, list_L

def waydist(points):
    #计算该路径的总长
    sum = 0
    for i in range(1, len(points)):
        sum += math.sqrt(math.pow(points[i][0] - points[i - 1][0], 2) + math.pow(points[i][1] - points[i - 1][1], 2))
    return sum


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

def interstance(linex1,liney1,linex2,liney2,cost,num):
    for i in range(0, (len(linex1) - 1)):
        n = 0
        for j in range((i + 1), len(linex1)):
            if is_interstance(linex1[i],liney1[i],linex2[i],liney2[i],linex1[j],liney1[j],linex2[j],liney2[j]):
                cost += cost/num
                n = 1
            if n == 0:
                cost -= 0.5*cost/num
    return cost

def Tree_interstance(points,routine,linex1,liney1,linex2,liney2,cost,num):
    for i in range(0, (len(routine) - 1)):
        n = 0
        for j in range(0, len(linex1)):
            if is_interstance(points[routine[i]][0],points[routine[i]][1],points[routine[i+1]][0],points[routine[i+1]][1],linex1[j],liney1[j],linex2[j],liney2[j]):
                cost += cost/num
                n = 1
            if n == 0:
               cost -= 0.2*cost/num
    return cost

def print_route(best_points):
    result_cur_best = []
    for i in best_points:
        result_cur_best += [i]
    for i in range(len(result_cur_best)):
        result_cur_best[i] += 1
    result_path = result_cur_best
    return result_path

def release_list(points, treepoints, n):
    #重构分支路径需要通过的点
    points.insert(0, treepoints[n])
    b = np.array(points)
    return b

def findmainway(points):
    #对主干进行排序
    point = np.array((1, 2))
    for i in range(1, (len(points))):
        n = i
        dist1 = abs(points[i - 1][0] - points[i][0]) + abs(points[i - 1][1] - points[i][1])
        for j in range((i + 1), (len(points))):
            dist2 = abs(points[i - 1][0] - points[j][0]) + abs(points[i -1][1] - points[j][1])
            if dist2 <= dist1:
                dist1 = dist2
                n = j
                point = points[n]
                points[n] = points[i]
                points[i] = point
    return points

def findtreeway(points_coordinate, num_points):
    global list_x1,list_y1,list_x2,list_y2
    if num_points > 6:
        treelist = []
        def cal_total_distance(routine):
            '''计算总距离：输入路线,返回总距离.
            cal_total_distance(np.arange(num_points))
            '''
            num_points, = routine.shape
            cost = 0
            for i in range(1, num_points):
                cost += abs(points_coordinate[i][0] - points_coordinate[i - 1][0]) + abs(points_coordinate[i][1] - points_coordinate[i - 1][1])
            cost = Tree_interstance(points_coordinate,routine,list_x1,list_y1,list_y2,list_y2,cost,num_points)
            return cost
        # 执行遗传(GA)算法
        ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=20, max_iter=50, prob_mut=0.8)  # 调用工具箱

        # 结果输出
        best_points, best_distance = ga_tsp.run()
        dist = best_distance[0]
        #将结果转化为坐标放如treelist当中
        for i in range(0, len(best_points)):
            treelist.append(points_coordinate[best_points[i]])
    else:
        treelist = findmainway(points_coordinate.tolist())
        dist = waydist(treelist)
    return treelist, dist

def cost(sum1, sum2):
    #真正的目标函数,sum1为主干，sum2为分支
    return 1/(1+math.exp(sum1*-10+sum2*-0.9))

def recost(cost):
    return 2 / (1 + math.exp(-cost)) - 1

def findway(list1, sum2, list_x1, list_y1, list_x2, list_y2):
    if list1.shape[0] > 1:
        treelist, dist = findtreeway(list1, list1.shape[0])
        sum2 += dist
        # 将路径转化为xy坐标形式
        for j in range(0, len(treelist) - 1):
            list_x1.append(treelist[j][0])
            list_y1.append(treelist[j][1])
        for j in range(1, len(treelist)):
            list_x2.append(treelist[j][0])
            list_y2.append(treelist[j][1])
        return sum2, list_x1, list_y1, list_x2, list_y2
    else:
        return sum2, list_x1, list_y1, list_x2, list_y2

def findallway(x):
    global cost2, num, time_cost, list_x1, list_y1, list_x2, list_y2
    start = perf_counter()  # 计时开始
    num += 1
    start_point = points1[-1]
    end_point = points1[-2]
    pass_points = points1[0:-2]
    sum1 = 0; sum2 =0;
    list_x1 = []; list_y1 =[]; list_x2 = []; list_y2 = []
    treepoints, pass_points = intinal_treepoint(pass_points, x[0:(len(x)-1)])
    main_points, pass_points = find_mainline(start_point, treepoints, pass_points, end_point, x[-3])
    #main_points, pass_points = find_mainline_none(start_point, treepoints, pass_points, end_point)
    main_points = findmainway(main_points)
    main_points, pass_points = find_mainline_again(main_points,pass_points, x[-2])
    main_points = findmainway(main_points)
    main_points, pass_points = find_mainline_again(main_points, pass_points, x[-1])
    main_points = findmainway(main_points)
    dist = waydist(main_points)
    sum1 += dist
    #将路径转化为几条直线的坐标
    for i in range(0, len(main_points) - 1):
        list_x1.append(main_points[i][0])
        list_y1.append(main_points[i][1])
    for i in range(1, len(main_points)):
        list_x2.append(main_points[i][0])
        list_y2.append(main_points[i][1])
    list_2 = find_tr_passpoint(treepoints, pass_points)
    for i in range(0, len(treepoints)):
        list_l, list_r, list_H, list_L = release_tr_passpoints(pass_points, list_2, treepoints, i)
        list_l = release_list(list_l, treepoints, i)
        sum2, list_x1, list_y1, list_x2, list_y2 = findway(list_l, sum2, list_x1, list_y1, list_x2, list_y2)
        list_r = release_list(list_r, treepoints, i)
        sum2, list_x1, list_y1, list_x2, list_y2 = findway(list_r, sum2, list_x1, list_y1, list_x2, list_y2)
        list_H = release_list(list_H, treepoints, i)
        sum2, list_x1, list_y1, list_x2, list_y2 = findway(list_H, sum2, list_x1, list_y1, list_x2, list_y2)
        list_L = release_list(list_L, treepoints, i)
        sum2, list_x1, list_y1, list_x2, list_y2 = findway(list_L, sum2, list_x1, list_y1, list_x2, list_y2)

    # 存储数据
    time_cost += perf_counter() - start
    if num % 100 == 0:
        print(f"epochs:{num},运行时间:{float(time_cost/60)}min")
    list_1 = []
    for i in range(0, len(list_x1)):
        list_1.append(0)
    cost1 = cost(sum1, sum2)
    cost1 = interstance(list_x1, list_y1, list_x2, list_y2, cost1, len(list_x1))
    if len(main_points) > (len(treepoints)+6):
        cost1 *= (len(main_points)-len(treepoints)-4)
    else:
        cost1 *= 0.9
    if cost2 == 0:
        cost2 = cost1
    if cost1 <= cost2:
        Datawrite = {
            'X1坐标': list_x1,
            'Y1坐标': list_y1,
            'X2坐标': list_x2,
            'Y2坐标': list_y2,
            'Z1坐标': list_1,
            'Z2坐标': list_1,
        }
        fwrite = pd.DataFrame(Datawrite)
        fwrite.to_excel(path+"\\Route.xlsx", index=False)
        cost2 = cost1
    return recost(cost1)

if __name__ == "__main__":
    x=[];
    os.mkdir(path)
    a = input("需要几个分叉点：")
    for i in range(0, int(a)):
        x.append(i)
    x.append(random.randrange(0, 5000))
    #x.append(random.randrange(0, 500))
    #x.append(random.randrange(0, 200))
    y = list(0 for i in range(int(a)+1))
    z = []
    for i in range(int(a)):
        z.append(len(pass_points))
    z.append(5000)
    #z.append(500)
    #z.append(500)
    print(len(y),len(z))
    #z.append(500)
    # 使用遗传算法求解目标函数的最小值
    # 构造遗传算法对象 ex_GA
    ex_GA = sko.GA.GA(func=findallway, n_dim=(int(a)+1), size_pop=6, max_iter=100, prob_mut=0.6,
                      lb=y, ub=z, precision=1e-2)
    # 执行遗传算法
    best = []
    best = ex_GA.run()
    print("总运行时间是: {:.5f}min".format((time_cost/60)))
    # 输出找到的最优解
    for i in range(0, len(best) - 1):
        print(f"Best point:{best[i]}")
    print(f"Best dist is:{best[-1]}")
    with open(path+"\\route_best.txt", mode= "w" ) as tf:
        tf.write(f"Best dist is:{best[-1]}")
        tf.close()
    # 可视化迭代过程
    # 获取历史搜索最佳解序列
    Y_history = ex_GA.all_history_Y
    # 创建画布对象和子图对象
    fig, ax = plt.subplots(2, 1)
    # 绘制搜索的目标函数值的历史曲线
    ax[0].plot(range(len(Y_history)), Y_history, '.', color='red')
    # 计算历史最小值的累积曲线
    min_history = [np.min(Y_history[:i + 1]) for i in range(len(Y_history))]
    # 绘制历史最小值的累积曲线
    ax[1].plot(range(len(min_history)), min_history, color='blue')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    # 保存并显示图像
    plt.savefig(path+"\\GA_path"+a+".png", dpi=400)
    plt.show()