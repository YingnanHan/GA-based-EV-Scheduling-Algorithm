
# 导入相关的包
import numpy as np
import random
import copy
import pandas as pd
import time


# 定义遗传算法类
class EVCSIndividual: # EVCS指的是充电站

    """
    individual of demand oriented method
    """

    def __init__(self,data,ev_ct,ev_cp,ev_n):
        # data 各个省市电车数据
        # ev_ct 电车充电时间
        # ev_cp 电车充电价格
        # ev_n  充电站数目
        self.data = data # 所有充电车的数据
        self.evcs = np.zeros(ev_n) # 记录每个充电站的车数
        self.individual = [] # 定义染色体 每一个解对应于一个充电桩分配模式
        self.fitness = [] # 存储当前个体的适应度
        self.ev_ct = ev_ct # 电车的充电时间
        self.ev_cp = ev_cp # 电车的充电价格
        self.ev_n = ev_n # 单位时间内每一个充电桩可以服务的电车数目

    def GenerateIndividual(self): ## 找到一个可行解(个体)  对每一个汽车找到一个最近充电站 暂时充当局部最优解
        '''
        generate a random chromosome for genetic algorithm
        '''
        gene = list() # gene是一个记录当前所有电车的局部最优解的二维列表 其规模为 <num of EV>X3
        for i in range(len(self.data)): # 对所有的充电车进行操作
            ind = sorted(self.data[i],key=lambda x:[x[1]])
            copy_ind = copy.deepcopy(ind[0]) # 在每一个电车的候选充电站列表中选择一个距离最近的作为当前的候选局部最优解
            copy_ind.append(i) # 给电车到充电站的距离大小排序后加上排序后的序号
            gene.append(copy_ind) # 在gene基因型当中加入指定车的当前最优解信息
        self.individual = gene # 将当前取得的局部最优解赋值给self.individual变量中
        # print(gene) # 输出结果是一个二维数组 每一行表示 第i号电车到当前距离最近的充电站的<编号>以及<距离> gene[i][2]所表示的电车在新的局部可行解下的编号,目前的算法版本不会使用到
        # 关于gene[i][2]的问题 我们对其进行排序之后一行的数据相对位置不会变 变动的是列方向的顺序 所以根据充电时间列表还会正确找到相应的充电时间

    # 本算法的重点环节
    def calculateFitness(self,cd_n = 5):  # cd_n指的是当前每一个充电站下充电桩的个数(类比一下插座和插孔关系)
        """
        计算个体的适应度，个体就是一个可行解
        """
        ev_n = np.zeros((self.ev_n,cd_n)) # self.ev_n：电站数目
        cd_wt = np.zeros((self.ev_n,cd_n)) # 一个充电桩的工作时间
        cd_st = np.zeros((self.ev_n,cd_n)) # 一个电桩的空闲时间
        st = 0 # 电桩空闲总时间
        qt = 0 # 电车排队总时间
        tt = 0 # 充电总时间 到充电桩的时间+排队时间+充电时间
        cost = 0 # 充电总成本
        n = []
        x_n = 0 # 排队中电车的数目
        for i in range(len(self.individual)): #对所有的个体下的基因i 也就是每一辆电车
            '''所有电车充电的总花费(收益) += i号车的行驶距离(假设电车速度恒定)*1元/km + i号电车所找到的充电站充电单价*该辆车的充电时间 '''
            cost += self.individual[i][1]+self.ev_cp[self.individual[i][0]]*self.ev_ct[self.individual[i][2]] # 这里为什么使用序号self.individual[i][2]描述电车的序号？ 因为给定的数据是固定的我们为其编码后可以快速找到某一个电车的充电时间
            k=int(self.evcs[self.individual[i][0]]%cd_n) # 本算法在这里根据充电站编号随机选择一个充电桩号
            self.evcs[self.individual[i][0]]+=1 # 找到具体的充电站之后，该充电站下面的待充电电车就增加1
            if cd_wt[self.individual[i][0]][k]<60: #如果当前找到的充电站的第k个充电桩的等待时间小于60个单位
                ev_n[self.individual[i][0]][k]+=1 # 那么就令self.individual[i][0]号充电站中的第k个充电桩等待充电的电车数量加一

            # 如果车辆到达充电站所需要的时间小于所分配充电桩的等待时间
            if self.individual[i][1]<cd_wt[self.individual[i][0]][k]: #k是充电桩的桩号
                tt += cd_wt[self.individual[i][0]][k]+self.ev_ct[self.individual[i][0]] # 那么总时间消耗就等于等待时间加上充电时间
                qt += cd_wt[self.individual[i][0]][k]-self.individual[i][1] # 排队时间就等于总共电桩等待的时间减去路上花的时间
                cd_wt[self.individual[i][0]][k]=cd_wt[self.individual[i][0]][k]+self.ev_ct[self.individual[i][0]]
                x_n += 1 # 此时需要排队，在对中排序的电车数目加一
            else: # 如果车辆到达充电站所需要的时间大于所分配充电桩的等待时间
                tt += self.individual[i][1]+self.ev_ct[self.individual[i][0]] #那么总时间消耗就等于等待时间加上充电时间
                st += self.individual[i][1]-cd_wt[self.individual[i][0]][k] # 总空闲时间
                cd_st[self.individual[i][0]][k]+=self.individual[i][1]-cd_wt[self.individual[i][0]][k] # 电桩的空闲时间就等于行驶时间减去电车在充电站self.individual[i][0]的充电桩k下所需要的等待时间
                cd_wt[self.individual[i][0]][k]=self.individual[i][1]+self.ev_ct[self.individual[i][0]] #i号电车所选充电站的第k个充电桩的等待时间就等于电车在路上的时间加上充电时间

        # 计算总收益 假设每一单单位时间收一块
        revenue=0 # 总收益
        t_ev_n=0 # 单位时间每个充电站充点电车数
        t_st_v=0 # 总浪费(空闲时间)
        for i in range(self.ev_n): # ev_n 电站数目
            for j in range(cd_n): # cd_n 电桩数目
                revenue+=(cd_wt[i][j]-cd_st[i][j])*self.ev_cp[i] # 根据上面的计算结果合计总收入 (包含成本)
                t_ev_n+=ev_n[i][j] # 按照每一个充电桩下总共的充电电车数进行一个数目统计
                t_st_v+=cd_st[i][j] # 对每一个充电桩的空闲时间进行统计

        n.append(revenue)  #总收益 [0]
        n.append(cost)     #总成本 [1]
        n.append(tt)       #总时间 [2]
        n.append(qt)       #总排队时间 [3]
        n.append(st)       #总空闲时间 [4] ≈ 后面几套代码的idle time
        n.append(t_ev_n)   #单位时间每个充电桩充电电车数 [5]

        self.fitness=np.array(n) #用于后期计算出适应度


def main():
    # 读取34省充电桩数据
    data = pd.read_csv("chargingstations.csv",delimiter=";",header=None).values
    cities = data[:,1:] #全国34所城市的经纬度坐标
    cities_name = data[:,0] # 全国34所城市的编号 每一个编号对应着一个经纬度坐标
    city_size = data.shape[0] # city_size=34 充电桩点的位置
    locations = np.arange(cities.shape[0]) # 问题的解集合
    '''读取电车数据'''
    ev_data_1000 = pd.read_csv('ev_data_1000.csv').values
    ev_x = ev_data_1000[:,1:2] # 当前各个电车的X坐标
    ev_x = [j for i in ev_x for j in i] # 将X坐标从二维数组形式转化为一维列表的形式
    ev_y = ev_data_1000[:,2:3] # 当前各个电车的Y坐标
    ev_y = [j for i in ev_y for j in i] # 将Y坐标从二维数组形式转化为一维列表的形式
    ev_ld= ev_data_1000[:,3:4]  # 当前某一辆电车所对应的剩余电量所可以支撑的剩余里程
    ev_ld= [j for i in ev_ld for j in i] # 将ev_ld从二维数组形式转化为一维列表的形式
    ev_ct= ev_data_1000[:,4:5]  # 每辆电车所需要的充电时间
    ev_ct= [j for i in ev_ct for j in i] # 将ev_ct从二维数组形式转化为一维列表的形式
    '''读取电价数据'''
    ev_cp_34 = pd.read_csv('ev_cp_34.csv').values # 读取存储充点电价的csv文件，其中第二列为全国各个省会城市的充电桩充电电价 排序方式与充电桩坐标相同
    ev_cp=ev_cp_34[:,1:] # ev_cp 充电电价
    ev_cp=[j for i in ev_cp for j in i] # 将充电电价从二维数组转化为一维列表的形式
    # print(ev_ct)

    t_c_l = [] # 每个电车的候选充电站 依据当前剩余行驶里程决定
    for i in range(1000): # 对当前所有的1000个还在行驶的电车
        c_l = []  # 用来存储当前编号为i的电车 的可到达的所有充电桩序号 到达充电桩的时间
        for j in range(34): # 针对目前仅存的所有的34个充电桩
            d = np.sqrt((ev_x[i] - cities[j][0]) ** 2 + (ev_y[i] - cities[j][1]) ** 2)
            if d<ev_ld[i]: # ev_ld应该是当前电车剩余电量可行驶里程

                c = [] # 做一个临时变量用于临时存取
                c.append(j) # 充电桩序号
                c.append(d) # 距离充电桩距离，也可作为行驶时间
                c_l.append(c)
        t_c_l.append(c_l) # 将当前编号的电车所对应的充电桩序号 距离等信息加到列表中 其最终结果是一个三维列表


    for i in range(len(t_c_l)-1,-1,-1): #为了遗传算法的顺利进行，将不包含数据的数组移除
        if t_c_l[i] ==[]: # 如果列表中的某一个行是空的 就将其对应的所有计算适应度的指标删除
            del ev_x[i]
            del ev_y[i]
            del ev_ld[i]
            del ev_ct[i]
            del t_c_l[i]

    ind = EVCSIndividual(t_c_l,ev_ct,ev_cp,len(cities)) # 初始化遗传算法类
    ind.GenerateIndividual() # 找到一个局部最优解
    ind.calculateFitness() # 计算当前情况下局部最优解的适应度
    print("True Fitness:", ind.fitness)
    print("Fitness:", 0.3 * (ind.fitness[3] / 40000) + 0.4 * (ind.fitness[4] / 3000) + 0.3 * (1 - ind.fitness[5] / 899)) # 与论文中的公式对应 -- 可能存在不完全对应的情况
    print("Average of queuing time (minute):", ind.fitness[3] / 899) # 计算平均排队时间
    print("Average of idle time (minute):", ind.fitness[4] / 170) # 计算所有充电桩的总平均空闲时间
    print("Number of charged EV within an hour:", int(ind.fitness[5])) # 得到单位时间内充点电车的数目 用于计算效率
    print("Total cost of all EV:", ind.fitness[1])  # 电车充电的总成本
    print("Total revenue of EVCS:", ind.fitness[0]) # 电车充电的总收益


if __name__ == '__main__':
    start=time.time() # 开始计时 计算算法执行时间
    main() # 遗传算法主函数
    end=time.time() # 停止计时 计算算法执行时间
    print("Time consuming (second ):",end-start) # 输出整个算法执的耗时