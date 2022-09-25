
# 导入相关包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import random


# 定义电车-充电桩算法解的个体类
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
        self.individual = [] # 定义染色体 每一个解对应于一个充电桩的分配模式
        self.fitness = [] # 存储当前个体适应度
        self.ev_ct = ev_ct # 电车充电时间
        self.ev_cp = ev_cp # 电车的充电价格
        self.ev_n = ev_n # 单位时间内每一个充电桩可以服务的电车数

    def GenerateIndividual(self): ## 找到一个可行解(个体)   对每一个汽车找到一个随机的充电站 暂时充当局部最优解
        '''
        generate a random chromosome for genetic algorithm
        '''
        gene = list() # gene是一个记录当前所有电车的局部最优解的二维列表 其规模为 <num of EV>X3
        for i in range(len(self.data)): # 对所有的充电车进行操作
            ind = random.choices(self.data[i]) # 所有的电车随机选择一个充电站
            copy_ind = copy.deepcopy(ind[0]) # 在每一个电车的候选充电站列表中选择第一个充电站作为当前的候选局部最优解
            copy_ind.append(i) # 为电车加上电车序号，用于后续操作 满足与其他算法的一致性
            gene.append(copy_ind) # 在gene基因型当中加入指定车的当前最优解信息
        self.individual = gene
        # print(gene) # 输出结果是一个二维数组 每一行表示 第i号电车到当前距离最近的充电站的<编号>以及<距离> gene[i][2]所表示的电车在新的局部可行解下的编号,目前的算法版本不会使用到

    # 本算法的终点环节
    def calculateFitness(self,cd_n=5): # cd_n指的是当前每一个充电站下充电桩的个数(类比一下插座和插孔关系)
        """
        计算个体的适应度，个体就是一个可行解
        """
        ev_n = np.zeros((self.ev_n,cd_n)) # self.ev_n：电站数目
        cd_wt = np.zeros((self.ev_n,cd_n)) # 一个充电桩的工作时间
        cd_st = np.zeros((self.ev_n,cd_n)) # 一个电桩的空闲时间
        st = 0. # 电桩空闲总时间
        qt = 0. # 电车排队总时间
        tt = 0. # 充电总时间 到充电桩的时间+排队时间+充电时间
        cost = 0. # 充电总成本
        n = []
        x_n = 0 # 排队中电车的数目
        for i in range(len(self.individual)): #对所有的个体下的基因i 也就是每一辆电车
            '''所有电车充电的总花费(收益) += i号车的行驶距离(假设电车速度恒定)*1元/km + i号电车所找到的充电站充电单价*该辆车的充电时间 '''
            cost += self.individual[i][1]*1+self.ev_cp[self.individual[i][0]]*self.ev_ct[self.individual[i][2]]  # 这里为什么使用序号self.individual[i][2]描述电车的序号？ 因为给定的数据是固定的我们为其编码后可以快速找到某一个电车的充电时间
            k = int(self.evcs[self.individual[i][0]]%cd_n) # 本算法在这里根据充电站编号随机选择一个充电桩号
            self.evcs[self.individual[i][0]]+=1 # 找到具体的充电站之后，该充电站下面的待充电电车就增加1
            if cd_wt[self.individual[i][0]][k]<60: #如果当前找到的充电站的第k个充电桩的等待时间小于60个单位
                ev_n[self.individual[i][0]][k]+=1 # 那么就令self.individual[i][0]号充电站中的第k个充电桩等待充电的电车数量加一

            # 如果车辆到达充电站所需要的时间小于所分配的充电桩等待时间
            if self.individual[i][1]<cd_wt[self.individual[i][0]][k]: #k是充电桩的桩号
                tt += cd_wt[self.individual[i][0]][k]+self.ev_ct[self.individual[i][0]] # 那么总时间消耗就等于等待时间加上充电时间
                qt += cd_wt[self.individual[i][0]][k]-self.individual[i][1] # 排队时间就等于总共电桩等待的时间减去路上花的时间
                cd_wt[self.individual[i][0]][k]+=self.ev_ct[self.individual[i][0]]
                x_n += 1 # 此时需要排队，在对中排序的电车数目加一
            else: # 如果车辆到达充电站所需要的时间大于所分配充电桩的等待时间
                tt += self.individual[i][1]+self.ev_ct[self.individual[i][0]] #那么总时间消耗就等于等待时间加上充电时间
                st += self.individual[i][1]-cd_wt[self.individual[i][0]][k] # 总空闲时间
                cd_st[self.individual[i][0]][k]+=self.individual[i][1]-cd_wt[self.individual[i][0]][k] # 电桩的空闲时间就等于行驶时间减去电车在充电站self.individual[i][0]的充电桩k下所需要的等待时间
                cd_wt[self.individual[i][0]][k]=self.individual[i][1]+self.ev_ct[self.individual[i][0]] # i号电车所选充电站的第k个充电桩的等待时间就等于电车在路上的时间加上充电时间
        # 计算总收益 假设每一单单位时间收一块
        revenue = 0 # 总收益
        t_ev_n = 0 # 单位时间每个充电站充点电车数
        t_st_n = 0 # 总浪费(空闲时间)
        for i in range(self.ev_n): # ev_n 电站数目
            for j in range(cd_n): # cd_n 电桩数目
                revenue += (cd_wt[i][j]-cd_st[i][j])*self.ev_cp[i] # 根据上面的计算结果合计总收入 (包含成本)
                t_ev_n += ev_n[i][j] # 按照每一个充电桩下总共的充电电车数进行一个数目统计
                t_st_n += cd_st[i][j] # 对每一个充电桩的空闲时间进行统计


        n.append(revenue)           #总收益 [0]
        n.append(cost)              #总成本 [1]
        n.append(tt)                #总时间 [2]
        n.append(qt)                #总排队时间 [3]
        n.append(st)                #总空闲时间 [4] ≈ 后面几套代码的idle time
        n.append(t_ev_n)            #单位时间每个充电桩充电电车数 [5]

        self.fitness = np.array(n)  #用于后期计算出适应度

class GeneticAlgorithm: # 定义遗传算法类
    def __init__(self,t_c_l,ev_ct,ev_cp,n,c_rate=0.7,m_rate=0.3,pop_size=200,maxnum=20): # 初始化遗传算法执行下来所需要的所有变量
        self.pop_size = pop_size #设定初始种群规模
        self.fitness = np.zeros(self.pop_size) # 记录每一个种群中个体的适应度
        self.c_rate = c_rate # 初始化交叉率
        self.m_rate = m_rate # 初始化变异率
        self.maxiternum = maxnum # 设定最大迭代次数
        self.population = [] # 初始种群
        self.bestfitness = 0 # 当前最佳适应度
        self.besttruefitness = [] # 记录每一轮最佳个体的适应度值
        self.bestIndex = 0 # 记录最优局部解个体在种群中的下标
        self.bestgene = np.array([]) # 记录中群众最优基因
        self.trace = np.zeros((self.maxiternum,2)) # 记录种群中最优解的迭代过程 里面的元素分别是 当前最优个体的适应度 当前的平均适应度
        self.avefitness = 0. # 用于后期计算平均适应度

        self.data = t_c_l # fit算法用的数据
        self.ev_ct = ev_ct # 每辆电车的充电时间
        self.ev_cp = ev_cp # 每一个充电站的电价
        self.cs_n = n # 充电站数量
        self.individual = [] # 记录解(一个个体)

    def initialize(self): # 遗传算法中的初始化种群的操作
        for i in range(0,int(self.pop_size)): # 执行int(self.pop_size)次迭代，初始化int(self.pop_size)个可行解
            ind = EVCSIndividual(self.data,self.ev_ct,self.ev_cp,self.cs_n)# 初始化一个种群的个体
            ind.GenerateIndividual()# 产生一个实际的个体
            self.population.append(ind) # 将这个个体加入种群当中去

    def evaluation(self):# 计算当前种群中每一个个体的适应度
        for i in range(0,int(self.pop_size)):# 对所有的个体计算适应度
            self.population[i].calculateFitness() # 调用EVCSIndividual类方法计算用于计算适应度分量的函数
            self.fitness[i] = 0.3*(self.population[i].fitness[3]/40000)+0.4*(self.population[i].fitness[4]/3000)+0.3*(1-self.population[i].fitness[5]/899) # 按照论文中的表示方法计算出数值型的个体适应度

    def selection(self):# 对种群进行选择操作
        for i in range(self.pop_size): # 对所有的个体进行遍历
            if i!=self.bestIndex and self.fitness[i]>self.avefitness: # 如果选择出来的个体不是最优个体(最优个体需要保留是因为怕之后不再出现更优的个体)并且适应度大于平均适应度(适者生存)
                pi = self.cross(self.population[self.bestIndex].individual,self.population[i].individual) #
                self.population[i].individual = self.mutate(pi) # 交叉后的新个体
                self.population[i].calculateFitness() # 重新计算新个体的适应度分量
                self.fitness[i] = 0.3 * (self.population[i].fitness[3] / 40000) + 0.4 * (self.population[i].fitness[4] / 3000) + 0.3 * (1 - self.population[i].fitness[5] / 899)#依照适应度分量计算实际适应度值

    def crossoverMutation(self): #交叉变异的组合操作
        for j in range(self.pop_size): # 对所有的中群众的个体 执行交叉变异操作
            r = np.random.randint(0,self.pop_size-1) # 找到中群众的一个个体
            if j != r and j != self.bestIndex: # 如果这个个体不是当前最优的个体的话
                nind = self.cross(self.population[j].individual,self.population[r].individual) # 将此个体与随机选中的个体进行交叉
                self.population[j].individual = self.mutate(nind) # 将此个体进行随机变异
                self.population[j].calculateFitness() # 将变异后的个体的适应度分量重新计算
                self.fitness[j] = 0.3 * (self.population[j].fitness[3] / 40000) + 0.4 * (self.population[j].fitness[4] / 3000) + 0.3 * (1 - self.population[j].fitness[5] / 899) # 将变异后的个体的适应度重新计算

    # 时刻注意的是 我们解向量中的每一个元素的位置是对应于车的编号的 交叉的部分变化的仅仅是这辆车去哪一个充电站以及到充电站的距离
    def cross(self,parent1,parent2): # 两个个体的交叉操作 产生一个新的个体
        """crossover"""
        if np.random.rand() > self.c_rate: # 根据交叉的概率来决定是否需要进行交叉操作
            return parent1
        index1 = np.random.randint(0,len(parent1)-1) # 随机找到一个交叉的开始点
        index2 = np.random.randint(index1,len(parent1)-1) # 随机找到一个交叉的结束点
        parent1[index1:index2] = parent2[index1:index2] # 执行交叉操作
        return parent1 # 返回交叉结束后的父本1


    def mutate(self,gene): # 对单独一个个体的变异操作 变异操作就是 在当前的解的情况下将某一辆电车需要去哪一个充电桩以及行驶距离进行一次修改/跳变，但是其效果可能好可能坏，目的是产生新的个体
        if np.random.rand() > self.m_rate: # 生成随机数决定是否进行变异操作
            return gene # 直接返回 不进行变异操作
        index1 = np.random.randint(0,len(gene)-1) # 随机找到一个变异的开始点
        index2 = np.random.randint(index1,len(gene)-1) # 随机找到一个变异的结束点
        for i in range(index1,index2+1): # 遍历被变异区段 每一个区段中的几个一维数组中的电车目标点状信息随机交换 -- 这样理解
            ind = random.choices(self.data[i])
            copy_ind = copy.deepcopy(ind[0])
            copy_ind.append(i) # 重新分配车辆编号
            gene[i]=copy_ind # 将重新得到的第i+index1-1号电车的解复制回去
        return gene # 重新返回整个解


    def solve(self):
        self.t = 0 # 记录遗传算法迭代次数
        self.initialize() # 初始化种群
        self.evaluation() # 计算种群中每一个个体的适应度
        self.bestfitness = np.min(self.fitness) # 找到最优适应度个体的适应度 ？？？？？为什么最小就是最好的
        self.bestIndex = np.argmin(self.fitness) # 找到具有最优适应度的个体的下标
        self.bestgene = copy.deepcopy(self.population[self.bestIndex]) # 拷贝最优适应度个体
        self.besttruefitness = self.population[self.bestIndex].fitness # 保存当前最优个体的适应度
        self.individual = self.population[self.bestIndex].individual # 取得当前最优个体适应度分量信息
        self.avefitness = np.mean(self.fitness) # 计算当前种群使用度的平均值
        print(self.t) # 输出当前执行遗传算法轮次
        self.trace[self.t,0] = self.fitness[self.bestIndex] # 在算法执行跟踪数组中 存放 当前最佳适应度
        self.trace[self.t,1] = self.avefitness # 以及当前平均适应度
        print("Iteration:%d,fitness:%f,avrage fitness:%f, best individual:%s." % (self.t + 1,self.trace[self.t,0],self.trace[self.t,1],str(self.bestgene.fitness)))# 输出当前适应度的信息
        while self.t < self.maxiternum - 1: # 执行遗传算法核心部分
            self.t += 1 # 当前迭代次数加一
            self.selection() # 选择
            self.crossoverMutation() #交叉 变异
            localbest = np.min(self.fitness) # 取得当前最佳适应度
            if localbest < self.bestfitness: # 更新最佳适应度分量 最佳适应度实际值 以及 具有当前最佳适应度的个体
                self.bestfitness = localbest
                self.besttruefitness = self.population[self.bestIndex].fitness
                self.individual = self.population[self.bestIndex].individual
            self.bestgene = copy.deepcopy(self.population[self.bestIndex]) # 复制一份最佳个体
            self.avefitness = np.mean(self.fitness) # 求解平均适应度
            self.trace[self.t,0] = self.fitness[self.bestIndex] # 在算法执行跟踪数组中 存放 当前最佳适应度
            self.trace[self.t,1] = self.avefitness # 以及当前平均适应度
            print("Iteration:%d,fitness:%f,average fitness:%f, best individual:%s." % (self.t + 1, self.trace[self.t, 0], self.trace[self.t, 1], str(self.bestgene.fitness)))
        print("True Best Fitness:",self.besttruefitness)
        print("Best Fitness:",self.bestfitness)
        print("Average of queuing time (minute):",self.besttruefitness[3]/899)
        print("Average of idle time (minute):",self.besttruefitness[4]/170)
        print("Number of charged EV within an hour:",int(self.besttruefitness[5]))
        print("Total cost of all EV:",self.besttruefitness[1])
        print("Total revenue of EVCS:",self.besttruefitness[0])
        print(list(self.trace[:,0]))

def main():
    # 读取34省充电桩数据
    data = pd.read_csv("chargingstations.csv",delimiter=";",header=None).values
    cities = data[:,1:] #全国34所城市的经纬度坐标
    cities_name = data[:,0] # 全国34所城市的编号 每一个编号对应着一个经纬度坐标
    city_size = data.shape[0] # city_size=34 充电桩点的位置
    locations = np.arange(cities.shape[0]) # 问题的解集合
    '''读取电车数据'''
    ev_data_1000 = pd.read_csv("ev_data_1000.csv").values

    ev_x = ev_data_1000[:, 1:2] # 当前各个电车的X坐标
    ev_x = [j for i in ev_x for j in i] # 将X坐标从二维数组形式转化为一维列表的形式
    ev_y = ev_data_1000[:, 2:3] # 当前各个电车的Y坐标
    ev_y = [j for i in ev_y for j in i] # 将Y坐标从二维数组形式转化为一维列表的形式
    ev_ld = ev_data_1000[:, 3:4] # 当前某一辆电车所对应的剩余电量所可以支撑的剩余里程
    ev_ld = [j for i in ev_ld for j in i] # 将ev_ld从二维数组形式转化为一维列表的形式
    ev_ct = ev_data_1000[:, 4:5] # 每辆电车所需要的充电时间
    ev_ct = [j for i in ev_ct for j in i] # 将ev_ct从二维数组形式转化为一维列表的形式
    '''读取电价数据'''
    ev_cp_34 = pd.read_csv('ev_cp_34.csv').values # 读取存储充点电价的csv文件，其中第二列为全国各个省会城市的充电桩充电电价 排序方式与充电桩坐标相同
    ev_cp = ev_cp_34[:, 1:] # ev_cp 充电电价
    ev_cp = [j for i in ev_cp for j in i] # 将充电电价从二维数组转化为一维列表的形式

    '''charging stations'''
    t_c_l = []  # 每个电车的候选充电站
    for i in range(1000): # 对当前所有的1000个还在行驶的电车
        c_l = [] # 用来存储当前编号为i的电车 的可到达的所有充电桩序号 到达充电桩的时间
        for j in range(34): # 针对目前仅存的所有的34个充电桩
            d = np.sqrt((ev_x[i] - cities[j][0]) ** 2 + (ev_y[i] - cities[j][1]) ** 2)
            if d < ev_ld[i]: # ev_ld应该是当前电车剩余电量可行驶里程
                c = [] # 做一个临时变量用于临时存取
                c.append(int(j))  # 充电站序号
                c.append(d)  # 距离充电站距离，也是行驶时间
                c_l.append(c)
        t_c_l.append(c_l) # 将当前编号的电车所对应的充电桩序号 距离等信息加到列表中 其最终结果是一个三维列表

    for i in range(len(t_c_l) - 1, -1, -1): #为了遗传算法的顺利进行，将不包含数据的数组移除
        if t_c_l[i] == []: # 如果列表中的某一个行是空的 就将其对应的所有计算适应度的指标删除
            del ev_x[i]
            del ev_y[i]
            del ev_ld[i]
            del ev_ct[i]
            del t_c_l[i]

    GA = GeneticAlgorithm(t_c_l, ev_ct, ev_cp, len(cities)) # 初始化遗传算法类
    GA.solve() #迭代求解

if __name__ == '__main__':

    start = time.time() # 开始计时 计算算法执行时间
    main() # 遗传算法主函数
    end = time.time() # 停止计时 计算算法执行时间
    print("Time consuming:", end - start) # 输出整个算法执的耗时