
# 导入相关包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random
import time


# 定义电车-充电桩算法解的个体类
class EVCSIndividual: # EVCS指的是充电站

    '''
    individual of demand oriented method
    '''

    def __init__(self,data,ev_ct,ev_cp,ev_n):
        # data 各个省市电车数据
        # ev_ct 电车充电时间
        # ev_cp 电车充电价格
        # ev_n  充电站数目
        self.data = data # 所有充电电车的数据
        self.evcs = np.zeros(ev_n) # 记录每个充电站的车数
        self.individual = [] # 定义染色体 每一个解对应于一个充电桩的分配模式
        self.fitness = [] # 存储当前个体适应度
        self.ev_ct = ev_ct # 电车充电时间
        self.ev_cp = ev_cp # 电车充电价格
        self.ev_n = ev_n # 充电站总数

    def GeneratedIndiviudal(self): # 找到一个可行解（个体）    对每一个汽车找到一个随机的充电站 暂时充当局部最优解
        '''
        generated a random chromosome for genetic algorithm
        '''
        gene = list() # gene是一个记录当前所有电车的局部最优解的二维列表 其规模为<num of EV>X3
        for i in range(len(self.data)): # 对所有的充电车进行操作
            ind = random.choices(self.data[i]) # 所有的电车随机选择充电站
            copy_ind = copy.deepcopy(ind[0]) # 在每一个电车的候选充电站列表中选择第一个充电站作为当前的候选局部最优解
            copy_ind.append(i) # 为电车加上电车序号，用于后续操作，并且满足与其它算法的一致性
            gene.append(copy_ind) # 在gene基因型当中加入指定车的当前最优解信息
        random.shuffle(gene)
        self.individual = gene
        print(gene) # 输出结果是一个二维数组 每一行表示 第i号电车到当前距离最近的充电站的<编号>以及<距离> gene[i][2]所表示的电车在新的局部可行解下的编号

    # 本算法的重点环节
    def calculateFitness(self,cd_n=5): # cd_n指的是当前每一个充电站下充电桩的个数(类比一下插座和插孔的关系)
        '''
        calculate the fitness of the indiviudal，individual is a feasible solution
        '''
        ev_n = np.zeros((self.ev_n,cd_n)) # self.ev_n: 电站数目
        cd_wt = np.zeros((self.ev_n,cd_n)) # 一个充电桩的工作时间
        cd_st = np.zeros((self.ev_n,cd_n)) # 一个充电桩的空闲时间
        st = 0. # 电桩空闲总时间
        qt = 0. # 电车排队总时间
        tt = 0. # 充电总时间 到充电桩的时间+排队时间+充电时间
        cost = 0. # 充电总成本
        n = []
        x_n = 0 # 排队中的电车数目
        for i in range(len(self.individual)): # 对所有个体下的基因i 也就是每一辆电车
            '''所有电车充电的总花费(收益) += i号车的行驶距离(假设电车速度恒定)*1元/km + i号电车所找到的充电站充电单价*该辆车的充电时间 '''
            cost += self.individual[i][1] + self.ev_cp[self.individual[i][0]]*self.ev_ct[self.individual[i][2]] # 这里为什么使用序号self.individual[i][2]描述电车的序号？ 因为给定的数据是固定的我们为其编码后可以快速找到某一个电车的充电时间
            k = int(self.evcs[self.individual[i][0]]%cd_n) # 本算法在这里根据充电站编号随机选择一个充电桩号

            self.evcs[self.individual[i][0]]+=1 # 找到具体的充电站之后，该充电站下面的待充电电车就增加1
            if cd_wt[self.individual[i][0]][k]<60: # 如果当前找到的充电站的第k个充电桩的等待时间小于60个单位
                ev_n[self.individual[i][0]][k]+=1 # 那么就令self.individual[i][0]号充电站中的第k个充电桩等待充电的电车数量加一

            # 如果车辆到达充电站所需要的时间小于所分配的充电桩等待时间
            if self.individual[i][1]<cd_wt[self.individual[i][0]][k]: #k是充电桩的桩号
                tt += cd_wt[self.individual[i][0]][k]+self.ev_ct[self.individual[i][0]] # 那么总时间消耗就等于等待时间加上充电时间
                qt += cd_wt[self.individual[i][0]][k]-self.individual[i][1] # 排队时间就等于总共电桩等待的时间减去路上花的时间

                cd_wt[self.individual[i][0]][k] += self.ev_ct[self.individual[i][0]]
                x_n += 1 # 此时需要排队，在对中排序的电车数目加一
            else:  # 如果车辆到达充电站所需要的时间大于所分配充电桩的等待时间
                tt += self.individual[i][1]+self.ev_ct[self.individual[i][0]] #那么总时间消耗就等于等待时间加上充电时间
                st += self.individual[i][1]-cd_wt[self.individual[i][0]][k]  # 总空闲时间
                cd_st[self.individual[i][0]][k] += self.individual[i][1]-cd_wt[self.individual[i][0]][k] # 电桩的空闲时间就等于行驶时间减去电车在充电站self.individual[i][0]的充电桩k下所需要的等待时间
                cd_wt[self.individual[i][0]][k] = self.individual[i][1]+self.ev_ct[self.individual[i][0]] # i号电车所选充电站的第k个充电桩的等待时间就等于电车在路上的时间加上充电时间
        # 计算总收益 假设每一单单位时间收一块
        revenue = 0 # 总收益
        t_ev_n = 0 # 单位时间每个充电站充点电车数
        t_st_v = 0 # 总浪费(空闲时间)
        for i in range(self.ev_n): # ev_n 电站数目
            for j in range(cd_n): # cd_n 电桩数目
                revenue += (cd_wt[i][j]-cd_st[i][j])*self.ev_cp[i] # 根据上面的计算结果合计总收入 (包含成本)
                t_ev_n += ev_n[i][j] # 按照每一个充电桩下总共的充电电车数进行一个数目统计
                t_st_v += cd_st[i][j] # 对每一个充电桩的空闲时间进行统计

        n.append(revenue)           #总收益 [0]
        n.append(cost)              #总成本 [1]
        n.append(tt)                #总时间 [2]
        n.append(qt)                #总排队时间 [3]
        n.append(st)                #总空闲时间 [4] ≈ 后面几套代码的idle time
        n.append(t_ev_n)            #单位时间每个充电桩充电电车数 [5]
        self.fitness = np.array(n)  #用于后期计算出适应度

class GeneticAlgorithm: # 定义遗传算法类

    def __init__(self,t_c_l,ev_ct,ev_cp,n,c_rate=0.7,m_rate=0.3,pop_size=200,maxnum=3000): # 初始化遗传算法执行下来所需要的所有变量
        self.pop_size = pop_size #设定初始种群规模
        self.fitness = np.zeros(self.pop_size) # 记录每一个种群中个体的适应度
        self.c_rate = c_rate # 初始化交叉率
        self.m_rate = m_rate # 初始化变异率
        self.maxiternum = maxnum # 设定最大迭代次数
        self.population = [] # 初始种群
        self.bestfitness = 0.# 当前最佳适应度
        self.besttruefitness = [] # 记录每一轮最佳个体的适应度值
        self.bestIndex = 0 # 记录最优局部解个体在种群中的下标
        self.bestgene = [] # 记录中群众最优基因
        self.trace = np.zeros((self.maxiternum,2)) # 记录种群中最优解的迭代过程 里面的元素分别是 当前最优个体的适应度 当前的平均适应度
        self.avefitness = 0. # 用于后期计算平均适应度

        self.data = t_c_l # fit算法用的数据
        self.ev_ct = ev_ct # 每辆电车的充电时间
        self.ev_cp = ev_cp # 每一个充电站的电价
        self.cs_n = n # 充电站数量
        self.bestindividual = [] # 记录解(一个个体)
        self.dataset = [] # 记录每一次EDA搜索的最优个体 这是一个三维列表
        self.matrix = np.zeros((len(t_c_l),self.cs_n)) # 记录每一次电车调度后的点装分配信息

    def initialize(self): # 遗传算法中的初始化种群的操作
        '''
        initialize the population of GA
        '''
        for i in range(0,int(self.pop_size)): # 执行int(self.pop_size)次迭代，初始化int(self.pop_size)个可行解
            ind = EVCSIndividual(self.data,self.ev_ct,self.ev_cp,self.cs_n) # 初始化一个种群的个体
            ind.GeneratedIndiviudal() # 产生一个实际的个体
            self.population.append(ind) # 将这个个体加入种群当中去

    def evaluation(self): # 计算当前种群中每一个个体的适应度
        '''
        evaluation the fitness of tghe population
        '''
        for i in range(0,int(self.pop_size)): # 对所有的个体计算适应度
            self.population[i].calculateFitness() # 调用EVCSIndividual类方法计算用于计算适应度分量的函数
            self.fitness[i] = 0.3*(self.population[i].fitness[3]/40000)+0.4*(self.population[i].fitness[4]/3000)+0.3*(1-self.population[i].fitness[5]/899)  # 按照论文中的表示方法计算出数值型的个体适应度

    # 时刻注意的是 我们解向量中的每一个元素的位置是对应于车的编号的 交叉的部分变化的仅仅是这辆车去哪一个充电站以及到充电站的距离
    def cross(self,parent1,parent2): # 两个个体的交叉操作 产生一个新的个体
        """交叉p1,p2的部分基因片段"""
        if np.random.rand() > self.c_rate:  # 根据交叉的概率来决定是否需要进行交叉操作
            return parent1
        index1 = np.random.randint(0,len(parent1)-1) # 随机找到一个交叉的开始点
        index2 = np.random.randint(index1,len(parent1)-1) # 随机找到一个交叉的结束点
        '''交叉基因片段'''
        tempGene = parent2[index1:index2] # 截取交叉片段
        tempGene_t = pd.DataFrame(tempGene,columns=["0","1","2"]) # 设置每一个列的列头后续用来判断是否成功交叉
        tempGene_c = list(tempGene_t["2"]) # 得到电车的编号列表
        newGene = [] # 用于存储新的基因
        p1len = 0
        for g in parent1: # 对于交叉前副本一的分配方案中的每一辆电车的分配情况
            if p1len == index1: # 从左至右开始遍历 如果到达开始交叉的位置
                newGene.extend(tempGene) # 将交叉片段放到新的基因容器里
            if g[2] not in tempGene_c: # 如果当前的g不是[index1:index2]之间的解决方案，那么就将其放入新的个体数组中
                newGene.append(g)
            p1len += 1 # 加上一是为了防止重复
        if len(newGene)!=len(parent1): # 如果发现因为上述特殊的交叉操作导致新的染色体长度不足，则直接生成一个新的解
            ind = EVCSIndividual(self.data,self.ev_ct,self.ev_cp,self.cs_n)
            ind.GeneratedIndiviudal()
            return ind.individual
        return newGene # 返回最终的新个体

    def reverse_gen(self,gen,i,j): # 将基因逆转
        '''函数 ：翻转基因i到j中的片段'''
        if i>=j: # 如果输入错误 返回
            return gen
        if j >len(gen)-1: # 如果输入错误 返回
            return gen
        parent1 = copy.deepcopy(gen) # 复制一个亲本
        tempGene = parent1[i:j] # 截取基因中的某一段
        parent1[i:j] = tempGene[::-1] # 将这一段基因翻转
        return parent1 # 返回原来的染色体

    def mutate(self,gene): # 对单个基因的变异操作
        """mutation"""
        if np.random.rand() > self.m_rate: # 依据变异概率大小判断是否执行变异操作
            return gene # 不进行变异，直接返回染色体
        index1 = np.random.randint(0,len(gene)-1) # 找到变异操作起始点
        index2 = np.random.randint(index1,len(gene)-1) # 找到变异操作结束点
        newGene = self.reverse_gen(gene,index1,index2) # 将变异区间的染色体翻转
        if len(newGene) != len(gene): # 假设最终的染色体因为变异操作失败，那么重新生成一个个体
            ind = EVCSIndividual(self.data,self.ev_ct,self.ev_cp,self.cs_n)
            ind.GeneratedIndiviudal()
            return ind.individual
        return newGene # 返回最终的变异结果

    def selection(self):  # 选择操作在这里是交叉与变异操作的结合
        self.dataset.append(self.population[self.bestIndex].individual) # self.dataset用于存储'''每一次'''选择过程中的最佳个体
        for i in range(self.pop_size):
            if i != self.bestIndex and self.fitness[i]>self.avefitness: # 选择操作中的任何子操作不涉及当前适应度为最佳的个体
                pi = self.cross(self.population[self.bestIndex].individual,self.population[i].individual) # 让所有的个体与最佳个体交叉产生新的个体
                self.population[i].individual = self.mutate(pi) # 将新的个体进行变异操作
                self.population[i].individual = sorted(self.population[i].individual,key=(lambda x:[x[0],x[1]])) # 根据所选充电桩号以及到哪里的距离从小到大排序 就是在选择同样充电桩条件下优先调度距离近得 这个就是贪心的思想
                self.population[i].calculateFitness() # 计算该个体的适应度分量
                self.fitness[i] = 0.3*(self.population[i].fitness[3]/40000)+0.4*(self.population[i].fitness[4]/3000)+0.3*(1-self.population[i].fitness[5]/899) # 计算该个体的适应度总值

    def makenewInd(self): # 生成一个新的个体
        newgene = []
        for i in range(len(self.matrix)):
            k = np.argmax(self.matrix[i])
            for j in range(len(self.data[i])):
                if self.data[i][j][0] == k:
                    cs = copy.deepcopy(self.data[i][j])
                    cs.append(i)
                    break
            newgene.append(cs)
        ind = EVCSIndividual(self.data,self.ev_ct,self.ev_cp,self.cs_n)
        ind.GeneratedIndiviudal()
        newgene = self.cross(newgene,ind.individual)
        return newgene

    # 1000X34大小的数组
    def computematrix(self,cd_t=5): # 用于记录每一个充电站下所有充电桩的电车充电情况计数
        for i in range(len(self.dataset)): # 对所有的电车
            for j in range(len(self.dataset[i])):  # 对所有的电车对应的电桩遍历并依次计算
                cs_n = int(self.dataset[i][j][0])  # 充电站序号
                ev_n = int(self.dataset[i][j][2])  # 电车序号
                self.matrix[ev_n][cs_n] += 1 # 这是一个用于表示所有充电桩在算法一次执行(产生一个解)后，充电桩分配的次数以及分情况

    def crossoverMutation(self): # 交叉变异的合并操作
        for j in range(self.pop_size):
            r = np.random.randint(0,self.pop_size-1)
            if j!=r:
                nind = self.cross(self.population[j].individual,self.population[r].individual)
                self.population[j].individual = self.mutate(nind)
                self.population[j].individual = sorted(self.population[j].individual,key=(lambda x:[x[0],x[1]]))
                self.population[j].calculateFitness()
                self.fitness[j] = 0.3 * (self.population[j].fitness[3] / 40000) + 0.4 * ( self.population[j].fitness[4] / 3000) + 0.3 * (1 - self.population[j].fitness[5] / 899)

    # EDA算法中的交叉操作
    def crossEDA(self,parent1,parent2):
        """cross over"""
        if np.random.rand() > self.c_rate:
            return parent1
        ind = self.makenewInd()
        tempGene = sorted(ind,key=(lambda x:[x[0],x[1]]))
        newGene = []
        p1len = 0
        index1 = np.random.randint(0,len(tempGene)-1)
        index2 = np.random.randint(index1,len(tempGene)-1)
        tempGene_t = pd.DataFrame(tempGene[index1:index2],columns=["0","1","2"])
        tempGene_c = list(tempGene_t["2"])
        q = random.random()
        if q>0.5:
            for g in parent1:
                if p1len == index1:
                    newGene.extend(tempGene[index1:index2])
                if g[2] not in tempGene_c:
                    newGene.append(g)
                p1len+=1
        if len(newGene)!=len(parent1):
            ind = EVCSIndividual(self.data,self.ev_ct,self.ev_cp,self.cs_n)
            ind.GeneratedIndiviudal()
            return ind.individual
        return newGene

    # EDA算法中的的选择操作
    def selectionEDA(self):
        self.dataset.append(self.population[self.bestIndex].individual)
        for i in range(self.pop_size):
            if i!=self.bestIndex:
                ind = self.makenewInd()
                nind = self.cross(self.population[self.bestIndex].individual,ind)
                self.population[i].individual = nind
                self.population[i].individual = sorted(self.population[i].individual,key=(lambda x:[x[0],x[1]]))
                self.population[i].calculateFitness()
                self.fitness[i] = 0.3*(self.population[i].fitness[3]/40000)+0.4*(self.population[i].fitness[4]/3000)+0.3*(1-self.population[i].fitness[5]/899)

    # EDA算法中的交叉变异混合操作
    def crossoverMutationEDA(self):
        for j in range(self.pop_size):
            if j != self.bestIndex:
                ind = self.makenewInd()
                nind = self.cross(self.population[j].individual,ind)
                self.population[j].individual = nind
                self.population[j].indiviudal = sorted(self.population[j].individual,key=(lambda x:[x[0],x[1]]))
                self.population[j].calculateFitness()
                self.fitness[j] = 0.3 * (self.population[j].fitness[3] / 40000) + 0.4 * ( self.population[j].fitness[4] / 3000) + 0.3 * (1 - self.population[j].fitness[5] / 899)

    def solve(self): # EV问题的解决方案
        xa,ya,za = [],[],[] # xa,ya,za分别对应适应度数组分量形式每一行的 3 4 5号元素
        self.t = 0 # 记录遗传算法迭代次数
        self.initialize() # 初始化种群
        self.evaluation() # 计算种群中每一个个体的适应度
        self.bestfitness = np.min(self.fitness) # 找到最优适应度个体的适应度
        self.bestIndex = np.argmin(self.fitness) # 找到具有最优适应度的个体的下标
        self.bestgene = copy.deepcopy(self.population[self.bestIndex]) # 拷贝最优适应度个体
        self.besttruefitness = self.population[self.bestIndex].fitness # 保存当前最优个体的适应度
        self.bestindividual = self.population[self.bestIndex].individual # 取得当前最优个体适应度分量信息
        self.avefitness = np.mean(self.fitness) # 计算当前种群使用度的平均值
        self.trace[self.t,0] = self.fitness[self.bestIndex] # 在算法执行跟踪数组中 存放 当前最佳适应度
        self.trace[self.t,1] = self.avefitness  # 以及当前平均适应度
        print("Iteration:%d, fitness:%f, average fitness:%f, best individual:%s." % (self.t, self.trace[self.t, 0], self.trace[self.t, 1], str(self.bestgene.fitness)))  # 输出当前适应度的信息

        print("Ordianry searching stage.........")

        while self.t < self.maxiternum - 125: # 执行遗传算法迭代部分
            self.t += 1 # 当前迭代次数加一
            self.selection() # 选择
            self.crossoverMutation() #交叉 变异
            localbest = np.min(self.fitness) # 取得当前最佳适应度
            self.bestIndex = np.argmin(self.fitness) # 更新最佳适应度分量 最佳适应度实际值 以及 具有当前最佳适应度的个体
            if localbest < self.bestfitness:
                self.bestfitness = localbest
                self.besttruefitness = self.population[self.bestIndex].fitness
                self.bestindividual = self.population[self.bestIndex].individual
            self.bestgene = copy.deepcopy(self.population[self.bestIndex]) # 复制一份最佳个体
            self.avefitness = np.mean(self.fitness) # 求解平均适应度

            self.trace[self.t,0] = self.fitness[self.bestIndex] # 在算法执行跟踪数组中 存放 当前最佳适应度
            self.trace[self.t,1] = self.avefitness # 以及当前平均适应度
            print("Iteration:%d, fitness:%f, average fitness:%f, best individual:%s." % (self.t, self.trace[self.t, 0], self.trace[self.t, 1], str(self.bestgene.fitness)))

            # 每经过25个执行步就进行一次适应度分量 3 4 5 的记录 并且重新进行初始化，再一次搜索
            if self.t % 25 == 0:
                xa.append(self.bestgene.fitness[3]/899)
                ya.append(self.bestgene.fitness[4]/170)
                za.append(self.bestgene.fitness[5])
                self.population.clear()
                self.initialize()
                self.evaluation()

        print("EDA searching stage.........") # 执行EDA算法迭代部分
        self.computematrix()   # 记录每一个充电桩的充电次数
        self.dataset.clear()   # 将此前记录的电车分配情况清空 配合计算matrix
        while self.t < self.maxiternum-1:
            self.t += 1 # 记录EDA搜索次数
            self.selectionEDA() # 选择
            self.computematrix()# 记录每一个充电桩的充电次数
            self.dataset.clear()# 将此前记录的电车分配情况清空 配合计算matrix
            self.crossoverMutation() # 交叉变异
            localbest = np.min(self.fitness) # 记录当前局部最优解的适应度
            if localbest < self.bestfitness: # 依据所找到的局部最优解的信息 更新局部最优解对应的
                self.bestfitness = localbest # 适应度实际值
                self.besttruefitness = self.population[self.bestIndex].fitness # 适应度分量
                self.bestindividual = self.population[self.bestIndex].individual # 最佳个体的分配电桩的情况
            self.bestIndex = np.argmin(self.fitness) # 记录最佳个体下标
            self.bestgene = copy.deepcopy(self.population[self.bestIndex]) # 复制种群中最佳个体
            self.avefitness = np.mean(self.fitness) # 计算当前平均适应度
            self.trace[self.t, 0] = self.fitness[self.bestIndex] # 将最佳适应度存入trace
            self.trace[self.t, 1] = self.avefitness # 将平均适应度存入trace
            print("Iteration:%d, fitness:%f, average fitness:%f, best individual:%s." % (self.t, self.trace[self.t, 0], self.trace[self.t, 1], str(self.bestgene.fitness)))

            xa.append(self.bestgene.fitness[3]/899) # xa记录平均电车排队时间
            ya.append(self.bestgene.fitness[4]/170) # ya记录平均电桩空闲时间
            za.append(self.bestgene.fitness[5])     # za记录单位时间/一次调度后在充电电车数

        print("True best fitness:",self.besttruefitness)
        print("Best fitness:",self.bestfitness)
        print("Average of queuing time (minute):", self.besttruefitness[3] / 899)
        print("Average of idle time (minute):", self.besttruefitness[4] / 170)
        print("Number of charged EV within an hour:", int(self.besttruefitness[5]))
        print("Total cost of all EV:", self.besttruefitness[1])
        print("Total revenue of EVCS:", self.besttruefitness[0])

        print("Best Indiviudal:", self.bestindividual)   ### 输出最佳个体

        # 依据所记录的数据绘制图像
        '''可视化调度过程中的信息'''
        x = np.array(xa[0:3])
        y = np.array(ya[0:3])
        z = np.array(za[0:3])
        ax = plt.subplot(projection='3d')
        ax.set_title('Trace of fitness')
        area1 = np.pi * 4 ** 2
        area2 = np.pi * 4 ** 1
        ax.scatter(x, y, z, marker="v", c='r', s=area1, label="Local optimum")
        x1 = np.array(xa[3:])
        y1 = np.array(ya[3:])
        z1 = np.array(za[3:])
        ax.scatter(x1, y1, z1, marker=".", c='b', s=area2,label="Trace of EDA searching stage")
        ax.set_xlabel('Total queuing time')
        ax.set_ylabel('Idle leisure time')
        ax.set_zlabel('Charged EV within an hour')
        plt.show()
        '''可视化随着算法进行，适应度的变化情况'''
        t = range(0, self.t + 1)
        plt.subplot()
        plt.title("Fitness of iteration of this algorithms")
        plt.plot(range(0,3000),list(self.trace[:,0]),linewidth = '1',color= '#800080',label="GA-based greedy Scheduling with EDA")
        plt.xlabel('Number of iteration')
        plt.ylabel('Value of fitness')
        plt.legend()
        plt.show()

def main():

    start = time.time()

    # 读取34省充电桩数据
    data = pd.read_csv("chargingstations.csv", delimiter=";", header=None).values
    cities = data[:, 1:] # 全国34所城市的经纬度坐标
    cities_name = data[:, 0] # 全国34所城市的编号 每一个编号对应着一个经纬度坐标
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
    t_c_l = [] # 每个电车的候选充电站
    for i in range(1000): # 对当前所有的1000个还在行驶的电车
        c_l = [] # 用来存储当前编号为i的电车 的可到达的所有充电桩序号 到达充电桩的时间
        for j in range(34): # 针对目前仅存的所有的34个充电桩
            d = np.sqrt((ev_x[i] - cities[j][0]) ** 2 + (ev_y[i] - cities[j][1]) ** 2)
            if d < ev_ld[i]:  # ev_ld应该是当前电车剩余电量可行驶里程
                c = []  # 做一个临时变量用于临时存取
                c.append(int(j)) # 充电站序号
                c.append(d) # 距离充电站距离，也是行驶时间
                c_l.append(c)
        t_c_l.append(c_l) # 将当前编号的电车所对应的充电桩序号 距离等信息加到列表中 其最终结果是一个三维列表

    for i in range(len(t_c_l) - 1, -1, -1):  # 为了遗传算法的顺利进行，将不包含数据的数组移除
        if t_c_l[i] == []: # 如果列表中的某一个行是空的 就将其对应的所有计算适应度的指标删除
            del ev_x[i]
            del ev_y[i]
            del ev_ld[i]
            del ev_ct[i]
            del t_c_l[i]

    print(time.time() - start)

    GA = GeneticAlgorithm(t_c_l,ev_ct,ev_cp,len(cities)) # 初始化遗传算法类
    GA.solve() #迭代求解

if __name__ == '__main__':
    start = time.time() # 开始计时 计算算法执行时间
    main() # 遗传算法主函数
    end = time.time() # 停止计时 计算算法执行时间
    print("Time consuming:", end - start) # 输出整个算法执的耗时