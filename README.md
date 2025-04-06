# Modular Connector Designer for Pneumatic Channels
##
如何生成曲线：两个函数
第一个函数是采样生成圆弧的方式，并且生成圆弧的面具有随机性，因为A->C就只有两个点，然后为了保证平滑，随后生成的曲线输入是：起点，终点和前面生成曲线的梯度向量，拟合的曲线采用B-spline curve，这个比较方便效果好

然后是如何避障，分成两个种方法，之所以用了两种是因为我不知道他们是不是等价的，对于phase1和2
phase1的方法就是改变随机数种子，多尝试几次找到起始面，问题就解决了，所以phase1的代码里有np.random.seed(None)这个语句，启用的话就是应用这种方法，这个方法感觉还挺好用的
[A,C,E],[B,D]这种情况

然后就是phase2,起初这个方法是想着和直线一样用插入点的方法，但是发现效果很不好，出现了很多问题，所以采用了经过改进的RRT算法，RRT算法就是给定一个障碍物，输入起始点和终点寻求最优路径，但是输出是折线段，所以我把输出的点列
继续用B-spline curve进行拟合，对于多个点，需要生成两段不同的B-spline curve曲线，，因为RRT算法只能输入起点和终点，同时为了保证平滑，在所有的曲线链接在一起后再次做了一次平滑处理，输入的点可以确保在曲线上。
        ['A', 'C', 'E', 'G'],   
        ['D', 'B','H']  

有些代码写的比较乱，没整合，比如检查是否碰撞的函数应该可以更简单一点
应该后续要加上曲率的检测器，弯曲过大生成管子应该是不合适的
然后没有写Union-Find



This is a program for automatically designing pneumatic channel connectors for modular slender-body soft robots.

(More details coming soon.)
