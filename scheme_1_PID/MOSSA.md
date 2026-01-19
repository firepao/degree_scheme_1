# Multi-objective sparrow search algorithm: A novel algorithm for solving complex multi-objective optimisation problems

![](images/8ffa6e1cee0366ea1c6f3c157435255aba0010b22605aa8da5cc97803ab1196e.jpg)


Bin Li a, Honglei Wang a,b,*

<sup>a</sup> Electrical Engineering College, Guizhou University, Guiyang 550025, China

<sup>b</sup> Key Laboratory of “Internet+” Collaborative Intelligent Manufacturing in Guizhou Province, Guiyang, Guizhou 550025, China

# ARTICLEINFO

# Keywords:

Multi-objective optimisation problems

Multi-objective sparrow search algorithm

Fast non-dominated sorting

$2k$  crowding-distance entropy

Adaptive parameters

# ABSTRACT

This study proposes a multi-objective sparrow search algorithm (MOSSA) based on a  $2k$  crowding-distance entropy and the optimal strategy of the positions to solve complex multi-objective optimisation problems (MOPs) using the excellent performance of the sparrow search algorithm (SSA). A fast non-dominated sorting approach is incorporated to permit SSA to solve MOPs. To maintain the evenness and spread of the Pareto solution set obtained in each run, a  $2k$  crowding-distance entropy is proposed to measure the diversity of the solution set. Modified formulas for updating positions and additional adaptive parameters can improve the global search ability of MOSSA. The position archive of the population is introduced to realise the optimal strategy of the positions. This strategy ensures that the positions of the population in each generation are optimal, which significantly improves the performance of MOSSA. MOSSA is compared with three well-known algorithms using a set of complex unconstrained test problems and a complex constrained engineering optimisation problem. This study explores the effect of the  $2k$  crowding-distance entropy and crowding distance on maintaining diversity. The impact of the  $k$  in  $2k$  crowding-distance entropy on the performance of MOSSA is analysed. The experimental results demonstrate that MOSSA exhibits competitive performance in solving complex MOPs.

# 1. Introduction

Multi-objective optimisation problems (MOPs) have two or more objectives, which possibly influence and conflict with one another (Jiang, Ong, Zhang, & Feng, 2014). MOPs are widely used in industrial and engineering designs, such as the dispatch of a microgrid (Alomoush, 2019), the capacity optimisation of an energy storage systems (Xu, Liu, Lin, Dai, & Li, 2018), path planning (Yao, Li, Pan, & Wang, 2022), wind-turbine control (Yin & Gao, 2021), industrial copper burdening problems (Ma, Li, et al., 2021), and robot control (Mahmoodabadi, Taherkhorsandi, & Bagheri, 2014). However, owing to the contradictions among different objectives, the MOP has no single optimal solution, but a set of trade-off solutions called the Pareto optimal set (Kahloul, Zouache, Brahmi, & Got, 2022). In the Pareto optimal set, there is no solution in which one or more objectives are improved and the remaining objectives are not degraded (Patel & Savsani, 2016). For decision makers to have better options, the convergence of the Pareto optimal set must be high, and its diversity should be excellent (Khalilpourazari, Naderi, & Khalilpourazary, 2020). Therefore, a Pareto optimal set with good convergence and diversity is significant for the development of

industrial and engineering designs.

Although the Pareto optimal set can be obtained using traditional gradient-based optimisation algorithms, their application conditions are strict, and these algorithms require objective functions and constraints to be differentiable, and aggregate multiple objectives into one objective (Pradhan & Panda, 2012). Most of these methods can only solve MOPs with a continuous and convex Pareto front, and when the Pareto front is non-convex or discontinuous, they fail to address the MOPs. However, the non-gradient mechanism and stochastic optimisation technology of multi-objective evolutionary algorithms (MOEAs) minimise the limitations of traditional methods and solve the non-linear MOPs appropriately (Mirjalili, Mirjalili, Saremi, Faris, & Aljarah, 2018). Particularly, MOEAs can easily be applied to industrial and engineering designs. Therefore, MOEAs have been recognised as important methods for solving MOPs (Yang, Li, Liu, & Zheng, 2013). Most MOEAs are extended from scalar-objective evolutionary algorithms (SOEAs) based on Pareto dominance. The main process is as follows: first, Pareto solutions of the evolutionary population are constructed; subsequently, the Pareto front continuously converge to the true Pareto front through the population iteration mechanism; finally, the Pareto optimal set is obtained

(Ahmadi, 2016). Thus, MOEAs have two main tasks in the process of solving MOPs: to minimise the distance between the obtained and true Pareto fronts and to maximise the obtained Pareto front evenly distributed in the space (Ren, Jiang, Yang, & Qiu, 2022). Currently, two popular MOEAs exist, based on which several MOEAs have been proposed.

The first popular MOEA is the non-dominated sorting genetic algorithm II (NSGA-II) (Deb, Pratap, Agarwal, & Meyarivan, 2002), which is an extension of the genetic algorithm (GA). In NSGA-II, the fast non-dominated sorting method and crowding distance are used to obtain the Pareto solutions and maintain the diversity of the solutions, respectively. An elite strategy was proposed to ensure that the next population is not inferior to the previous population. The method of extending GA to NSGA-II has received considerable attention. Ren et al. (2022) extended the teaching-learning-based optimisation (TLBO) algorithm to a multi-objective elitist feedback teaching-learning-based optimisation (MEFTO) using fast non-dominated sorting, crowding distance, and elite strategy. The multi-objective sine-cosine algorithm (MO-SCA) (Tawhid & Savsani, 2019) and the non-dominated sorting moth flame optimisation (NS-MFO) algorithm (Savsani & Tawhid, 2017) have been proposed based on the expansion method of NSGA-II. The multi-objective particle swarm optimisation (MOPSO) algorithm is the second most popular MOEA. It is based on the particle swarm optimisation (PSO) algorithm and is expanded by incorporating an external archive and the adaptive grid (Coello, Pulido, & Lechuga, 2004). The Pareto solutions are stored in the archive and an adaptive grid is applied to maintain the diversity of solutions in the archive. The proposal of the MOPSO algorithm provides another method for the swarm intelligence algorithms, similar to PSO, extending to MOEAs, such as the multi-objective artificial bee colony (MOABC) algorithm (Akbari, Hedayatzadeh, Ziarati, & Hassanizadeh, 2012), multi-objective water cycle algorithm (MOWCA) (Sadollah, Eskandar, & Kim, 2015), multi-objective grey wolf optimiser (MOGWO) (Mirjalili, Saremi, Mirjalili, & Coelho, 2016), and multi-objective seagull optimisation algorithm (MOSOA) (Dhiman, Singh, Soni, et al., 2021).

The convergence and diversity are used to evaluate the quality of a Pareto solution set, and the diversity comprises evenness and spread (Tian, Cheng, Zhang, Li, & Jin, 2019a). Research on MOEAs, focusing on the convergence of the Pareto solution set, has progressed. Hu and Yen (2015) proposed a parallel cell coordinate system (PCCS) to improve the MOPSO, and the experimental results demonstrated that the PCCS can make the Pareto solution set have better convergence compared with the crowding distance and adaptive grid. To address large-scale many-objective optimisation problems, X. Zhang, Tian, Cheng, and Jin (2018) proposed a specially tailored evolutionary algorithm based on a decision variable clustering method that enhances the convergence of the Pareto solution set. However, MOEAs committed to the diversity of the Pareto solution set still require significant progress. An adaptive localised decision variable analysis approach was proposed to adaptively balance the convergence and diversity of Pareto solutions in the objective space (Ma, Huang, Yang, Wang, & Wang, 2021). In multi-modal MOPs, a novel subset selection framework was used to select a well-distributed Pareto solution set (Peng & Ishibuchi, 2021). M. Li, Yang, and Liu (2014) proposed a shift-based density estimation (SDE) strategy to evaluate the density and the application of SDE in three popular MOEAs demonstrated its superiority in solving multi-objective problems. In addition, a radial space division-based evolutionary algorithm was proposed, in which the solutions in a high-dimensional objective space are projected into a grid-divided two-dimensional radial space for diversity maintenance (He, Tian, Jin, Zhang, & Pan, 2017). Although the literature shows that the Pareto fronts obtained by these MOEAs can effectively approximate the true Pareto fronts of MOPs, "No Free Lunch" indicates that no MOEA can completely solve all types of MOPs (Joyce & Herrmann, 2018). Methods that can better maintain the evenness and spread of the Pareto solution set must be studied.

Generally, the performance of an MOEA depends largely on its SOEA

performance. The better the performance of the SOEA, the greater is the potential of the corresponding MOEA. The sparrow search algorithm (SSA) was inspired by the foraging and anti-predation behaviour of sparrows (Xue & Shen, 2020). Xue and Shen (2020) verified that SSA has a faster convergence speed and better performance compared to PSO and grey wolf optimiser (GWO) through 19 test problems and two classic engineering problems. Zhang and Ding (2021) used SSA to optimise the stochastic configuration network, and Liu and Rodriguez (2021) optimised the operation of the renewable energy system of residential houses using SSA. As discussed in (Kathiroli & Selvadurai, 2021; Li, Wang, Wang, Negnevitsky, & Li, 2022; Zhu & Yousefi, 2021), the improved SSA is used to solve engineering problems and its performance is better than that of traditional evolutionary algorithms. These studies demonstrate that the SSA and improved SSA are superior to traditional evolutionary algorithms in terms of the benchmark test and engineering problems. Therefore, the MOEA extended from the SSA has the potential for solving MOPs.

From the literature above, most of the MOEAs of swarm intelligence algorithms that are similar to PSO adopt the expansion method of MOPSO, whereas the expansion approach of NSGA-II is less applied. This study proposes a multi-objective sparrow search algorithm (MOSSA) to solve complex MOPs. Additionally, MOSSA verifies that the expansion method of NSGA-II is applicable to swarm intelligence optimisation algorithms similar to PSO and compensates for SSA not having its own MOEA. The main contributions of this study are as follows:

- A  $2k$  crowding-distance entropy is proposed to maintain the diversity, which can compensate for the limitation of crowding distance to measure evenness. The impact of the  $k$  in  $2k$  crowding-distance entropy on the convergence and diversity of the algorithm is discussed.

- Because SSA easily falls into the local optimum and the convergence speed is considerably fast, this study modifies the formulas for updating positions and adds adaptive parameters to improve the global search ability of MOSSA.

- The optimal strategy of the positions implemented using the position archive of the population can ensure the optimality of the population positions after each iteration, which significantly improves the performance of MOSSA.

- This study demonstrates that MOSSA outperforms the traditional MOEAs using a set of complex unconstrained test problems and a complex constrained engineering optimisation problem.

The remainder of this paper is organised as follows. In Section 2, SSA is introduced. Section 3 describes MOSSA. Section 4 discusses the experimental results for a set of complex unconstrained test problems. Section 5 presents the application of MOSSA to an engineering problem and its performance in solving the problem. Finally, Section 6 concludes the paper.

# 2. Sparrow search algorithm

The outstanding performance of SSA stems from the updating mechanism of the positions, as shown in Fig. 1. The population has three roles in the foraging process: explorers, followers, and defenders. Explorers search for food-rich and safe areas, and provide foraging areas and directions for followers. The remaining sparrows are followers, and they adjust their foraging behaviour based on the positions of the explorers. If any member of the population perceives danger of predation, this sparrow becomes a defender and responds to the danger.

In the SSA, the population  $Pop(t)$  can be expressed as follows:

![](images/da946318157ac04820041a9955f4eac065fe812bf34f6d71877a83c334a121a1.jpg)



Fig. 1. Updating mechanism of positions.


$$
P o p (t) = \left[ \begin{array}{c} X _ {1} ^ {t} \\ X _ {2} ^ {t} \\ \vdots \\ X _ {m} ^ {t} \end{array} \right] = \left[ \begin{array}{c c c c} x _ {1 1} ^ {t} & x _ {1 2} ^ {t} & \dots & x _ {1 n} ^ {t} \\ x _ {2 1} ^ {t} & x _ {2 2} ^ {t} & \dots & x _ {2 n} ^ {t} \\ \vdots & \vdots & \ddots & \vdots \\ x _ {m 1} ^ {t} & x _ {m 2} ^ {t} & \dots & x _ {m n} ^ {t} \end{array} \right] \tag {1}
$$

where,  $t$  represents the current iteration;  $m$  is the number of sparrows;  $n$  is the dimension of the decision variable;  $X_{m}^{t}$  represents the position of the  $m$ -th sparrow at  $t$  iteration, whose fitness is  $f(X_{m}^{t})$ .

The explorer guides the population to move to a better foraging area. Explorers exhibit two behavioural patterns. When the alarm value is less than the safety threshold, the current foraging environment is safe, and explorers continue to search in the vicinity of the current area. When the alarm value is greater than the safety threshold, explorers perceive the current area to be dangerous and lead the population to other areas for food. The position of the explorer is updated as follows:

$$
X _ {i} ^ {t + 1} = \left\{ \begin{array}{c c} X _ {i} ^ {t} \cdot \exp \left(- \frac {t}{\alpha \cdot T}\right) & R _ {2} <   S T \\ X _ {i} ^ {t} + Q \cdot L & R _ {2} \geq S T \end{array} \right. \tag {2}
$$

where,  $T$  is the maximum iteration;  $\alpha$  is a random number from 0 to 1;  $R_{2}$  and  $ST$  are the alarm value and safety threshold, respectively.  $Q$  represents a random number obeying a normal distribution.  $L$  represents a  $1 \times n$  matrix in which each element is one. The ratio of explorers  $p_{E}$  is set to 0.2. All sparrows are sorted in ascending order based on their fitness values. The top  $p_{E}$  sparrows are selected as explorers, and the remaining individuals are followers.

Followers exhibit two behavioural patterns when following explorers. Followers close to explorers constantly monitor the explorers and compete for food to increase their predation rates. Followers far from the explorers are more likely to fly to other places for food. The positions of the followers can be described as follows:

$$
X _ {i} ^ {t + 1} = \left\{ \begin{array}{c c} X _ {\text {B e s t}} ^ {t} + \left| X _ {i} ^ {t} - X _ {\text {B e s t}} ^ {t} \right| \cdot A ^ {+} \cdot L & i \leq \frac {m}{2} \\ Q \cdot \exp \left(\frac {X _ {\text {W o r s t}} ^ {t} - X _ {i} ^ {t}}{l ^ {2}}\right) & i > \frac {m}{2} \end{array} \right. \tag {3}
$$

where,  $X_{Best}^{t}$  and  $X_{Worst}^{t}$  are the current global best and worst positions, respectively;  $A$  is a  $1 \times n$  matrix in which each element is randomly assigned 1 or -1, and  $A^{+} = A^{T}(AA^{T})^{-1}$ .

All sparrows can detect and warn of predation; thus, explorers and followers have a certain probability of becoming defenders. When an individual in the middle of the group discovers danger, it randomly walks around to draw closer to others, whereas a sparrow at the edge of the group will move to the safe area to obtain a better position. The positions of the defenders are updated as follows:

$$
X _ {i} ^ {t + 1} = \left\{ \begin{array}{c c} X _ {i} ^ {t} + \kappa \cdot \frac {\left| X _ {i} ^ {t} - X _ {\text {W o r s t}} ^ {t} \right|}{\left(f \left(X _ {i} ^ {t}\right) - f \left(X _ {\text {W o r s t}} ^ {t}\right)\right) + \varepsilon} & f \left(X _ {i} ^ {t}\right) = f \left(X _ {\text {B e s t}} ^ {t}\right) \\ X _ {\text {B e s t}} ^ {t} + \beta \cdot \left| X _ {i} ^ {t} - X _ {\text {B e s t}} ^ {t} \right| & f \left(X _ {i} ^ {t}\right) > f \left(X _ {\text {B e s t}} ^ {t}\right) \end{array} \right. \tag {4}
$$

where,  $\kappa$  represents a random number from  $-1$  to  $1$ ;  $\varepsilon$  is an arbitrarily small constant number and is used to avoid zero-division error;  $\beta$  is the step-size control parameter, which obeys a normal distribution of random numbers with a mean value of zero and a variance of one. The ratio of defenders  $p_D$  is set to 0.2, and defenders are selected at random.

The fitness values of  $X_{i}^{t + 1}$  and  $X_{i}^{t}$  are compared to ensure that the updated population is not inferior to the original population. If  $f(X_{i}^{t + 1})$  is less than  $f(X_{i}^{t})$ , the position of the  $i$ -th sparrow is updated to  $X_{i}^{t + 1}$ . Otherwise, the position of the  $i$ -th sparrow remains unchanged.

Based on the above model, the main steps of SSA are summarised as follows.

# Algorithm SSA

1 /\*Initialisation\*/

2 Set the population of sparrows as  $m$ ;

3 Set the number of explorers as  $m_e$ ;

4 Set the number of defenders as  $m_{d}$

5 Set the safety threshold as  $ST$ ;

6 Set the maximum iterations  $T$

7 Initialise the position of  $m$  sparrows;

8 /*Main loop*/

9 For  $t = 1$  to  $T$ :

10 Calculate the fitness value of s sparrows;

11 Rank the fitness and identify the best and worst individuals in the  $t$ -th iteration;

12 For  $i = 1$  to  $m_{e}$

13 Update the positions of explorers using Eq. (2);

14 For  $i = (m_e + 1)$  to  $m$

15 Update the positions of followers using Eq. (3);

16 For  $i = 1$  to  $m_d$ :

17 Update the positions of defenders using Eq. (4);

18 Compare the fitness of each sparrow at  $t + 1$  -th iteration and  $t$  -th iteration;

19 If the position of the  $t + 1$  -th iteration time is better, update the position.

20 Output the best solution;

Compared to the classical PSO, which uses the speed and position of each particle to update its position, the advantages of the updating mechanism of positions in SSA are as follows. First, the explorer can update the position by learning about its previous position, and some of the followers and defenders learn from the best individual to update the positions. These two strategies significantly improve the local search ability of SSA. Second, some followers and defenders update their positions by learning the worst individual, which can enhance global search capability. Therefore, SSA outperforms classical PSO.

# 3. Multi-objective sparrow search algorithm

Some components, including the fast non-dominated sorting and  $2k$  crowding-distance entropy, were integrated into SSA to obtain MOSSA. Modified formulas for updating positions, adaptive parameters, and the optimal strategy of the positions improve the MOSSA performance.

# 3.1. Main algorithm

Fig. 2 shows a flow chart of MOSSA. The main steps are as follows.

1) Initialise the related parameters, number of population  $m$ , maximum iteration  $T$ , and iteration  $t = 1$ .

2) Initialise the population  $Pop(t)$ :

For  $i = 1$  to  $m$ :

$$
X _ {i} ^ {t} = l b + r a n d \cdot (u b - l b);
$$

$$
\text {C a l c u l a t e} F \left(X _ {i} ^ {t}\right) = \left[ f _ {1} \left(X _ {i} ^ {t}\right), f _ {2} \left(X _ {i} ^ {t}\right), \dots , f _ {g} \left(X _ {i} ^ {t}\right) \right];
$$

where,  $lb$  and  $ub$  are the lower and upper bounds of the decision

![](images/11fadcb790b582829ad34383df1bdd555331b60e189ef007eafa33f03fa37887.jpg)



Fig. 2. Flow chart of MOSSA.


variable, respectively.  $F(X_{i}^{r})$  is the vector function and  $f_{r}(X_{i}^{r})(r = 1,2,\dots ,g)$  represents the  $r$ -th objective function.  $g$  is the number of objective functions.

3) The population  $Pop(t)$  is sorted using the fast non-dominated sorting approach:

First, the fast non-dominated sorting approach is used to calculate the non-dominated rank of each sparrow in  $Pop(t)$ . Then, all the sparrows are sorted in ascending order based on non-dominated rank and the sorted population  $Pop(t)$  is obtained.

4) The population  $Pop(t)$  is sorted using  $2k$  crowding-distance entropy sorting method:

The  $2k$  crowding-distance entropy proposed in this study is applied to sort the population  $Pop(t)$ . In the same non-dominated front, the  $2k$  crowding-distance entropy of each sparrow is calculated and they are sorted in descending order based on the  $2k$  crowding-distance entropy. Sorted population  $Pop(t)$  is obtained.

5) Initialise the position archive of the population:  $\text{Arc} = \text{Pop}(t)$ .

6) Main loop:

For  $t = 2$  to  $T$ :

a) Take the first and last sparrow in  $Pop(t)$  as the best and worst individuals in the  $t$ -th iteration, respectively.

b) The positions of the population are updated based on the modified formulas for updating positions and adaptive parameters, and the new population  $Pop(t)$  is obtained:

For  $i = 1$  to  $m_{\varepsilon}$

Update the positions of explorers using Eq. (14);

For  $i = m_e + 1$  to  $m$ :

Update the positions of followers using Eq. (18);

For  $i = 1$  to  $m_{d}$ :

Update the positions of defenders using Eq. (19);

For  $i = 1$  to  $m$ :

Calculate  $F(X_{i}^{t}) = [f_{1}(X_{i}^{t}), f_{2}(X_{i}^{t}), \dots, f_{g}(X_{i}^{t})]$ ;

where,  $m_e$  is the number of explorers;  $m_d$  is the number of defenders.

c) The population  $Pop(t)$  and the archive  $Arc$  are updated using the optimal strategy of the positions:

i)  $\operatorname{Arc}$  and  $Pop(t)$  are combined into the position set  $AP$ .

ii) First, the set  $AP$  is sorted using the fast non-dominated sorting approach. Then, the set  $AP$  is sorted by  $2k$  crowding-distance entropy sorting method.

iii) The top  $m$  positions in the set  $AP$  are taken as the archive  $Arc$ . And the positions of the archive  $Arc$  are updated as the positions of the  $t$ -th iteration population  $Pop(t)$ .

7) After the main loop,  $Pop(T)$  is the obtained Pareto solutions.

# 3.2. 2k crowding-distance entropy

All sparrows are divided into different non-dominated fronts using the non-dominated rank, as shown in Fig. 3. Diversity includes two aspects: evenness and spread. Evenness measures how evenly the Pareto solutions are dispersed in the objective space, and spread measures how close the Pareto solutions are to the extreme values in the true Pareto front. Therefore, the more evenly distributed and wider the population,

![](images/7a7a035007f366d89a8e557aa41a507af3e5bb9c5f00b907012e8d10909fdd6d.jpg)



Fig. 3. Different non-dominated fronts and crowding-distance calculation.


the better is the diversity of the population. It is necessary to measure the evenness and spread of sparrow in the same non-dominated front to maintain diversity. Deb et al. (2002) proposed the crowding distance to measure the diversity. For the two-objective optimisation problem shown in Fig. 3, the crowding-distance value of the  $i$ -th point  $CD(i)$  is the sum of the length and width of the red rectangle, and  $CD(i)$  is defined as follows:

$$
C D (i) = \frac {\left| f _ {1} \left(X _ {i + 1}\right) - f _ {1} \left(X _ {i - 1}\right) \right|}{\max  _ {1 \leq j \leq \hbar} f _ {1} \left(X _ {j}\right) - \min  _ {1 \leq j \leq \hbar} f _ {1} \left(X _ {j}\right)} + \frac {\left| f _ {2} \left(X _ {i + 1}\right) - f _ {2} \left(X _ {i - 1}\right) \right|}{\max  _ {1 \leq j \leq \hbar} f _ {2} \left(X _ {j}\right) - \min  _ {1 \leq j \leq \hbar} f _ {2} \left(X _ {j}\right)} \tag {5}
$$

where,  $h$  is the number of all individuals in the non-dominated front where the  $i$ -th point is located. For brevity, this study marks the  $i$ -1-th and  $i + 1$ -th points as the first pair of adjacent points of the  $i$ -th point, and the  $i$ -2-th and  $i + 2$ -th points as the second pair of adjacent points. By analogy, the  $i$ -k-th and  $i + k$ -th points are recorded as the  $k$ -th pair of adjacent points of the  $i$ -th point.

For a MOP with  $g$  objectives,  $CD(i)$  is expressed as:

$$
C D (i) = \sum_ {r = 1} ^ {g} \frac {\left| f _ {r} \left(X _ {i + 1}\right) - f _ {r} \left(X _ {i - 1}\right) \right|}{\max  _ {1 \leq j \leq h} f _ {r} \left(X _ {j}\right) - \min  _ {1 \leq j \leq h} f _ {r} \left(X _ {j}\right)} \tag {6}
$$

However, the crowding distance has limitations in two scenarios for measuring the diversity:

1) Scenario 1: In the same non-dominated front, the crowding-distance values of two points are equal, but their evenness is different.

2) Scenario 2: In the same non-dominated front, the difference between the crowding-distance values of two points is considerably small, and their distributions are different.

As shown in Fig. 4 (a), in Scenario 1, the crowding-distance value of the  $i$ -th and  $j$ -th points is equal,  $CD(i) = CD(j) = 0.625$ . Fig. 4 (a) shows that the distance from the  $i$ -th point to the two points of its first pair of adjacent points is equal, whereas the  $j$ -th point is far from the  $j$ -1-th point and close to the  $j + 1$ -th point. This shows that the  $i$ -th point is more evenly distributed compared to the  $j$ -th point. In Scenario 1, the distribution of the  $i$ -th point is better than that of the  $j$ -th point; therefore, the diversity of the former is better than that of the latter. Similarly, as shown in Fig. 5 (a), in Scenario 2, the crowding distance of the  $i$ -th point is slightly smaller than that of the  $j$ -th point,  $CD(i) = 0.625 < CD(j) = 0.65625$ , whereas the  $i$ -th point is more evenly distributed in the population than the  $j$ -th point. In this case, the crowding distance cannot be used to measure the evenness of the  $i$ -th point and the  $j$ -th point. Scenarios 1 and 2 show that the crowding distance for the measurement of diversity focuses more on spread, and its measurement of evenness is not accurate.

This study proposes a crowding-distance entropy to measure diversity to resolve the problem of inaccurate measurement of evenness by crowding distance. Fig. 4 (a) and 5 (a) show that the distribution of the  $i$ -th point is not associated with its absolute distance to the first pair of adjacent points but only the relative distance from it to the first pair of adjacent points. Therefore, the relative distances  $d_{i,i-1}^r$  and  $d_{i,i+1}^r$  between the  $i$ -th point and its first pair of adjacent points in the  $r$ -th objective function are respectively defined as follows:

$$
d _ {i, i - 1} ^ {r} = \frac {\left| f _ {r} \left(X _ {i}\right) - f _ {r} \left(X _ {i - 1}\right) \right|}{\left| f _ {r} \left(X _ {i + 1}\right) - f _ {r} \left(X _ {i - 1}\right) \right| + \varepsilon} \tag {7}
$$

$$
d _ {i, i + 1} ^ {r} = \frac {\left| f _ {r} \left(X _ {i + 1}\right) - f _ {r} \left(X _ {i}\right) \right|}{\left| f _ {r} \left(X _ {i + 1}\right) - f _ {r} \left(X _ {i - 1}\right) \right| + \varepsilon} \tag {8}
$$

Therefore, the entropy of the  $i$ -th point in the  $r$ -th objective function  $E_{i}^{r}$  is defined as follows:

$$
E _ {i} ^ {r} = - \left(d _ {i, i - 1} ^ {r} \ln \left(d _ {i, i - 1} ^ {r} + \varepsilon\right) + d _ {i, i + 1} ^ {r} \ln \left(d _ {i, i + 1} ^ {r} + \varepsilon\right)\right) \tag {9}
$$

where,  $\varepsilon$  is used to avoid errors in which the true number of natural

![](images/a451513980b3d1fd5927ade08d1d3c6c950120ac3a64185e4e6975aea18b17b8.jpg)



(a)


![](images/f7d815a6d22b911c722b34e5caf666279b5131236391f35e21f9f256e88e85c4.jpg)



(b)



Fig. 4. Calculation of (a) crowding distance and (b) crowding-distance entropy in Scenario 1.


![](images/8188c16272cf66baf1b5abd4af1be3dafb87eacc600aaf7d2fcb15fd6ebad4bf.jpg)



(a)


![](images/c80ed5e1f5e3ab918770cda5c257329b1ae8d8474e2465f8666b02dcd7931f9d.jpg)



(b)



Fig. 5. Calculation of (a) crowding distance and (b) crowding-distance entropy in Scenario 2.


logarithms is zero. The entropy  $E_{i}^{r}$ , calculated using the relative distance, can only measure the evenness. This study combines absolute distance and entropy to obtain the definition of the crowding-distance entropy of the  $i$ -th point  $CDE(i)$  to measure the diversity. This is described as follows:

$$
\begin{array}{l} C D E (i) = - \sum_ {r = 1} ^ {g} \left| f _ {r} \left(X _ {i + 1}\right) - f _ {r} \left(X _ {i - 1}\right) \right| E _ {r} ^ {i} \\ = - \sum_ {r = 1} ^ {g} \left| f _ {r} \left(X _ {i + 1}\right) - f _ {r} \left(X _ {i - 1}\right) \right| \left(d _ {i, i - 1} ^ {r} \ln \left(d _ {i, i - 1} ^ {r} + \varepsilon\right) + d _ {i, i + 1} ^ {r} \ln \left(d _ {i, i + 1} ^ {r} + \varepsilon\right)\right) \\ + \varepsilon)) \\ \end{array}
$$

(10)

According to the interpretation of entropy (Shannon, 1948), combined with the definition of crowding-distance entropy, the greater is the crowding-distance entropy of an individual, the better are the evenness and spread of the individual. Therefore, a larger crowding-distance entropy shows that the diversity of the Pareto solution is better.

Eq. (10) is applied to calculate the crowding-distance entropy of the  $i$ -th and  $j$ -th points in Scenarios 1 and 2. Fig. 4 (b) and 5 (b) show  $CDE(i)$ $\approx 6.93 > CDE(j)\approx 6.07$  and  $CDE(i)\approx 6.93 > CDE(j)\approx 5.76$  respectively. These two comparison results indicate that the diversity of the  $i$ -th point is better than that of the  $j$ -th point, which aligns with the analysis results from Figs. 4 and 5. When the difference in the crowding-distance value between the two points is zero or small, the crowding-distance entropy can reflect the difference in the evenness of different individuals to distinguish the pros and cons of different individuals.

These results demonstrate that the crowding-distance entropy proposed in this study is better than the crowding distance for the comprehensive measurement of the evenness and spread of the diversity of a population.

However, in the same non-dominated front, if the crowding-distance entropy values of numerous individuals are equal, the crowding-distance entropy cannot be used to sort these individuals, which is not conducive to the evolution of the next-generation population. The crowding-distance entropy of the  $i$ -th and  $j$ -th points are equal,  $CDE(i) = CDE(j) \approx 6.93$ , as shown in Fig. 6 (a). Simultaneously, the diversity of the  $i$ -th and  $j$ -th points cannot be compared using the crowding-distance entropy. Therefore, more pairs of adjacent points of each sparrow are added to the diversity calculation, and a  $2k$  crowding-distance entropy is proposed. The  $2k$  crowding-distance entropy of the  $i$ -th point is defined as the average value of the crowding-distance entropy of  $k$  pairs of adjacent points of the  $i$ -th point, and is expressed as follows:

$$
\begin{array}{l} C D E (i) = - \frac {1}{k} \sum_ {r = 1} ^ {g} \sum_ {s = 1} ^ {k} \left| f _ {r} \left(X _ {i + s}\right) - f _ {r} \left(X _ {i - s}\right) \right| \left(d _ {i, i - s} ^ {r} \ln \left(d _ {i, i - s} ^ {r} + \varepsilon\right) \right. \\ + d _ {i, i + s} ^ {r} \ln \left(d _ {i, i + s} ^ {r} + \varepsilon\right) \tag {11} \\ \end{array}
$$

where,  $k$  is a positive integer. If  $k = 1$ ,  $2k$  crowding-distance entropy is the crowding-distance entropy mentioned previously.

This study uses  $k = 2$ ; thus, this study considers the first and second pairs of adjacent points of each point, as shown in Fig. 6 (b).  $CDE(i) \approx 17.44 > CDE(j) \approx 16.97$  indicates that the diversity of the  $i$ -th point is better than that of the  $j$ -th point. Evidently, the  $2k$  crowding-distance

![](images/e967df73d6e3c98667812ba7505f51274a2692045f6e5052380e9d46f4e38e20.jpg)



(a)


![](images/bae59ce0c7d5a7015565cd30a8a86dd0478077debde3c5354bf7e9811bf19ae4.jpg)



(b)



Fig. 6. Calculation of (a) crowding-distance entropy and (b)  $2k$  crowding-distance entropy.


entropy compensates for the aforementioned limitation of crowding-distance entropy. The calculation steps of the  $2k$  crowding-distance entropy are summarised as follows.

$2k$  crowding-distance entropy

<table><tr><td>1</td><td>For i = 1 to h</td><td></td></tr><tr><td>2</td><td>CDE(i) = 0;</td><td>/*Initialise 2k crowding-distance entropy*/</td></tr><tr><td>3</td><td>For r = 1 to g:</td><td></td></tr><tr><td>4</td><td>F = sort(F, r);</td><td>/*Sort using each objective value*/</td></tr><tr><td>5</td><td>For i = 1 to h:</td><td></td></tr><tr><td>6</td><td>If F(i, r) == 1 or F(i, r) == m</td><td>/*F(i, r) is the r-th objective value of the i-th individual*/</td></tr><tr><td>7</td><td>Then CD(i) = ∞;</td><td>/*Make sure to always select boundary points*/</td></tr><tr><td>8</td><td>Else</td><td></td></tr><tr><td>9</td><td>For s = 1 to k:</td><td></td></tr><tr><td>10</td><td>CDE(i) = CDE(i) + |F(i + s, r) - F(i·s, r)|;</td><td></td></tr><tr><td>11</td><td>CDE(i) = CDE(i)/k;</td><td></td></tr></table>

# 3.3. Modified formulas for updating positions and adaptive parameters

When  $R_{2} < ST$  in Eq. (2), the variation  $y$  in the range of the position of an explorer is as follows:

$$
y = \exp \left(- \frac {t}{\alpha \cdot T}\right) \tag {12}
$$

Fig. 7 (a) shows the relationship between  $y$  and iteration  $t$  in SSA. As  $t$

increases, the range of y gradually converges from (0, 1) to (0, 0.3), indicating that the decision variables of the explorers gradually approach zero. This search strategy makes SSA advantageous for solving the scalar objective optimisation problems (SOPs) where the optimal solution is the origin or near the origin. However, the solutions of most SOPs are not distributed near or at the origin.

To improve the performance of MOSSA to solve MOPs, this study improves the formula for updating positions of explorers as follows:

$$
X _ {i} ^ {t + 1} = \left\{ \begin{array}{c c} X _ {i} ^ {t} \cdot \left(1 \pm \exp \left(- \frac {t}{\alpha \cdot T}\right)\right) & R _ {2} <   S T \\ X _ {i} ^ {t} + Q \cdot L & R _ {2} \geq S T \end{array} \right. \tag {13}
$$

When  $R_{2} < ST$  in Eq. (14), the variation  $y$  in the range of the position of an explorer is as follows:

$$
y = 1 \pm e x p \left(- \frac {t}{\alpha \cdot T}\right) \tag {14}
$$

In MOSSA, the relationship between  $y$  and iteration  $t$  is shown in Fig. 7 (b). As  $t$  increases, the range of  $y$  gradually converges from (0, 2) to (0.7, 1.3). At the beginning of the iteration,  $y$  is mainly distributed in the vicinity of zero and two. As the iteration increases,  $y$  is concentrated in the vicinity of one. This indicates that the decision variables of explorers have a larger range and can explore more possibilities of solutions at the beginning of the iteration; they stabilise near a certain value at the end of the iteration. This search strategy significantly enhances the ability of

![](images/3608371b9eb27f5bf536367aed3211fd30ce4b59f197349e129f601dc3a43b5f.jpg)



(a) Distribution of the random variables of traditional SSA


![](images/2082f0ac66defe2896cdadc6c885b9f02567b93d9eba5a1256d5ce1dbaf8a4ca.jpg)



(b) Distribution of the random variables of MOSSA



Fig.7. Distribution of the random variables in (a) SSA and (b) MOSSA.


MOSSA to solve problems in which the solutions are not at or near the origin.

At the beginning of foraging, the environment of the sparrows may be dangerous, and the safety threshold is low. During the foraging process, the safety of foraging environments continues to improve; thus, the safety threshold increases rapidly. The safety threshold gradually decreases and no longer increases significantly when the population discovers food. Therefore, this study adopts an adaptive parameter  $ST$  to describe this change in the safety threshold.  $ST$  is defined as follows:

$$
S T = \frac {S T ^ {\text {m a x}} + S T ^ {\text {m i n}}}{2} + \frac {S T ^ {\text {m a x}} - S T ^ {\text {m i n}}}{2} \tanh  (- 4 + 8 \frac {t}{T}) \tag {15}
$$

where,  $ST^{min}$  and  $ST^{max}$  are the minimum and maximum values of  $ST$ , respectively. Similarly, the proportion of explorers in the population should continue to increase. As the number of iterations increases, there are more suitable individuals; that is, numerous sparrows are qualified to become explorers. Enlarging the proportion of explorers can help MOSSA explore more possibilities for solutions and converge to the best Pareto front. Therefore, this study modifies the proportion of explorers to an adaptive parameter  $p_{E}$ , which is expressed as follows:

$$
p _ {E} = \frac {p _ {E} ^ {\max } + p _ {E} ^ {\min }}{2} + \frac {p _ {E} ^ {\max } - p _ {E} ^ {\min }}{2} \tanh  (- 4 + 8 \frac {t}{T}) \tag {16}
$$

where,  $p_E^{min}$  and  $p_E^{max}$  are the minimum and maximum values of  $p_E$ , respectively. Fig. 8 shows the curves of  $ST$  and  $p_E$ .

This study improves the distribution of the number of individuals in the two behavioural patterns in the formulas for updating positions of followers to ensure that it is reasonable when the number of explorers increases. The modified formula for updating positions of followers is expressed as follows:

$$
X _ {i} ^ {t + 1} = \left\{ \begin{array}{c c} X _ {\text {B e s t}} ^ {t} + \left| X _ {i} ^ {t} - X _ {\text {B e s t}} ^ {t} \right| \cdot A ^ {+} \cdot L & i \leq m _ {e} + \frac {m - m _ {e}}{2} \\ Q \cdot \exp \left(\frac {X _ {\text {W o r s t}} ^ {t} - X _ {i} ^ {t}}{i ^ {2}}\right) & i > m _ {e} + \frac {m - m _ {e}}{2} \end{array} \right. \tag {17}
$$

SSA has a significantly high convergence speed, and such a convergence speed may be harmful in MOPs because it may cause MOSSA to converge to a wrong Pareto front. Based on the effect of defender in enhancing the global search ability in SSA, this study improves the formula for updating positions of defenders and the proportion of defenders to decelerate the convergence speed of MOSSA and improve its ability to explore more solutions. The modified formula for updating positions of defenders is improved as follows:

![](images/35b486803fe944f30e3f95e2ffdb554136559513fa2ad617662cc0d3c3823df0.jpg)



Fig. 8. The curves of adaptive parameters.


$$
X _ {i} ^ {t + 1} = \left\{ \begin{array}{c c} X _ {i} ^ {t} + \frac {\kappa}{g} \sum_ {r = 1} ^ {g} \frac {\left| X _ {i} ^ {t} - X _ {\text {W o r s t}} ^ {t} \right|}{\left(f _ {r} \left(X _ {i} ^ {t}\right) - f _ {r} \left(X _ {\text {W o r s t}} ^ {t}\right)\right) + \varepsilon} & u \geq 0. 5 \\ X _ {\text {B e s t}} ^ {t} + \beta \left| X _ {i} ^ {t} - X _ {\text {B e s t}} ^ {t} \right| & u <   0. 5 \end{array} \right. \tag {18}
$$

where,  $u$  is a random number that obeys 0-1 distribution. Probability is used to describe the two behavioural options of the defenders. Compared with Eq. (4), Eq. (19) increases the ratio of the first behaviour, thereby enhancing the search capability of MOSSA. The proportion of defenders is improved to an adaptive parameter  $p_D$ , defined as follows:

$$
p _ {D} = \frac {p _ {D} ^ {\max } + p _ {D} ^ {\min }}{2} + \frac {p _ {D} ^ {\max } - p _ {D} ^ {\min }}{2} \tanh  (- 4 + 8 \frac {T - t}{T}) \tag {19}
$$

where,  $p_D^{min}$  and  $p_D^{max}$  are the minimum and maximum values of  $p_D$ , respectively. Fig. 8 shows the curve of  $p_D$ . When the iteration begins, numerous defenders are required to protect the population because of the dangerous environment, which can decelerate the convergence speed and increase the global search ability. It is expedient to rapidly reduce the number of defenders as the iteration increases because the foraging environment becomes safer, which can boost the convergence speed.

# 3.4. Optimal strategy of the positions

In SSA, the new position of each sparrow obtained using Eqs. (2)-(4) is compared with the original position, and only if the former is better than the latter, the new position is updated as the position of the next generation. This strategy ensures that the positions of the next generation are better than or not inferior to those of the last generation. To achieve similar effects in MOSSA, this study proposes the optimal strategy of the positions by introducing a position archive of the population.

After calculating the non-dominated rank of each individual in the population and the  $2k$  crowding-distance entropy of all individuals in each non-dominated front, each individual has the following two attributes:

(1) Non-dominated rank,  $\text{Rank}(i)(i = 1,2,\dots,m)$ ;

(2)  $2k$  crowding-distance entropy,  $CDE(i)(i = 1,2,\dots,m)$ .

This study defines a partial order  $\succ_{m}$ : if  $(Rank(i) < Rank(j))$  or  $(Rank(i) = Rank(j)$  and  $CDE(i) > CDE(j)$ ), then the  $i$ -th individual  $\succ_{m}$  the  $j$ -th individual. That is, between two individuals with different non-dominated ranks, the individual with a lower non-dominated rank is better than the individual with a higher non-dominated rank; between two individuals with the same non-dominated rank, the individual with a larger  $2k$  crowding-distance entropy is superior to the individual with a smaller  $2k$  crowding-distance entropy. This partial order is used to sort all the sparrows.

The position archive of the population  $Arc$  and the positions of the new population  $Pop(t)$  are merged into a position set  $AP$ , as shown in Fig. 9. Different from the elite strategy of NSGA-II, the optimal strategy of the positions is achieved by two-stage sorting. The first sorting stage is as follows: After the non-dominated rank of each position in  $AP$  is calculated, all the positions in  $AP$  are sorted in ascending order based on their non-dominated rank. And the sorted  $AP$  is obtained. The second sorting stage is as follows: In the same Pareto front, the  $2k$  crowding-distance entropy of all the positions in  $AP$  is calculated; then, they are sorted in descending order based on their  $2k$  crowding-distance entropy. After two-stage sorting, all the positions in  $AP$  are already sorted according to partial order. Therefore, the first  $m$  positions in  $AP$  are taken as the archive  $Arc$  and all the positions in  $Arc$  are updated as the positions of the next generation  $Pop(t)$ .

Since all the sparrows in  $Pop(t)$  have been sorted at the beginning of the iteration  $t + 1$ , the first and last sparrows in  $Pop(t)$  are set to the best

![](images/fb2ff468ce22226df497184d66fce3a3aa4dc925769ed63fc38ce18cfd3cabdc.jpg)



Fig. 9. Optimal strategy of the positions.


and worst individuals in the  $t$ -th iteration, respectively. The top  $p_E$  individuals in  $Pop(t)$  are selected as explorers, while the remaining individuals are regarded as followers.

# 3.5. Computation complexity analysis

The computation complexity is defined as the total number of objective function comparisons (Raquel & Naval, 2005). This section sets  $M$  and  $N$  to be the number of objective functions and the size of population, respectively. The computation complexity of each part in MOSSA is as follows:

(1) The computation complexity of the non-dominated sorting in the optimal strategy of the positions is  $O(M(2N)^2)$ .

(2) The computation complexity of the  $2k$  crowding-distance entropy sorting in the optimal strategy of the positions is  $O(M(2N)\log (2N))$ .

Therefore, the overall complexity of the MOSSA is  $O(M(2N)^2)$ . Since there are  $k$  pairs of adjacent points involved in the calculation of the  $2k$  crowding distance entropy, MOSSA requires more computation time. However, the performance of MOSSA is significantly improved without consuming a lot of extra time.

# 4. Results and discussion

In the complex unconstrained MOPs, the performance of the MOSSA proposed in this study is verified. This section outlines the test problems, performance metrics and experimental settings, provides the results, discusses the effects of  $2k$  crowding-distance entropy and crowding distance on MOSSA performance, and analyzes the impact of the  $k$  in  $2k$  crowding-distance entropy on the performance of the algorithm.

# 4.1. Test problems and performance metrics

This study selects 8 complex unconstrained multi-objective test problems with different Pareto fronts, namely, IMOP1-8 (Tian, Cheng, Zhang, Li, & Jin, 2019b). Table 1 gives the definitions of the eight MOPs, and all the test problems are to be minimised. These test problems include three bi-objective MOPs and five tri-objective MOPs with irregular Pareto fronts, which can pose a huge challenge to the performance of MOEAs.

To verify the performance of the MOSSA for solving these test functions, this study uses the inverted generational distance (IGD) (Dhiman, Singh, Slowik, et al., 2021) to measure the convergence and


Table 1 Test problems.


<table><tr><td>Problem</td><td>Definition</td></tr><tr><td rowspan="3">Common</td><td>X = [x1, ..., xn1, ..., xn1+n2] ∈ [0, 1]n1+n2, n1 = n2 = 5, a1 = 0.05, a2 = 0.05, a3 = 10</td></tr><tr><td>y1 = (1/n1 ∑i=1n1 xi) a1, y2 = (1/[n1/2] ∑i=1n1/2 xi) a2, y3 = (1/n1/2 ∑i=1n1/2+1 xi) a3,</td></tr><tr><td>g = ∑i=n1+n2 (xi - 0.5)2</td></tr><tr><td>IMOP1</td><td>f1 = g + cos8(π/2y1), f2 = g + sin8(π/2y1)</td></tr><tr><td>IMOP2</td><td>f1 = g + cos0.5(π/2y1), f2 = g + sin0.5(π/2y1)</td></tr><tr><td>IMOP3</td><td>f1 = g + 1 + 1/5 cos(10πy1) - y1, f2 = g + y1</td></tr><tr><td>IMOP4</td><td>f1 = (g+1)y1.f2 = (1 + g)(y1 + 1/10 sin(10πy1)), f3 = (g+1)(1 - y1)</td></tr><tr><td>IMOP5</td><td>h1 = 0.4cos(π/4[8y2]) + 0.1y3cos(16πy2), h2 = 0.4sin(π/4[8y2]) + 0.1y3sin(16πy2)</td></tr><tr><td rowspan="3">IMOP6</td><td>f1 = g + h1, f2 = g + h2, f3 = g + 0.5 - h1 - h2</td></tr><tr><td>r1 = max{0, min{sin2(3πy2), sin2(3πy3)} - 0.05}</td></tr><tr><td>f1 = (1 + g)y2 + [r], f2 = (1 + g)y3 + [r], f3 = (0.5 + g)(2 - y2 - y3) + [r]</td></tr><tr><td rowspan="3">IMOP7</td><td>h1 = (1 + g)cos(π/2y2)cos(π/2y3), h2 = (1 + g)cos(π/2y2)sin(π/2y3), h3 = (1 + g)sin(π/2y2)</td></tr><tr><td>r1 = min{min{[h1 - h2], |h2 - h3|}, |h3 - h1|}</td></tr><tr><td>f1 = h1 + 10max{0, r - 0.1}, f2 = h2 + 10max{0, r - 0.1}, f3 = h3 + 10max{0, r - 0.1}</td></tr><tr><td>IMOP8</td><td>f1 = y2, f2 = y3, f3 = (1 + g)[3 - ∑i=2n3 yi(1 + sin(19πyi)) / 1 + g]</td></tr></table>

diversity and selects spacing (SP) (Mirjalili et al., 2016) and maximum spread (MS) (Wang, Ren, Qiu, & Qiu, 2021) to measure the evenness and spread, respectively.

IGD calculates the distance between the obtained Pareto front  $PF$  and the true Pareto front  $PF^{*}$ , which is defined as follows:

$$
I G D \left(P F, P F ^ {*}\right) = \frac {\sqrt {\sum_ {i = 1} ^ {| P F ^ {*} |} d _ {i} ^ {2}}}{| P F ^ {*} |} \tag {20}
$$

$$
d _ {i} = \min  _ {X _ {j} \in P F} \| F \left(X _ {i} ^ {*}\right) - F \left(X _ {j}\right) \|, X _ {i} ^ {*} \in P F ^ {*} \tag {21}
$$

where,  $|PF^{*}|$  is the number of  $PF^{*}$ .  $d_{i}$  is the Euclidean distance between the objective function of the  $i$ -th solution in the true Pareto optimal set and the objective function of the closest solution in the obtained Pareto optimal solution. IGD can measure the convergence and diversity of  $PF$ . A smaller IGD indicates that the algorithm has superior overall performance.

SP measures the evenness of the distribution of the solutions in the obtained Pareto optimal set, and it is expressed as follows:

$$
S P (P F) = \sqrt {\frac {1}{m - 1} \sum_ {i = 1} ^ {m} (\bar {d} - d _ {i}) ^ {2}} \tag {22}
$$

$$
d _ {i} = \min  _ {X _ {j} \in P F, X _ {j} \neq X _ {i}} \left(\sum_ {r = 1} ^ {g} \left| f _ {r} \left(X _ {i}\right) - f _ {r} \left(X _ {j}\right) \right|\right), X _ {i} \in P F \tag {23}
$$

where,  $d_{i}$  is the Manhattan distance between the objective function of the  $i$ -th solution and the objective function of the nearest solution in the obtained Pareto optimal set.  $\overline{d}$  is the average of all  $d_{i}$ . A smaller SP shows that the solutions distribute more even in the obtained Pareto optimal set.

MS calculates the coverage of the obtained Pareto optimal set in the feasible region. If MS is equal 1, the coverage of the obtained Pareto optimal set is equal to the coverage of the true Pareto optimal set. A larger MS shows that the coverage is wider. MS is defined as follows:

$$
\begin{array}{l} M S \left(P F, P F ^ {*}\right) = \sqrt {\frac {1}{g} \sum_ {r = 1} ^ {g} \delta_ {r} ^ {2}} (24) \\ \delta_ {r} = \frac {\operatorname* {m i n} _ {X _ {i} \in P F} \left(\max  _ {X _ {j} ^ {*} \in P F} f _ {r} \left(X _ {i}\right) - \max  _ {X _ {j} ^ {*} \in P F} f _ {r} \left(X _ {j} ^ {*}\right)\right) - \operatorname* {m a x} _ {X _ {i} \in P F} \left(\min  _ {X _ {j} ^ {*} \in P F} f _ {r} \left(X _ {i}\right) - \min  _ {X _ {j} ^ {*} \in P F} f _ {r} \left(X _ {j} ^ {*}\right)\right)}{\max  _ {X _ {i} ^ {*} \in P F} f _ {r} \left(X _ {i} ^ {*}\right) - \min  _ {X _ {j} ^ {*} \in P F} f _ {r} \left(X _ {j} ^ {*}\right)} (25) \\ \end{array}
$$

# 4.2. Experimental setting

MOSSA is compared with three well-known algorithms, namely NSGA-II, MOPSO, and MOGWO to verify the performance of MOSSA. In the above four MOEAs, the number of the population  $m$  is set to 100, and the maximum number of iterations  $T$  reaches 2000. All algorithms were executed 20 times on each test problem to avoid the influence of randomness on the experimental results. All experiments have been run using Matlab2020b on a 64-bit Windows 10 based computer with an AMD Ryzen 53600X processor clocking at 3.8 GHz and 16 GB of RAM. The parameters of MOSSA are set as:

- The value of  $k$  in  ${2k}$  crowding-distance entropy is 2 .

-  $ST^{max} = 0.9, ST^{min} = 0.7; p_{E}^{max} = 0.3, p_{E}^{min} = 0.2; p_{D}^{max} = 0.9, p_{D}^{min} = 0.2.$

The parameters of NSGA-II are set as:

- The crossover rate  $p_c = 0.6$  and the mutation rate  $p_m = 0.5$ .

- The distribution indices in the simulated binary crossover and polynomial mutation are set to  $\eta_c = 20$  and  $\eta_m = 20$ , respectively.

The parameters of MOPSO are set as:

- The size of the repository is set to 100.

- Inertia weight  $\omega = \frac{2}{(\phi - 2 + \sqrt{\phi^2 + 4\phi})}$ , where  $\phi = 4.25$ .

- The personal learning coefficient  $c_{1} = 1.65$  and the global learning coefficient  $c_{2} = 1.7$ .

- The number of grids per dimension  $nGrid = 10$  and the inflation rate  $\alpha Grid = 0.1$ .

- The leader and deletion selection pressure are set to 2.

The size of the repository, the number of grids per dimension, the inflation rate, the leader and deletion selection pressure of MOGWO are all the same as MOPSO.

# 4.3. Comparison results between 2k crowding-distance entropy and crowding distance

To explore the effect of  $2k$  crowding-distance entropy on the performance of MOSSA, this study uses MOSSA with crowding distance and  $2k$  crowding-distance entropy to solve eight test problems, respectively. The average IGD, SP, MS, and runtime obtained by the two kinds of MOSSA are listed in Table 2, where CD and CDE represent the MOSSA

with crowding distance and  $2k$  crowding-distance entropy, respectively. In this section, MOSSA with crowding distance and MOSSA with  $2k$  crowding-distance entropy are denoted as MOSSA-CD and MOSSA-CDE, respectively. The IGD and SP of MOSSA-CDE in IMOP1-8 are smaller than those of MOSSA-CD except for IMOP1 and IMOP4, which indicates that  $2k$  crowding-distance entropy can make MOSSA have better convergence and evenness. And the MS of the two algorithms is almost the same, suggesting that  $2k$  crowding-distance entropy can measure the spread of the Pareto solution set as accurately as crowding distance. The average runtime shows that the calculation of  $2k$  crowding-distance entropy takes a little more time. Although  $2k$  crowding-distance entropy adds about one second of computation time, the performance of the MOSSA-CDE can be significantly improved.

# 4.4. Impact of the  $k$  in 2k crowding-distance entropy

This study selects  $k$  to take 1, 2, 3, and 4 to analyse the impact of the  $k$  in  $2k$  crowding-distance entropy on the convergence and diversity of MOSSA, that is, when calculating  $2k$  crowding-distance entropy, it is necessary to consider 1, 2, 3, and 4 pairs of adjacent points. Fig. 10 shows the relationship curves between the mean IGD and  $k$  value of each test problem.

It can be seen from the 8 curves that these curves appear two changing trends with the increase of  $k$ . The first is that the mean IGD decreases first and then increases as the  $k$  increases, such as the curves of IMOP1, IMOP2, IMOP3, IMOP4, and IMOP6. In these MOPs except IMOP6, when  $k$  is equal to 2, MOSSA has the best performance, and  $k$  greater than or less than 2 will cause MOSSA to fail to converge to the optimal Pareto front. This fact indicates that too many or too few pairs of adjacent points make the measurement of diversity inaccurate in the calculation of  $2k$  crowding-distance entropy. The second is that the mean IGD decreases first and then converges to a certain value as the  $k$  increases, such as the curves of IMOP5, IMOP7, and IMOP8. When  $k$  takes 2 or greater, MOSSA can achieve the best performance. In these MOPs, although more pairs of adjacent points give a small performance boost to MOSSA, a larger  $k$  value requires more computing time.

The average runtime of the MOSSA with different  $k$  is listed in Table 3. In IMOP1 to IMOP8, the average runtime of the MOSSA keeps increasing with the increase of  $k$ . And the increase of  $k$  does not cause a sharp increase in the runtime. The MOSSA obtain the optimal Pareto set when  $k$  takes 2 in IMOP1-3, IMOP5, and IMOP7. If  $k$  takes 3, the MOSSA has the best performance of IMOP4 and IMOP6. For IMOP8, when  $k$  is equal to 4, MOSSA converges to the optimal Pareto front.

# 4.5. Comparison results of the four MOEAs

It can be seen from the above results that when  $k$  is equal to 2, MOSSA has a good performance in solving IMOP1-8. Therefore, this study compares the performance of MOSSA with  $k = 2$  with the other three MOEAs using IMOP1-8. In Tables 4-7, “+”, “=”, and “-” shows that the performance of MOSSA is better, similar to or worse than that of the other MOEAs, respectively. The last row “+/=-” in the table is a summary of the performance of MOSSA compared to other algorithms


Table 2



The mean results of MOSSA-CD and MOSSA-CDE over 20 runs.


<table><tr><td rowspan="2"></td><td colspan="2">IGD</td><td colspan="2">SP</td><td colspan="2">MS</td><td colspan="2">Runtime (s)</td></tr><tr><td>CD</td><td>CDE(k=2)</td><td>CD</td><td>CDE(k=2)</td><td>CD</td><td>CDE(k=2)</td><td>CD</td><td>CDE(k=2)</td></tr><tr><td>IMOP1</td><td>7.666E-03</td><td>8.146E-03</td><td>9.534E-03</td><td>1.131E-02</td><td>9.943E-01</td><td>9.876E-01</td><td>342.06</td><td>342.98</td></tr><tr><td>IMOP2</td><td>2.242E-02</td><td>1.976E-02</td><td>3.431E-02</td><td>3.132E-02</td><td>1.000E+00</td><td>1.000E+00</td><td>322.03</td><td>323.02</td></tr><tr><td>IMOP3</td><td>2.585E-02</td><td>1.348E-02</td><td>4.813E-02</td><td>8.767E-03</td><td>1.000E+00</td><td>1.000E+00</td><td>321.29</td><td>322.15</td></tr><tr><td>IMOP4</td><td>1.407E-02</td><td>2.194E-02</td><td>1.578E-02</td><td>3.139E-02</td><td>1.000E+00</td><td>1.000E+00</td><td>261.28</td><td>261.93</td></tr><tr><td>IMOP5</td><td>5.643E-02</td><td>5.016E-02</td><td>4.523E-02</td><td>4.172E-02</td><td>9.851E-01</td><td>9.844E-01</td><td>320.86</td><td>321.47</td></tr><tr><td>IMOP6</td><td>6.892E-02</td><td>6.766E-02</td><td>5.717E-02</td><td>5.279E-02</td><td>1.000E+00</td><td>1.000E+00</td><td>263.66</td><td>264.35</td></tr><tr><td>IMOP7</td><td>1.107E-01</td><td>6.153E-02</td><td>6.866E-02</td><td>6.526E-02</td><td>1.000E+00</td><td>1.000E+00</td><td>288.58</td><td>289.11</td></tr><tr><td>IMOP8</td><td>1.423E-01</td><td>1.308E-01</td><td>2.066E-01</td><td>1.208E-01</td><td>9.833E-01</td><td>9.925E-01</td><td>266.04</td><td>266.69</td></tr></table>

![](images/d432bd9051e0c893a8fbb3152891b0a39026531205f58e1df0b6bbdab0487bd3.jpg)



IMOP1


![](images/3ea80bc9fa471323cc8bf539d698f6c267de421fae3a93cbc421f71adf917102.jpg)



IMOP2


![](images/b07f01d8fe2e3ec5abb95f31170d3d2e28dcb94c73e355396341289d904a4be7.jpg)



IMOP3


![](images/9235b71ef992fd93a14c3aa7b8cd247cfafe80a14f09ae442cddc684b9b7e0a1.jpg)



k IMOP4


![](images/ca461e090a5cab5df93ac96df71d6743fe2dba1e140cb1023e6fbdc57f3e2218.jpg)



IMOP5


![](images/028b1388971fcf9c812317a537b301ee593a8a1b75b61443854a28c3636b5404.jpg)



k IMOP6


![](images/05d3e9778f7b8900c21b11b598fa837aba0cc475285fe130f4b211d19015dae0.jpg)



IMOP7


![](images/9f8a1164a178a5ac0819af08631f90c40c165ec99f558e0812bbcbaede4c4081.jpg)



IMOP8



Fig. 10. The relationship curves between the mean IGD and  $k$  value.



Table 3 Mean runtime of the MOSSA with different  $k$  over 20 runs.


<table><tr><td>Runtime (s)</td><td>k=1</td><td>k=2</td><td>k=3</td><td>k=4</td></tr><tr><td>IMOP1</td><td>341.82</td><td>342.98</td><td>343.49</td><td>344.29</td></tr><tr><td>IMOP2</td><td>322.43</td><td>323.02</td><td>324.14</td><td>324.72</td></tr><tr><td>IMOP3</td><td>321.55</td><td>322.15</td><td>322.9</td><td>323.96</td></tr><tr><td>IMOP4</td><td>261.39</td><td>261.93</td><td>262.69</td><td>263.22</td></tr><tr><td>IMOP5</td><td>320.59</td><td>321.47</td><td>322.43</td><td>322.99</td></tr><tr><td>IMOP6</td><td>263.75</td><td>264.35</td><td>264.93</td><td>265.47</td></tr><tr><td>IMOP7</td><td>288.6</td><td>289.11</td><td>290.21</td><td>290.77</td></tr><tr><td>IMOP8</td><td>265.61</td><td>266.69</td><td>267.64</td><td>268.66</td></tr></table>

and the optimal value is bolded. Figs. 11 and 12 show the comparison between the true Pareto front and the obtained Pareto front by four MOEAs. The red dots are the obtained Pareto front, while the blue dots represent the true Pareto front.

Table 4 shows the mean IGD of the four MOEAs. In IMOP1, IMOP3, and IMOP4, NSGA-II has the smallest IGD, followed by MOSSA. Among four MOEAs, the IGD of MOPSO and MOSSA is optimal and sub-optimal in IMOP2, respectively. In IMOP5-8, MOSSA achieves better IGD than that of the other three algorithms, which indicates that the convergence and diversity of the Pareto solution set obtained by MOSSA are better


Table 4 The mean IGD of the four MOEAs over 20 runs.


<table><tr><td>IGD</td><td>MOSSA</td><td>NSGA-II</td><td>MOPSO</td><td>MOGWO</td></tr><tr><td>IMOP1</td><td>8.146E-03</td><td>7.188E-03(-)</td><td>1.053E-02(+)</td><td>1.657E-01(+)</td></tr><tr><td>IMOP2</td><td>1.976E-02</td><td>1.217E-01(+)</td><td>1.377E-02(-)</td><td>7.037E-01(+)</td></tr><tr><td>IMOP3</td><td>1.348E-02</td><td>5.070E-03(-)</td><td>1.121E-01(+)</td><td>3.555E-02(+)</td></tr><tr><td>IMOP4</td><td>2.194E-02</td><td>1.139E-02(-)</td><td>2.444E-02(+)</td><td>4.814E-02(+)</td></tr><tr><td>IMOP5</td><td>5.016E-02</td><td>5.343E-02(+)</td><td>2.582E-01(+)</td><td>1.259E-01(+)</td></tr><tr><td>IMOP6</td><td>6.766E-02</td><td>3.504E-01(+)</td><td>1.587E-01(+)</td><td>1.386E-01(+)</td></tr><tr><td>IMOP7</td><td>6.153E-02</td><td>7.193E-02(+)</td><td>1.451E-01(+)</td><td>2.824E-01(+)</td></tr><tr><td>IMOP8</td><td>1.308E-01</td><td>1.446E-01(+)</td><td>2.060E-01(+)</td><td>2.082E-01(+)</td></tr><tr><td>+/-=-</td><td></td><td>5/0/3</td><td>7/0/1</td><td>8/0/0</td></tr></table>

than those of the other MOEAs. And the last row  $\frac{+/ = / - }{}$  of MOSSA shows that the comprehensive performance of MOSSA in IMOP1-8 is much better than that of NSGA-II, MOPSO, and MOGWO.

The mean SP of the four MOEAs is present in Table 5. Since SP cannot take into account the spread of the Pareto front, the evenness of the Pareto set should be analyzed in conjunction with images. In IMOP2, the SP of MOSSA is larger than that of the others, but the obtained Pareto fronts of NSGA-II and MOGWO gather in one place and do not spread out, which results in their SP being smaller. That is, the SP of NSGA-II


Table 5 The mean SP of the four MOEAs over 20 runs.


<table><tr><td>SP</td><td>MOSSA</td><td>NSGA-II</td><td>MOPSO</td><td>MOGWO</td></tr><tr><td>IMOP1</td><td>1.131E-02</td><td>8.543E-03(-)</td><td>9.246E-03(-)</td><td>8.606E-02(+)</td></tr><tr><td>IMOP2</td><td>3.132E-02</td><td>4.114E-03(-)</td><td>1.486E-02(-)</td><td>2.447E-03(-)</td></tr><tr><td>IMOP3</td><td>8.767E-03</td><td>8.865E-03(+)</td><td>8.578E-02(+)</td><td>5.797E-02(+)</td></tr><tr><td>IMOP4</td><td>3.139E-02</td><td>1.890E-02(-)</td><td>3.537E-02(+)</td><td>8.370E-02(+)</td></tr><tr><td>IMOP5</td><td>4.172E-02</td><td>5.002E-02(+)</td><td>4.043E-02(-)</td><td>8.288E-02(+)</td></tr><tr><td>IMOP6</td><td>5.279E-02</td><td>1.437E-02(-)</td><td>4.209E-02(-)</td><td>2.429E-02(-)</td></tr><tr><td>IMOP7</td><td>6.526E-02</td><td>4.537E-02(-)</td><td>9.435E-02(+)</td><td>2.403E-02(-)</td></tr><tr><td>IMOP8</td><td>1.208E-01</td><td>8.067E-02(-)</td><td>1.076E-01(-)</td><td>8.728E-02(-)</td></tr><tr><td>+/-=/-</td><td></td><td>6/0/2</td><td>3/0/5</td><td>4/0/4</td></tr></table>


Table 6 The mean MS of the four MOEAs over 20 runs.


<table><tr><td>MS</td><td>MOSSA</td><td>NSGA-II</td><td>MOPSO</td><td>MOGWO</td></tr><tr><td>IMOP1</td><td>9.876E-01</td><td>9.942E-01(-)</td><td>9.717E-01(+)</td><td>7.779E-01(+)</td></tr><tr><td>IMOP2</td><td>1.000E + 00</td><td>9.885E-01(+)</td><td>1.000E + 00(=)</td><td>6.426E-02(+)</td></tr><tr><td>IMOP3</td><td>1.000E + 00</td><td>1.000E + 00(=)</td><td>9.981E-01(+)</td><td>9.972E-01(+)</td></tr><tr><td>IMOP4</td><td>1.000E + 00</td><td>1.000E + 00(=)</td><td>1.000E + 00(=)</td><td>9.999E-01(+)</td></tr><tr><td>IMOP5</td><td>9.844E-01</td><td>9.993E-01(-)</td><td>6.960E-01(+)</td><td>9.024E-01(+)</td></tr><tr><td>IMOP6</td><td>1.000E + 00</td><td>1.000E + 00(=)</td><td>1.000E + 00(=)</td><td>7.404E-01(+)</td></tr><tr><td>IMOP7</td><td>1.000E + 00</td><td>1.000E + 00(=)</td><td>1.000E + 00(=)</td><td>5.312E-01(+)</td></tr><tr><td>IMOP8</td><td>9.925E-01</td><td>1.000E + 00(-)</td><td>9.847E-01(+)</td><td>8.083E-01(+)</td></tr><tr><td>+/-=-</td><td></td><td>1/4/3</td><td>4/4/0</td><td>8/0/0</td></tr></table>


Table 7 The mean runtime of the four MOEAs over 20 runs.


<table><tr><td>Runtime (s)</td><td>MOSSA</td><td>NSGA-II</td><td>MOPSO</td><td>MOGWO</td></tr><tr><td>IMOP1</td><td>342.98</td><td>312.16(-)</td><td>133.46(-)</td><td>151.74(-)</td></tr><tr><td>IMOP2</td><td>323.02</td><td>315.85(-)</td><td>135.67(-)</td><td>66.45(-)</td></tr><tr><td>IMOP3</td><td>322.15</td><td>298.74(-)</td><td>78.57(-)</td><td>171.02(-)</td></tr><tr><td>IMOP4</td><td>261.93</td><td>311.38(+)</td><td>131.35(-)</td><td>251.06(-)</td></tr><tr><td>IMOP5</td><td>321.47</td><td>305.31(-)</td><td>76.02(-)</td><td>304.87(-)</td></tr><tr><td>IMOP6</td><td>264.35</td><td>302.63(+)</td><td>72.23(-)</td><td>299.18(+)</td></tr><tr><td>IMOP7</td><td>289.11</td><td>314.22(+)</td><td>72.56(-)</td><td>166.32(-)</td></tr><tr><td>IMOP8</td><td>266.69</td><td>274.06(+)</td><td>71.54(-)</td><td>294.53(-)</td></tr><tr><td>+/-=/-</td><td></td><td>4/0/4</td><td>8/0/0</td><td>8/0/0</td></tr></table>

and MOGWO in IMOP2 is invalid. The obtained Pareto front of MOPSO is widely distributed. Therefore, the evenness of MOSSA in IMOP2 is inferior to MOPSO but superior to NSGA-II and MOGWO. In IMOP5, although the SP of MOPSO is smaller than that of MOSSA, the Pareto front of MOPSO is concentrated in two places and is not scattered enough. In fact, the evenness of MOSSA is better than that of the other algorithms in IMOP5. Similarly, the evenness of MOSSA outperforms the other algorithms in IMOP6. It can be seen from Fig. 11 that the evenness comparison results of IMOP1, IMOP3, IMOP4, IMOP7, and IMOP8 in the table are valid. Therefore, the last row “+/-” of NSGA-II, MOPSO, and MOGWO in Table 5 should be “4/0/4”, “5/0/3”, and “6/0/2”, respectively, which indicates that MOSSA has the best evenness in solving IMOP1-8.

Table 6 is the mean MS of the four MOEAs. From the last row  $+ / = / -$  in the Table 6, the spread of MOSSA is almost the same as NSGA-II, slightly better than that of MOPSO and MOGWO. The mean runtime of the four MOEAs are presented in Table 7. The last row  $+ / = / -$  in the Table 7 shows that the runtime of MOSSA and NSGA-II is similar, but both are greater than the that of MOPSO and MOGWO. These results show that the improvement of MOSSA performance comes at the cost of consuming more runtime.

From Fig. 11, for bi-objective MOPs, the performance of MOSSA in evenness and spread is similar to that of MOPSO, but the convergence of the former is better than that of the latter. NSGA-II has a poor performance in IMOP2, which may be caused by the weak simulated binary crossover operator and polynomial mutation. It can be seen from Fig. 11 that MOSSA has the best comprehensive performance in solving three bi

objective MOPs.

From Figs. 11 and 12, for tri-objective MOPs, MOSSA performs similar to NSGA-II in IMOP4, IMOP5, IMOP7, and IMOP8, and outperforms both MOPSO and MOGWO. In IMOP6, MOSSA outperforms NSGA-II, which is due to  $2k$  crowding-distance entropy that can make the population of MOSSA have the better diversity to converge to the true Pareto front. Compared with the obtained Pareto fronts of MOPSO and MOGWO, the convergence, evenness, and spread of MOSSA are much better than those of MOPSO and MOGWO. Particularly, MOSSA outperforms the other three MOEAs in IMOP6. The excellent performance of MOSSA in tri-objective MOPs suggests that  $2k$  crowding-distance entropy may be more suitable for addressing tri-objective MOPs.

In summary, the MOSSA can provide competitive Pareto solutions in addressing MOPs, and the comprehensive performance of MOSSA is better than that of the other three MOEAs.

# 4.6. Analysis of the performance of MOSSA

In the four MOEAs, MOSSA is extended for SSA based on the expansion method of NSGA-II and MOGWO is extended for GWO based on the expansion method of MOPSO. The comprehensive performance of MOSSA and NSGA-II is much better than that of MOPSO and MOGWO, which indicates that the expansion method of NSGA-II also suits swarm intelligence algorithms similar to PSO. It can be seen from Tables 4-7 that the MOSSA have competitive convergence and diversity but need more runtime. The better performance is due to the fact that the expansion method of NSGA-II is suitable for SSA, and the computational complexity of this expansion method required more runtime.

In Tables 2 and 4, the mean IGD of MOSSA with crowding distance is similar to that of NSGA-II from IMOP1 to IMOP8. The IGD of MOSSA with crowding distance and NSGA-II in IMOP1 and IMOP4 is better than that of MOSSA with  $2k$  crowding-distance entropy, suggesting that the crowding distance is more suitable for solving IMOP2 and IMOP4. However, the comprehensive performance of MOSSA with  $2k$  crowding-distance entropy is better than MOSSA with crowding distance and NSGA-II, which indicates that  $2k$  crowding-distance entropy brings excellent diversity to MOSSA so the performance of MOSSA has been greatly improved. The performance of MOSSA with  $2k$  crowding-distance entropy and crowding distance shows that the use of  $2k$  crowding-distance entropy achieves a significant performance improvement at the cost of a little additional time cost.

The experimental results of MOSSA with different  $k$  show that too large or too small  $k$  in  $2k$  crowding-distance entropy cannot make MOSSA obtain the Pareto optimal set, that is, the optimal performance of the MOSSA can only be achieved by using suitable pairs of adjacent points. It can be seen from the performance of the above eight test problems that when  $k$  is set to 2, MOSSA can achieve the best convergence and diversity on most MOPs. IMOP4 and IMOP6 show that appropriately increasing  $k$  by one can make MOSSA obtain better optimisation results for some MOPs.

The overall performance of MOSSA on these eight complex unconstrained MOPs is superior to NSGA-II, MOPSO, and MOGWO. Compared with NSGA-II, MOPSO, and MOGWO, the main advantages of MOSSA are high convergence, good evenness, and wide spread. The modified formulas for updating positions, adaptive parameters, and optimal strategy of the positions make the Pareto front constantly converge towards the true Pareto front, which causes high convergence.

# 5. Testing MOSSA in a complex engineering problem

The multi-objective optimal scheduling problem of microgrid (MG) is adopted in the experiment to deeply test the performance of MOSSA. This experiment can test the performance of MOSSA in dealing with a complex engineering problem and explore the ability of MOSSA for resolving the MOP with equality and inequality constraints.

![](images/7db4ea97602655d5ff12cbd699d7c27c5d4efadc176875343ddf94be8c8a6b4d.jpg)


![](images/d141ed43c2ccf5130b76bc2a44dbf15bed2193b612944c2ffb9dbea0943d5b75.jpg)


![](images/540bea05f8177c93d519b52a20c967a8304bc2451deaaf68e9eb2245e6ab4848.jpg)


![](images/0e284eaf22e3453bb46ba121fcfba5f2738728e43fe620035fcb11db9b63abd1.jpg)


![](images/0f9223d881be3247d10006c83bc9984a5c52813431fb5a8e38ff6a5d9923f1f7.jpg)


![](images/f261ec5a8827c3ecbe01962a22189fe1378c6f240a2ab4545acd9d0dd5074662.jpg)


![](images/4a3810df815b7d6d4725f0979bb3cd203a2d6de4d5e814eb040f9563b5b507ef.jpg)


![](images/aebac1d9a0828c987b5f1ff4f6885d9784b05baa00eb6cf974098075fc5a15f2.jpg)


![](images/6715647f332c28b723e00570ad36d98f3059cdc2c8dca79945b66bb941fb5a7c.jpg)


![](images/8ed07ba0f5eea759aad8408c88c43ffbb76af7364f299c65e832fa9aa2e06235.jpg)


![](images/11b2671fd8f49aa521dc79a75bf4a385054eeebb9c35bcfd034e7ce6592c1883.jpg)


![](images/a8e838cf5e4684fa2bfc31c06c1358e3f31df3cc67a896766b6ac860d7a9a68b.jpg)


![](images/75240d3bfb7330b09c00be46faed2883c140843c28a8700618459a274c392e5c.jpg)


![](images/b838772da3a2ca3b23f1a8a97e04d61174a36da6f6ed1de9b3737cd3cd29b879.jpg)


![](images/d4d58c67dfa49e2db2eab4daa888a7bf440ce44f6e197397e84e73fca700fc32.jpg)


![](images/1114b0daf2566a1f817f4be03bfd099efde268291fea5abf0c2161fc393b7132.jpg)



Fig. 11. Obtained Pareto fronts by MOSSA, NSGA-II, MOPSO, and MOGWO for IMOP1 to IMOP4.


# 5.1. Optimisation model

The grid-connected MG is used as the research object, and its structure is shown in Fig. 13. The MG includes wind turbines (WT), photovoltaic panels (PV), an energy storage system (ESS), and a micro turbine (MT). The MG is connected to the distribution network that can serve as support. The time interval is  $1\mathrm{h}$  and there are 24 scheduling periods in a day. This study takes the output power of ESS and MT as decision variables, treats the output power of WT and PV and the load power as input, and applies MOSSA, NSGA-II, MOPSO, and MOGWO to solve this MOP, respectively.

# 5.1.1. Models of MG

# (1) Model of ESS

The output power of ESS at time  $t$  is one of the decision variables for the scheduling of MG (Shuai, Fang, Ai, Wen, & He, 2019). Its state of charge (SOC) at time  $t$  SOC(t) can be expressed as follows:

$$
S O C (t) = S O C (t - 1) + \frac {P _ {E S S} ^ {c} (t) \eta_ {E S S} ^ {c} - P _ {E S S} ^ {d} (t) / \eta_ {E S S} ^ {d}}{E _ {E S S} ^ {r}} \tag {26}
$$

where,  $P_{ESS}^{c}(t)$  and  $P_{ESS}^{d}(t)$  are the charging and discharging power of ESS, respectively;  $\eta_{ESS}^{c}$  and  $\eta_{ESS}^{d}$  are the charging and discharging efficiency of ESS, respectively, 0.8 and 0.9;  $E_{ESS}^{r}$  is the rated capacity of ESS, which is 1000kWh.

The output power of ESS  $P_{ESS}(t)$  is defined as follows:

$$
P _ {E S S} (t) = P _ {E S S} ^ {c} (t) - P _ {E S S} ^ {d} (t) \tag {27}
$$

The following constraint ensures that the ESS doesn't have simultaneous charging and discharging.

$$
\forall t P _ {E S S} ^ {c} (t) \cdot P _ {E S S} ^ {d} (t) = 0 \tag {28}
$$

# (2) Model of MT

MT is a new type of generator that transfers the chemical energy of natural gas into electricity (Cui et al., 2020). Its generation efficiency  $\eta_{MT}(t)$  is expressed as follows:

$$
\eta_ {M T} (t) = 0. 0 7 5 3 \left(\frac {P _ {M T} (t)}{P _ {M T} ^ {r}}\right) ^ {3} - 0. 3 0 9 5 \left(\frac {P _ {M T} (t)}{P _ {M T} ^ {r}}\right) ^ {2} + 0. 4 1 7 4 \frac {P _ {M T} (t)}{P _ {M T} ^ {r}} + 0. 1 0 6 8 \tag {29}
$$

where,  $P_{MT}(t)$  represents the output power of MT at time  $t$ ;  $P_{MT}^{r}$  is the rated power of MT.

The fuel cost of MT  $C_{fuel}$  is defined as follows:

$$
C _ {\text {f u e l}} = C _ {\text {n g}} \frac {1}{L H V} \sum_ {t = 1} ^ {2 4} \frac {P _ {M T} (t)}{\eta_ {e} (t)} \tag {30}
$$

where,  $C_{ng}$  is the natural gas price, 2.65 RMB/m $^3$ ;  $LHV$  is the lower heat value of natural gas, 9.7 kWh/m $^3$ .

# (3) Scheduling strategy

According to the output power of  $\mathrm{WT}P_{WT}(t)$  and PV  $P_{PV}(t)$  and the demand power of load  $P_{Load}(t)$ , the difference power  $\Delta P(t)$  at time  $t$  can

![](images/47bd696cc82d6f13e9029bfe87dd0ce707ce6bcc7bc9415cb51cf2290dc09909.jpg)



Fig. 12. Obtained Pareto fronts by MOSSA, NSGA-II, MOPSO, and MOGWO for IMOP5 to IMOP8.


![](images/312ff86cb6d648b759a88c0cd1ae6e66d682d646d7d0de0fffdc7e085ecec9a0.jpg)



Fig. 13. The structure of the MG.


be obtained as follows:

$$
\Delta P (t) = P _ {W T} (t) + P _ {P V} (t) - P _ {L o a d} (t) \tag {31}
$$

The difference power is compensated by ESS, MT and the distribution network.

If  $\Delta P(t) \geq 0$ , renewable energy sources (RES) on the supply side are surplus, and MT doesn't need to work. The MG flushes surplus RES into the ESS or sells them to the distribution network.

If  $\Delta P(t) < 0$ , RES are insufficient. The MG achieves the power balance between the supply side and the demand side through ESS discharge, MT generation, or power purchase from the distribution network.

# 5.1.2. Objectives

Considering the economy and environmental protection of the system operation, two objectives of the operating cost and the environmental treatment cost are established to ensure the economic and environmental protection of the MG.

# (1) Operating cost of MG

The minimum operating cost of the MG in one day is taken as the objective function one  $f_{1}$ , it is defined as follows:

$$
\operatorname {m i n f} _ {1} = C _ {o m} + C _ {f u e l} + C _ {t r a n} \tag {32}
$$

where,  $C_{om}$  is the operation and maintenance cost of MG;  $C_{tran}$  is the cost of energy transaction between the MG and the distribution network.

$C_{om}$  is the sum of the operation and maintenance costs of all units, expressed as follows:

$$
C _ {o m} = K _ {W T} ^ {o m} \left| P _ {W T} (t) \right| + K _ {P V} ^ {o m} \left| P _ {P V} (t) \right| + K _ {E S S} ^ {o m} \left| P _ {E S S} (t) \right| + K _ {M T} ^ {o m} \left| P _ {M T} (t) \right| \tag {33}
$$

where,  $K^{om}$  is the operation and maintenance coefficient. Table 8 summarises the rated power and the operation and maintenance coefficients of each unit.

$C_{tran}$  is defined as the difference between the purchase and sale cost of the electricity of the MG to the distribution network. It can be written as follows:

$$
C _ {t r a n} = \sum_ {t = 1} ^ {2 4} \left(\rho_ {b u y} (t) P _ {b u y} (t) - \rho_ {s e l l} (t) P _ {s e l l} (t)\right) \tag {34}
$$

where,  $P_{buy}(t)$  and  $P_{sell}(t)$  represent the purchasing power and the selling power by the MG to the distribution network at time  $t$ , respectively;  $\rho_{buy}(t)$  and  $\rho_{sell}(t)$  represent the purchasing price and the selling price, respectively. MG cannot buy and sell electricity to the distribution network at the same time, therefore, the following constraint need to be added.

$$
\forall t P _ {b u y} (t) \cdot P _ {s e l l} (t) = 0 \tag {35}
$$

# (2) Environmental treatment cost

The electricity generated by MT and the electricity purchased from the distribution network will cause environmental pollution in the process of power generation. Only considering the economical scheduling cannot meet the requirement of building an environmental-friendly society. Therefore, it is necessary to take the minimum environmental treatment cost as the objective function two  $f_{2}$ , and  $f_{2}$  is defined as follows:

$$
\operatorname {m i n f} _ {2} = \sum_ {t = 1} ^ {2 4} \sum_ {j = 1} ^ {3} \left(c _ {j} u _ {j} ^ {M T} P _ {M T} (t) + c _ {j} u _ {j} ^ {D N} P _ {b u y} (t)\right) \tag {36}
$$

where,  $c_{j}$  is the treatment cost of the  $j$ -th pollutant;  $u_{j}^{MT}$  and  $u_{j}^{DN}$  represent the emission coefficient of  $j$ -th pollutant of the MT and


Table 8



The rated power and the operation and maintenance coefficients of each unit.


<table><tr><td>Type</td><td>WT</td><td>PV</td><td>ESS</td><td>MT</td></tr><tr><td>Rated power (kW)</td><td>250</td><td>200</td><td>60</td><td>65</td></tr><tr><td>Kom(RMB/kWh)</td><td>0.0296</td><td>0.0096</td><td>0.045</td><td>0.0648</td></tr></table>

distribution network, respectively. The parameters of carbon dioxide and pollutant emissions are shown in Table 9.

# 5.1.3. Constraints

(1) Power balance of MG:

$$
P _ {W T} (t) + P _ {P V} (t) + P _ {E S S} (t) + P _ {M T} (t) + P _ {b u t} (t) + P _ {s e l l} (t) = P _ {L o a d} (t) \tag {37}
$$

(2) Output power limits of all units:

$$
P _ {i} ^ {\text {m i n}} \leq P _ {i} (t) \leq P _ {i} ^ {\text {m a x}} \tag {38}
$$

where,  $P_{i}(t)$  is the output power of  $i$ -th unit;  $P_{i}^{min}$  and  $P_{i}^{max}$  are the minimum power and maximum power.

(3) Ramp rate limits of MT:

$$
R _ {M T} ^ {\text {d o w n}} \leq P _ {M T} (t + 1) - P _ {M T} (t) \leq R _ {M T} ^ {\text {u p}} \tag {39}
$$

where,  $R_{MT}^{down}$  and  $R_{MT}^{up}$  are Ramp-down and ramp-up rate limits, respectively.

(4) SOC constraint of ESS:

$$
S O C ^ {\min } \leqslant S O C (t) \leqslant S O C ^ {\max } \tag {40}
$$

where,  $SOC^{min}$  and  $SOC^{max}$  represent the minimum and maximum SOC of ESS, respectively.

# 5.2. Simulation results

In this section, the results of the four algorithms for the optimal scheduling of the MG are compared. The performance of MOSSA in addressing a complex engineering problem with equality and inequality constraints is analyzed.

# 5.2.1. Case illustration

The output power curve of WT and PV and the power curve of load of the MG are shown in Fig. 14 (a). Fig. 14. (b) is the time-of-use (TOU) price between the MG and distribution network (Ajoulabadi, Ravadanegh, & Behnam, 2020). The data in Fig. 14 is used as the input of this problem.

Since there is no true Pareto front in the engineering problem that can be compared with the obtained Pareto front, the previous performance metrics cannot be used which require the true Pareto front to participate in the calculation. Therefore, this study uses hypervolume (HV) (Zitzler & Thiele, 1999) to evaluate the performance of the four algorithms. HV not only does not need the true Pareto front to participate in the calculation, but also can measure the convergence and diversity of the solution set. HV is defined as follows:

$$
H V \left(P F, Z ^ {r e f}\right) = \wedge \left(\bigcup_ {X ^ {*} \in P F} \left\{X \mid X ^ {*} \succ X \succ Z ^ {r e f} \right\}\right) \tag {41}
$$

where,  $\wedge$  is the Lebesgue measure;  $Z^{ref}$  is reference point. 1000 RMB operating cost and 200 RMB environmental treatment cost are selected as the reference point, namely  $Z^{ref} = (1000, 200)$ . HV calculates the volume enclosed by the obtained Pareto front and the reference point. A larger HV indicates that the algorithm has superior overall performance. The parameters in MOSSA, NSGA-II, MOPSO, and MOGWO are the same to Section 4.


Table 9



The parameters of carbon dioxide and pollutant emissions.


<table><tr><td rowspan="2">Type</td><td colspan="2">Pollutant emission coefficient (g/kWh)</td><td rowspan="2">Treatment cost (RMB/kg)</td></tr><tr><td>MT</td><td>Distribution network</td></tr><tr><td>CO2</td><td>724</td><td>889</td><td>0.21</td></tr><tr><td>SO2</td><td>0.0036</td><td>1.8</td><td>6</td></tr><tr><td>NOx</td><td>0.2</td><td>1.6</td><td>8</td></tr></table>

![](images/8dcdc81d438d2f2a97e54e466ac727d5ef227fd0c6ae843967a43571fe10bb2d.jpg)



(a)


![](images/cf3d5fa5517141b87444e502e75cffa321f458ee76fcd57febe50832e4d44d63.jpg)



(b)



* The data is provided by China Southern Power Grid CO., LTD.



Fig. 14. (a) The power of the WT, PV, and load and (b) TOU price between MG and distribution network. * The data is provided by China Southern Power Grid CO., Ltd.


# 5.2.2. Optimisation results and analysis

All algorithms were executed 20 times to avoid randomness, and the average HV of four MOEAs are provided in Table 10. MOSSA has the largest average HV in solving the multi-objective optimal scheduling problem of MG, that is, the convergence and diversity of MOSSA are the best. The mean runtime of MOSSA is the longest, which suggests that the performance improvement will consume more time.

The four obtained Pareto fronts are shown in Fig. 15 (a). Compared with other algorithms, the Pareto front of MOSSA is closest to the two coordinate axes, which means that when the operating cost is the same, the environmental treatment cost by MOSSA is smaller. This fact shows that the Pareto front of MOSSA dominates the Pareto fronts of other algorithms, that is, the scheduling results obtained by MOSSA are better than those of other algorithms. The coverage of MOSSA is also superior to the others, and it can provide the operators of MG with more solutions. In Fig. 15 (b), the boxplot of MOSSA is much higher than the others, followed by NSGA-II, MOGWO, and MOPSO. The results of 20 runs by MOSSA are almost better than the others. The narrowest boxplot and the smallest standard deviation of MOSSA indicate that MOSSA has superior robustness on this MOP.

Therefore, MOSSA in solving the complex engineering problem and the MOP with equality and inequality constraints outperforms NSGA-II, MOPSO, and MOGWO. The performance of NSGA-II is second, and MOGWO and MOPSO perform the worst.

Compared with the Pareto fronts of MOPSO and MOGWO, the Pareto fronts of MOSSA and NSGA-II cover wider, which indicates that the MOEAs based on the expansion method of NSGA-II have better spread. The Pareto front of MOSSA outperforms NSGA-II in convergence, which is owing to  $2k$  crowding-distance entropy, modified formulas for updating positions, adaptive parameters, and optimal strategy of the positions. In this complex engineering MOP with equality and inequality constraints, the performance of MOSSA is consistent with the overall performance of MOSSA in IMOP1-8. Therefore, MOSSA also has competitive convergence and diversity in addressing the MOP with equality and inequality constraints.


Table 10 Average HV and runtime of MOSSA, NSGA-II, MOPSO, and MOGWO.


<table><tr><td>Average</td><td>MOSSA</td><td>NSGAII</td><td>MOPSO</td><td>MOGWO</td></tr><tr><td>HV</td><td>0.1608</td><td>0.1535</td><td>0.1366</td><td>0.1378</td></tr><tr><td>Runtime (s)</td><td>306.84</td><td>267.53</td><td>48.34</td><td>93.44</td></tr></table>

# 6. Conclusion

This study proposed a novel MOEA based on SSA, called MOSSA, to solve complex MOPs. Fast non-dominated sorting was used to divide the population into different non-dominated fronts. The proposed  $2k$  crowding-distance entropy was applied to maintain diversity. MOSSA obtained an outstanding global search ability via the modified formulas for updating positions and adaptive parameters. The optimal strategy of the positions was realised using the position archive of the population, which can ensure that the positions of the next generation are better than or not inferior to those of the last generation. MOSSA exhibits excellent global search capability, and the local search capability relies on these technologies.

Eight complex unconstrained test problems and a complex constrained engineering optimisation problem were used to test the performance of MOSSA, NSGA-II, MOPSO, and MOGWO. The performance of MOSSA with  $2k$  crowding-distance entropy and crowding distance indicates that  $2k$  crowding-distance entropy while improving the performance of the algorithm, also requires more running time. MOSSA with  $k = 2$  in the  $2k$  crowding-distance entropy can achieve the Pareto optimal set in most MOPs. The comparison results of the four MOEAs in addressing IMOP1 - IMOP8 demonstrated that MOSSA can provide competitive optimisation results. On the one hand, the statistical data of IGD quantitatively proved that MOSSA has high convergence and diversity; on the other hand, the statistical data of SP and MS and the obtained Pareto fronts quantitatively and qualitatively illustrated the wide spread and good evenness of MOSSA. The results of the multi-objective optimal scheduling of the MG indicate that the performance of MOSSA is superior to that of NSGA-II, MOPSO, and MOGWO in solving the complex engineering problem and the MOP with equality and inequality constraints. These facts indicate that swarm intelligence optimisation algorithms similar to PSO can equally obtain excellent MOEAs based on the expansion approach of NSGA-II.

In future studies, MOSSA will be applied to address engineering problems, particularly MOPs in power systems. Additionally, MOSSA will be compared with other excellent MOEAs to further explore the performance of MOSSA in solving engineering problems.

# CRediT authorship contribution statement

Bin Li: Conceptualization, Methodology, Investigation, Writing - original draft, Writing - review & editing. Honglei Wang:

![](images/dd45a7de633309c06177c6bea0a835b1031c4231e46c22464c752df19f3fd706.jpg)



(a)


![](images/bbc75eefe72a94cf1743ead0aec58fec6b3638f4f9b0cb205b75cc1dbe30f372.jpg)



(b)



Fig. 15. The obtained Pareto fronts by four algorithms.


Conceptualization, Methodology, Supervision, Funding acquisition, Writing - review & editing.

# Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Data availability

Data will be made available on request.

# Acknowledgements

This work is supported by the National Natural Science Foundation of China under Grant 52067004 and the Science and Technology Plan Project of Guizhou Province under Grant [2016]5103.

# Appendix A

The following are some important definitions of MOPs.

Global minimum (Coello et al., 2004): Given a function  $f \colon \Omega \subseteq R^n \to R, \Omega \neq \emptyset, f^* \coloneqq f(X^*) \rangle - \infty, X \in \Omega$  is called the global minimum if and only if

$$
\forall X \in \Omega : f \left(X ^ {*}\right) \leq f (X) \tag {42}
$$

where,  $X = [x_{1}, x_{2}, \dots, x_{n}]$  is a vector of decision variable;  $X^{*} = [x_{1}^{*}, x_{2}^{*}, \dots, x_{n}^{*}]$  is the global minimum solution; the set  $\Omega$  is the feasible region.

Multi-objective optimisation problem (Mirjalili et al., 2016): Find  $X^{*} = [x_{1}^{*}, x_{2}^{*}, \dots, x_{n}^{*}]$  in  $\Omega$  which minimises the vector function  $F(X)$

$$
\operatorname {m i n} F (X) = \left[ f _ {1} (X), f _ {2} (X), \dots , f _ {g} (X) \right] \tag {43}
$$

and satisfy the  $p$  inequality constraints:

$$
g _ {i} (X) \geq 0 (i = 1, 2, \dots , p) \tag {44}
$$

the  $q$  equality constraints:

$$
h _ {j} (X) = 0 (j = 1, 2, \dots , q) \tag {45}
$$

$$
\Omega = \{X | g _ {i} (X) \geq 0 \land h _ {j} (X) = 0, i = 1, 2, \dots , p, j = 1, 2, \dots , q \}
$$

In the scalar objective optimisation problem (SOP), it is easy to compare different decision variables because of the single objective. However, most decision variables in a multi-objective space cannot be directly compared due to multi-criterion comparison metrics. Therefore, Pareto dominance is used to compare two solutions in MOPs.

Pareto dominance (Tawhid & Savsani, 2019): For a given MOP  $\min F(X)$ ,  $X = [x_{1}, x_{2}, \dots, x_{n}]$  dominates  $Y = [y_{1}, y_{2}, \dots, y_{n}]$  (denoted by  $X \succ Y$ ) if and only if.

$$
\forall i \in \{1, 2, \dots , g \}, f _ {i} (X) \leq f _ {i} (Y) \wedge \exists i \in \{1, 2, \dots , g \}, f _ {i} (X) <   f _ {i} (Y) \tag {46}
$$

If solution  $X$  is not dominated by any other solutions, then  $X$  is considered to be the Pareto optimality.

Pareto optimality (Pradhan & Panda, 2012): For a given MOP  $\min F(X)$ , a solution  $X^{*} \in \Omega$  is called Pareto optimality either.

$$
\forall i \in \{1, 2, \dots , g \}, f _ {i} (X) = f _ {i} \left(X ^ {*}\right), \forall X \in \Omega \tag {47}
$$

or

$$
\exists i \in \{1, 2, \dots , g \}, f _ {i} (X) \rangle f _ {i} \left(X ^ {*}\right), \forall X \in \Omega \tag {48}
$$

Pareto optimal set (Ren et al., 2022): For a given MOP  $\min F(X)$ , its Pareto optimal set  $P^*$  is defined as:

$$
P ^ {*} := (X \in \Omega | X \text {i s P a r e t o o p t i m a l i t y}) \tag {49}
$$

Pareto front (Carlos A. Coello Coello & Cortés, 2005): For a given MOP  $\min F(X)$  and its Pareto optimal set  $P^*$ , its Pareto front is defined as:

$$
P F ^ {*} := \{F (X) | X \in P ^ {*} \} \tag {50}
$$

For most engineering problems, the formulaic expressions of their Pareto front cannot be derived. Therefore, the Pareto front is obtained by calculating the objective function corresponding to each solution in Pareto optimal set.

# References



Ahmadi, A. (2016). Memory-based adaptive partitioning (MAP) of search space for the enhancement of convergence in Pareto-based multi-objective evolutionary algorithms. Applied Soft Computing, 41, 400-417. https://doi.org/10.1016/j.asoc.2016.01.029





Ajoulabadi, A., Ravadanegh, S. N., & Behnam, M.-I. (2020). Flexible scheduling of reconfigurable microgrid-based distribution networks considering demand response program. Energy, 196, Article 117024. https://doi.org/10.1016/j.energy.2020.117024





Akbari, R., Hedayatzadeh, R., Ziarati, K., & Hassanizadeh, B. (2012). A multi-objective artificial bee colony algorithm. Swarm and Evolutionary Computation, 2, 39-52. https://doi.org/10.1016/j.swevo.2011.08.001





Alomoush, M. I. (2019). Microgrid combined power-heat economic-emission dispatch considering stochastic renewable energy resources, power purchase and emission tax. Energy Conversion and Management, 200, Article 112090. https://doi.org/10.1016/j.enconman.2019.112090





Coello, C. A. C., & Cortes, N. C. (2005). Solving multiobjective optimization problems using an artificial immune system. Genetic Programming and Evolvable Machines, 6(2), 163-190. https://doi.org/10.1007/s10710-005-6164-x





Coello, C. A. C., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives with particle swarm optimization. IEEE Transactions on Evolutionary Computation, 8(3), 256-279. https://doi.org/10.1109/TEVC.2004.826067





Cui, Q., Ma, P., Huang, L., Shu, J., Luv, J., & Lu, L. (2020). Effect of device models on the multiobjective optimal operation of CCHP microgrids considering shiftable loads. Applied Energy, 275, Article 115369. https://doi.org/10.1016/j.apenergy.2020.115369





Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197. https://doi.org/10.1109/4235.996017





Dhiman, G., Singh, K. K., Slowik, A., Chang, V., Yildiz, A. R., Kaur, A., & Garg, M. (2021). EMoSOA: A new evolutionary multi-objective seagull optimization algorithm for global optimization. International Journal of Machine Learning and Cybernetics, 12(2), 571-596. https://doi.org/10.1007/s13042-020-01189-1





Dhiman, G., Singh, K. K., Soni, M., Nagar, A., Dehghani, M., Slowik, A., ... Cengiz, K. (2021). MOSOA: A new multi-objective seagull optimization algorithm. Expert Systems with Applications, 167, Article 114150. https://doi.org/10.1016/j.eswa.2020.114150





He, C., Tian, Y., Jin, Y., Zhang, X., & Pan, L. (2017). A radial space division based evolutionary algorithm for many-objective optimization. Applied Soft Computing, 61, 603-621. https://doi.org/10.1016/j.asoc.2017.08.024





Hu, W., & Yen, G. G. (2015). Adaptive multiobjective particle swarm optimization based on parallel cell coordinate system. IEEE Transactions on Evolutionary Computation, 19 (1), 1-18. https://doi.org/10.1109/TEVC.2013.2296151





Jiang, S., Ong, Y. S., Zhang, J., & Feng, L. (2014). Consistencies and contradictions of performance metrics in multiobjective optimization. IEEE Transactions on Cybernetics, 44(12), 2391-2404. https://doi.org/10.1109/TCYB.2014.2307319





Joyce, T., & Herrmann, J. M. (2018). A review of No Free Lunch theorems, and their implications for metaheuristic optimisation. In X.-S. Yang (Ed.), Nature-Inspired Algorithms and Applied Optimization (pp. 27-51). Cham: Springer International Publishing. https://doi.org/10.1007/978-3-319-67669-2_2.





Kahloul, S., Zouache, D., Brahmi, B., & Got, A. (2022). A multi-external archive-guided Henry Gas Solubility Optimization algorithm for solving multi-objective optimization problems. Engineering Applications of Artificial Intelligence, 109, Article 104588. https://doi.org/10.1016/j.engappai.2021.104588





Kathiroli, P., & Selvadurai, K. (2021). Energy efficient cluster head selection using improved Sparrow Search Algorithm in Wireless Sensor Networks. Journal of King Saud University - Computer and Information Sciences. https://doi.org/10.1016/j.jksuci.2021.08.031





Khalilpourazari, S., Naderi, B., & Khalilpourazary, S. (2020). Multi-objective stochastic fractal search: A powerful algorithm for solving complex multi-objective optimization problems. Soft Computing, 24(4), 3037-3066. https://doi.org/10.1007/s00500-019-04080-6





Li, B., Wang, H., Wang, X., Negnevitsky, M., & Li, C. (2022). Tri-stage optimal scheduling for an islanded microgrid based on a quantum adaptive sparrow search algorithm. Energy Conversion and Management, 261, Article 115639. https://doi.org/10.1016/j.enconman.2022.115639





Li, M., Yang, S., & Liu, X. (2014). Shift-based density estimation for pareto-based algorithms in many-objective optimization. IEEE Transactions on Evolutionary Computation, 18(3), 348-365. https://doi.org/10.1109/TEVC.2013.2262178





Liu, B., & Rodriguez, D. (2021). Renewable energy systems optimization by a new multi-objective optimization technique: A residential building. Journal of Building Engineering, 35, Article 102094. https://doi.org/10.1016/j.jjobe.2020.102094





Ma, L., Huang, M., Yang, S., Wang, R., & Wang, X. (2021). An adaptive localized decision variable analysis approach to large-scale multiobjective and many-objective optimization. IEEE Transactions on Cybernetics, 1-13. https://doi.org/10.1109/TCYB.2020.3041212





Ma, L., Li, N., Guo, Y., Wang, X., Yang, S., Huang, M., & Zhang, H. (2021). Learning to optimize: Reference vector reinforcement learning adaption to constrained many-objective optimization of industrial copper burdening system. IEEE Transactions on Cybernetics, 1-14. https://doi.org/10.1109/TCYB.2021.3086501





Mahmoodabadi, M. J., Taherkhorsandi, M., & Bagheri, A. (2014). Optimal robust sliding mode tracking control of a biped robot based on ingenious multi-objective PSO. Neurocomputing, 124, 194-209. https://doi.org/10.1016/j.neucom.2013.07.009





Mirjalili, S., Saremi, S., Mirjalili, S. M., & Coelho, L. D. S. (2016). Multi-objective grey wolf optimizer: A novel algorithm for multi-criterion optimization. Expert Systems with Applications, 47, 106-119. https://doi.org/10.1016/j.eswa.2015.10.039





Mirjalili, S. Z., Mirjalili, S., Saremi, S., Faris, H., & Aljarah, I. (2018). Grasshopper optimization algorithm for multi-objective optimization problems. Applied Intelligence, 48(4), 805-820. https://doi.org/10.1007/s10489-017-1019-8





Patel, V. K., & Savsani, V. J. (2016). A multi-objective improved teaching-learning based optimization algorithm (MO-ITLBO). Information Sciences, 357, 182-200. https://doi.org/10.1016/j.ins.2014.05.049





Peng, Y., & Ishibuchi, H. (2021). A diversity-enhanced subset selection framework for multi-modal multi-objective optimization. IEEE Transactions on Evolutionary Computation, 1-1. https://doi.org/10.1109/TEVC.2021.3117702





Pradhan, P. M., & Panda, G. (2012). Solving multiobjective problems using cat swarm optimization. Expert Systems with Applications, 39(3), 2956-2964. https://doi.org/10.1016/j.eswa.2011.08.157





Raquel, C. R., & Naval, P. C. (2005). An effective use of crowding distance in multiobjective particle swarm optimization. Paper presented at the In Genetic and evolutionary computation conference, Washington DC, USA. https://doi.org/10.1145/1068009.1068047





Ren, Z., Jiang, R., Yang, F., & Qiu, J. (2022). A multi-objective elitist feedback teaching-learning-based optimization algorithm and its application. Expert Systems with Applications, 188, Article 115972. https://doi.org/10.1016/j.eswa.2021.115972





Sadollah, A., Eskandar, H., & Kim, J. H. (2015). Water cycle algorithm for solving constrained multi-objective optimization problems. Applied Soft Computing, 27, 279-298. https://doi.org/10.1016/j.asoc.2014.10.042





Savsan, V., & Tawhid, M. A. (2017). Non-dominated sorting moth flame optimization (NS-MFO) for multi-objective problems. Engineering Applications of Artificial Intelligence, 63, 20-32. https://doi.org/10.1016/j.engappai.2017.04.018





Shannon, C. E. (1948). A mathematical theory of communication. The Bell System Technical Journal, 27(3), 379-423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x





Shuai, H., Fang, J., Ai, X., Wen, J., & He, H. (2019). Optimal real-time operation strategy for microgrid: An ADP-based stochastic nonlinear optimization approach. IEEE Transactions on Sustainable Energy, 10(2), 931-942. https://doi.org/10.1109/TSTE.2018.2855039





Tawhid, M. A., & Savsani, V. (2019). Multi-objective sine-cosine algorithm (MO-SCA) for multi-objective engineering design problems. Neural Computing and Applications, 31 (2), 915–929. https://doi.org/10.1007/s00521-017-3049-x





Tian, Y., Cheng, R., Zhang, X., Li, M., & Jin, Y. (2019a). Diversity assessment of multi-objective evolutionary algorithms: Performance metric and benchmark problems. IEEE Computational Intelligence Magazine, 14(3), 61-74. https://doi.org/10.1109/MCI.2019.2919398





Tian, Y., Cheng, R., Zhang, X., Li, M., & Jin, Y. (2019b). Diversity Assessment of Multi-Objective Evolutionary Algorithms: Performance Metric and Benchmark Problems [Research Frontier]. IEEE Computational Intelligence Magazine, 14(3), 61-74. https://doi.org/10.1109/MCI.2019.2919398





Wang, L., Ren, Y., Qiu, Q., & Qiu, F. (2021). Survey on performance indicators for multi-objective evolutionary algorithms. Chinese Journal of Computers, 44(8), 30. https://doi.org/10.1189/SP.j.1016.2021.01590





Xu, F., Liu, J., Lin, S., Dai, Q., & Li, C. (2018). A multi-objective optimization model of hybrid energy storage system for non-grid-connected wind power: A case study in China. Energy, 163, 585-603. https://doi.org/10.1016/j.energy.2018.08.152





Xue, J., & Shen, B. (2020). A novel swarm intelligence optimization approach: Sparrow search algorithm. Systems Science & Control Engineering, 8(1), 22-34. https://doi.org/10.1080/21642583.2019.1708830





Yang, S., Li, M., Liu, X., & Zheng, J. (2013). A grid-based evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 17(5), 721-736. https://doi.org/10.1109/TEVC.2012.2227145





Yao, X., Li, W., Pan, X., & Wang, R. (2022). Multimodal multi-objective evolutionary algorithm for multiple path planning. Computers & Industrial Engineering, 169, Article 108145. https://doi.org/10.1016/j.cie.2022.108145





Yin, L., & Gao, Q. (2021). Multi-objective proportional-integral-derivative optimization algorithm for parameters optimization of double-fed induction generator-based wind





turbines. Applied Soft Computing, 110, Article 107673. https://doi.org/10.1016/j.asoc.2021.107673





Zhang, C., & Ding, S. (2021). A stochastic configuration network based on chaotic sparrow search algorithm. Knowledge-Based Systems, 220, Article 106924. https://doi.org/10.1016/j.knosys.2021.106924





Zhang, X., Tian, Y., Cheng, R., & Jin, Y. (2018). A decision variable clustering-based evolutionary algorithm for large-scale many-objective optimization. IEEE





Transactions on Evolutionary Computation, 22(1), 97-112. https://doi.org/10.1109/TEVC.2016.2600642





Zhu, Y., & Yousefi, N. (2021). Optimal parameter identification of PEMFC stacks using adaptive sparrow search algorithm. International Journal of Hydrogen Energy, 46(14), 9541-9552. https://doi.org/10.1016/j.ijhydene.2020.12.107





Zitzler, E., & Thiele, L. (1999). Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach. IEEE Transactions on Evolutionary Computation, 3(4), 257-271. https://doi.org/10.1109/4235.797969

