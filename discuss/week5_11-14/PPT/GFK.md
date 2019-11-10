# [Geodesic Flow Kernel (GFK)](https://www.cs.utexas.edu/users/grauman/papers/subspace-cvpr2012.pdf)

设$G(d,D)$是数据$D$的原始空间在$d$维子空间的全部集合，那么任意一个数据D的d维子空间就是G上的一个点，两点之间的测地线可以在两个子空间之间构成一条路径，从一个点通过测地线走到另一个点可以看成从一个空间迁移到另一个空间

![](./assets/MEDA-GFK.png)

$P_s,P_t \in R^{D\times d}$ 代表源域和目标域的子空间的两组基

$R_s \in R^{D\times (D-d)}$ 代表$P_s$的补，也就是$R^T_S P_S=0$

我们定义测地线映射函数:

$\Phi(t)=P_S U_1 \Gamma(t) - R_S U_2 \Sigma(t)$ 

----
$\Phi:t\in[0,1]\rightarrow\Phi(t)\in G(d,D)$ ，并且设$P_s=\Phi(0)$, $P_t=\Phi(1)$

<br/>

$P^T_S P_T=U_1 \Gamma V^T,R^T_SP_T=-U_2\Sigma V^T$

$\Phi$中的参数是上式的奇异值分解的结果

Γ and Σ 是对角矩阵，他们的对角元素分别是cosθ和sinθ（principal angles）