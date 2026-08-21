# 端到端 SLO 约束的 Token 屏障多层专家批释放调度

## 面向固定部署 AFD MoE 解码的问题定义与复杂度分析

本文研究 Attention–FFN Disaggregation（AFD）MoE 解码系统中的专家 batch 调度。模型部署、专家路由和非 FFN 执行均已确定，调度器只控制 FFN GPU 上各 block–expert 队列的释放时间与执行顺序。该问题称为**端到端 SLO 约束的 token 屏障多层专家批释放调度**（Token-Barrier Multi-layer Expert Batch-Release Scheduling under End-to-End SLOs，简称 **TB-MEBR**）。

---

## 1. 背景与研究动机

### 1.1 AFD 改变了专家 batch 的边界

MoE Transformer block 包含 Attention、router 和稀疏 FFN。Attention 依赖 KV cache；稀疏 FFN 保存专家权重，并只执行 router 为当前 token 选中的少量专家。AFD 将两类模块部署在不同设备组中，使 Attention 与 FFN 可以独立扩展。

[MegaScale-Infer](https://arxiv.org/abs/2504.02263) 展示了 Attention–FFN 分离、多对多 activation 传输和 microbatch 流水。[FinDEP](https://arxiv.org/abs/2512.21487) 将计算与通信切分为细粒度任务并优化重叠顺序。这些系统提高了流水利用率，但 AFD 还产生了一个独立的专家侧问题：来自不同 Attention worker、请求和 microbatch 的 token 会异步到达同一 block–expert，FFN 侧可以跨原 global batch 重新合批。

### 1.2 Global batch 增大不等于 expert batch 增大

router 按 token 内容选择专家。增大 Attention 侧 global batch 只能增加 token 总数，不能保证每个专家得到足够任务。热门专家可能持续排队，冷门专家仍然形成小 batch；若执行沿用 global batch 屏障，其他分支还要等待最慢专家。

因此，调度对象应是 block–expert 队列。每条队列独立决定何时释放；映射到同一 GPU 的不同专家和不同层还要竞争执行顺序。等待可以扩大 batch 并减少单位 token 的执行成本，但会增加当前 token 的排队时间。

### 1.3 SLO 使多层调度不能退化为当前时刻打分

[AMoE](https://arxiv.org/abs/2505.08944) 通过异步专家并行、token 依赖跟踪、μ-queuing 和 adaptive re-batching 缓解 barrier 与小 batch 问题，其主要目标是吞吐、利用率和成本效率。在端到端 deadline 下，只依据当前队列长度或当前 batch 收益还不够，因为当前层等待会改变后续层任务的到达时间和剩余预算。

同一个 token 在一层内可能路由到多个专家。只有这些专家全部完成并汇合后，该 token 才能进入下一层。后层任务的 ready time 因而由前层调度内生决定。完整问题同时包含跨 token 合批、token 内 Top-K 汇合、层间前序依赖和共享 GPU 竞争。

---

## 2. 问题定义

给定一个固定部署的 AFD MoE 模型和一个有限的已接纳 token-step 集合。每个 token 的外部到达时间、端到端 deadline 和每层 Top-K 路由结果已知；每个 block–expert 的 owner GPU、最大 batch size 和 batch 执行时间已知；Attention、router、dispatch、combine 和层间通信由固定延迟表示。

每个 token 在每个 MoE block 产生若干专家任务。只有属于同一 block–expert 的任务可以合并为一个 batch。batch 不可抢占，成员在 batch 结束时同时完成。同一 token 必须等待当前层全部已选专家完成，才能产生下一层任务。因此，第一层以外的任务到达时间不是独立输入，而是前层调度的结果。

每条 block–expert 队列采用 capped FIFO full-drain。释放队列时，当前已经 ready 且尚未执行的 FIFO 队首任务全部进入 batch，但成员数不超过容量；启动后到达的任务进入后续 batch。调度器决定队列释放时间以及同一 GPU 上不同 batch 的执行顺序，不任意选择 batch 成员。

目标是在所有 token 满足端到端 deadline 的条件下，最小化最大的端到端 token 延迟。输入 token 已经完成 admission；若不存在满足全部 deadline 的调度，则实例不可行。



**定义 1（TB-MEBR）。**

**输入：** 固定部署的 AFD MoE 模型，以及一组路由结果、到达时间和端到端 deadline 已知的 token。专家放置、GPU 数量、非 FFN 延迟和不同 batch size 的执行时间均已知。

**决策：** 决定每条 block–expert 队列何时释放，以及共享 GPU 上不同 expert batches 的执行顺序。batch 成员由 capped FIFO full-drain 规则确定。

**约束：** batch 不可抢占，同一 GPU 上的 batches 不得重叠。每个 token 只有在当前层全部已选专家完成后才能进入下一层，并且必须在端到端 deadline 前完成。

**目标：** 最小化所有 token 中最大的端到端延迟。

**输出：** 满足上述约束的 expert batch 调度；若不存在，则该实例不可行。

---

## 3. 系统模型

模型层、专家 placement、专家副本绑定、FFN GPU 数量、Attention 调度、通信方式和路由结果均固定。若一个逻辑专家存在多个副本，token 使用的物理副本由输入给定。每张 FFN GPU 被建模为一个不可抢占执行通道。本文不联合优化 placement、replica scaling、Attention batch、prefill、通信路径或 GPU 内部资源分配。

### 3.1 固定输入

记 token 集合和按执行顺序排列的 MoE block 集合为

$$
\mathcal I,
\qquad
\mathcal L=\{1,\ldots,L\}.
$$

token 在一个 block 中必须完成的物理专家集合记为

$$
\mathcal R_{i,l}.
$$

block–expert 的固定 owner GPU、最大 batch size 和 size-dependent 执行时间分别记为

$$
g(l,e),
\qquad
B_{l,e},
\qquad
\tau_{l,e}(b).
$$

token 的外部到达时间和绝对 deadline 分别为

$$
a_i,
\qquad
d_i.
$$

从外部到达到第一层、相邻两层之间以及末层到最终输出之间的固定非 FFN 延迟统一记为

$$
\delta_{i,l},
\qquad l=0,\ldots,L.
$$

其中第零段位于第一层之前，最后一段位于末层之后。

### 3.2 Expert batch

一个专家任务由 token、block 和物理专家确定：

$$
(i,l,e),
\qquad e\in\mathcal R_{i,l}.
$$

对任一 batch，记其 block、专家、成员集合和启动时间为

$$
l_\beta,
\qquad
e_\beta,
\qquad
M_\beta,
\qquad
s_\beta.
$$

成员集合由 capped FIFO full-drain 规则确定：它包含启动时队列中已经 ready、尚未执行的前若干任务，成员数等于当前可用任务数与容量中的较小值。由此得到

$$
1\le\lvert M_\beta\rvert\le B_{l_\beta,e_\beta}.
$$

每个专家任务恰好属于一个 batch，且 batch 的所有成员具有相同的 block 和物理专家。batch 的完成时间为

$$
c_\beta
=
s_\beta
+
\tau_{l_\beta,e_\beta}
\!\left(\lvert M_\beta\rvert\right).
$$

若一个专家任务属于该 batch，它必须在 batch 启动前 ready，并在 batch 结束时完成：

$$
(i,l_\beta,e_\beta)\in M_\beta
\;\Longrightarrow\;
r_{i,l_\beta}\le s_\beta,
$$

$$
(i,l_\beta,e_\beta)\in M_\beta
\;\Longrightarrow\;
C_{i,l_\beta,e_\beta}=c_\beta.
$$

映射到同一 GPU 的任意两个 batch 不能重叠：

$$
g(l_\beta,e_\beta)=g(l_{\beta'},e_{\beta'})
\;\Longrightarrow\;
c_\beta\le s_{\beta'}
\;\lor\;
c_{\beta'}\le s_\beta.
$$

### 3.3 多层 token 屏障

token 经过第一段固定非 FFN 延迟后产生第一层专家任务：

$$
r_{i,1}=a_i+\delta_{i,0}.
$$

token 在一个 block 的完成时刻是其已选专家中的最晚完成时间：

$$
J_{i,l}
=
\max_{e\in\mathcal R_{i,l}}
C_{i,l,e}.
$$

该最大值只覆盖当前 token 的路由集合，不等待该层其他专家，也不等待同一 global batch 中的其他 token。

对每个非末层 block，下一层 ready time 由当前层汇合时间严格确定：

$$
r_{i,l+1}
=
J_{i,l}+\delta_{i,l},
\qquad l=1,\ldots,L-1.
$$

调度器不能人为推迟 ready time；如果希望等待未来同专家任务，只能推迟相应 batch 的启动。

末层汇合后，token 的最终完成时间为

$$
F_i=J_{i,L}+\delta_{i,L}.
$$

每个 token 必须满足硬 deadline：

$$
F_i\le d_i,
\qquad \forall i\in\mathcal I.
$$

专家任务没有预先指定的逐层 deadline。各层共享同一个端到端预算，后层剩余时间由前层实际完成时间决定。

---

## 4. 优化问题

token 的端到端延迟是最终完成时间与外部到达时间之差。TB-MEBR 在全部合法调度中求解

$$
\min_{\text{合法调度}}
\;
\max_{i\in\mathcal I}
\left(F_i-a_i\right),
$$

并满足

$$
F_i\le d_i,
\qquad \forall i\in\mathcal I.
$$

deadline 决定可行域，目标函数比较可行调度的最坏端到端延迟。该目标不会用某个 token 的 deadline 失败换取其他 token 的低延迟。

对应的决策问题为：给定延迟上界

$$
\Theta,
$$

是否存在一个合法调度，使全部 token 满足 deadline，并且

$$
\max_{i\in\mathcal I}
\left(F_i-a_i\right)
\le\Theta.
$$

假定所有时间参数均为多项式位数的整数或有理数。对有理数统一缩放即可得到整数时间实例；调度使用同一离散时钟，batch 只能在整数 tick 启动。一个证书列出每个非空 batch 的成员、启动时间和 GPU 顺序；验证者可以依次检查 FIFO full-drain、batch 执行、GPU 互斥、层内汇合、层间 ready time、最终完成、deadline 和延迟上界。因此，决策问题属于 NP。

---

## 5. 汇合隐藏等待

Top-K 汇合使一部分 batch 等待可以被兄弟分支遮蔽。考虑同一个 token 在同一 block 中的一个专家分支。若该分支立即执行时的完成时间为

$$
c^0,
$$

其他已选专家的最晚完成时间为

$$
h,
$$

则当前分支在不改变本层汇合时间的前提下，最多可以增加

$$
\sigma^{\mathrm{join}}
=
[h-c^0]^+,
\qquad
[x]^+=\max\{x,0\}.
$$

具体地，若额外等待满足

$$
0\le w\le\sigma^{\mathrm{join}},
$$

则

$$
\max\{c^0+w,h\}
=
\max\{c^0,h\}.
$$

因此，这部分等待可以用于聚合未来同专家任务，而不会直接推迟下一层。超过该范围的等待才会增加当前层汇合时间。

该结论以兄弟分支完成时间不变为条件。兄弟分支已经完成，或者已经启动且不可抢占时，可以精确计算这一等待空间；兄弟分支尚未排程时，只能使用预测值或保守下界。该性质只描述本层汇合，不单独保证最终 deadline。

---

## 6. NP-completeness

### 6.1 来源问题

规约来源为 Job-shop Scheduling Problem（JSSP）的 makespan 决策版本。源实例包含若干 jobs 和 machines。每个 job 包含一条有序 operation 链；每道 operation 具有正处理时间并指定一台 machine。同一 machine 一次只能执行一道不可抢占 operation。给定上界，问题询问是否存在 makespan 不超过该上界的调度。该问题是 NP-complete [Garey、Johnson 和 Sethi，1976](https://doi.org/10.1287/moor.1.2.117)。

使用每个 job 具有相同 operation 数量的标准形式。令 job 数、每个 job 的 operation 数和 makespan 上界分别为

$$
n,
\qquad
m,
\qquad
D.
$$

### 6.2 规约构造

固定任意正整数常数 Top-K：

$$
\kappa\ge1.
$$

对每个 JSSP job 创建一个 token，对 job 的第若干道 operation 创建对应 token 的同序号 block，因此

$$
\lvert\mathcal I\rvert=n,
\qquad
L=m.
$$

每台 JSSP machine 映射为一张主 FFN GPU。每道 operation 创建一个唯一主 block–expert；该队列只有一个任务，容量为 1，singleton batch 执行时间等于来源 operation 的正处理时间，owner GPU 对应来源 machine。

当 Top-K 大于 1 时，为每个主任务增加辅助专家任务，使每层路由分支总数等于固定 Top-K。每个辅助专家位于独占 GPU 上，容量为 1，处理时间与同层主任务相同且严格为正。

所有外部到达时间和非 FFN 延迟设为零：

$$
a_i=0,
\qquad
\delta_{i,l}=0,
\qquad l=0,\ldots,L.
$$

所有 token 使用共同 deadline，决策问题的延迟上界也等于来源 makespan 上界：

$$
d_i=D,
\qquad
\Theta=D.
$$

每条 block–expert 队列只有一个任务且容量为 1，因此 full-drain 自动成立，也不存在有效 batching。辅助任务使用正处理时间，构造不依赖零时间任务。

### 6.3 正向证明

假设来源 JSSP 存在 makespan 不超过上界的调度。对每道 operation，在相同启动时间执行对应主 singleton batch。来源 job precedence 保证后一层主任务不会早于前一层主任务完成；来源 machine 互斥保证同一主 GPU 上的 batch 不重叠。

对每个辅助任务，在其主任务的同一启动时间、独占 GPU 上执行。辅助任务与主任务同时完成，所以本层汇合时间等于主任务完成时间，下一层 ready time 与来源 job 下一道 operation 的可开始时间一致。

全部来源 jobs 在上界前完成。token 外部到达时间为零，因此全部 token 的端到端延迟不超过共同上界，并满足共同 deadline。构造的 TB-MEBR 实例回答 YES。

### 6.4 反向证明

假设构造的 TB-MEBR 实例存在满足共同 deadline 和延迟上界的调度。删除全部辅助 batch，只保留主 singleton batch，并把它们映射回来源 operations。

同一 token 的后一层主任务只能在前一层汇合后启动，而汇合不早于前一层主任务完成，因此来源 job precedence 成立。主 batch 在对应 machine GPU 上互斥且不可抢占，因此来源 machine 约束成立。辅助任务即使被推迟，也只会推迟汇合，不会破坏主路径前序。

每个 token 的外部到达时间为零，最终完成时间不超过共同上界，所以最后一道主 operation 也在该上界前完成。提取出的 JSSP 调度 makespan 不超过来源上界，源实例回答 YES。

### 6.5 构造规模与结论

构造包含

$$
nm\kappa
$$

个专家任务。主 GPU 数等于来源 machine 数，辅助 GPU 数至多为

$$
nm(\kappa-1).
$$

由于 Top-K 是固定常数，构造规模为来源实例规模的多项式，所有处理时间只被复制。

**定理 1。** 对任意固定正整数 Top-K，TB-MEBR 的决策版本是 NP-complete。该结论在固定 owner、共同 deadline、零非 FFN 延迟、正处理时间、每条队列只有一个任务且容量为 1 的受限条件下成立。

**证明。** 第 4 节说明 TB-MEBR 决策问题属于 NP。第 6.2 节给出多项式构造，第 6.3 节和第 6.4 节证明来源 JSSP 与构造实例当且仅当可行。因此，决策问题是 NP-complete，最坏端到端延迟优化问题是 NP-hard。证毕。

该规约保留多层前序依赖、固定 Top-K、正专家处理时间和共享 GPU 竞争，但关闭有效 batching。因此，多层前序与共享 GPU 竞争已经足以产生 NP-hardness；动态 batching 和非退化汇合存在于一般问题中，但不是该证明成立所必需的条件。该结论针对层数、专家数或 GPU 数可以随输入增长的一般问题族。

