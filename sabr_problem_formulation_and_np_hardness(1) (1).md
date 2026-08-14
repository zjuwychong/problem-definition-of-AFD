# 多层 AFD 中 SLO-Aware Expert Batch Scheduling：背景、问题定义与 NP-Hardness 分析

本文研究 Attention–FFN disaggregation（AFD）架构下的多层 Mixture-of-Experts（MoE）解码调度问题。研究对象不是某一个孤立专家层，而是一个 token 从第一层进入模型、在每层经过 Attention、Top-K 专家分叉与结果汇合、最终完成全部模型层的完整执行过程。

全文依次说明问题背景、完整的多层离线问题定义，以及该问题的 NP-hardness。在线算法不在本文讨论范围内。

## 1. 问题背景

### 1.1 MoE 推理中的 Attention 和 FFN

一个 MoE Transformer block 通常包含 Attention 和稀疏 FFN 两个主要模块。Attention 保存并读取请求的 KV cache，计算当前 token 与历史上下文之间的关系；稀疏 FFN 通过 router 为每个 token 选择少量专家，只执行被选中的专家网络。

两类模块的资源特征不同。随着上下文长度增加，Attention 的 KV cache 容量和读取开销持续增加，因此通常受到显存容量与显存带宽限制。FFN 则需要保存大量专家权重。在解码阶段，如果一个专家每次只处理少量 token，GPU 仍然需要读取该专家的大量权重，固定启动与权重读取成本难以在多个 token 之间分摊。

传统部署将 Attention 和 FFN 放在同一组 GPU 上，并使用相同的并行配置。这会使两类算子竞争资源，也限制了系统根据各自负载独立扩展设备数量和并行方式。

### 1.2 Attention–FFN 分离

AFD 将 Attention 和 FFN 部署在不同设备池中：

- Attention worker 保存 KV cache，执行 Attention 和 router；
- FFN worker 保存专家权重，执行专家计算；
- 两个设备池之间传输路由后的 activation 和专家输出。

这种架构允许 Attention 和 FFN 独立扩展，也允许多个 Attention worker 共享同一个 FFN 专家池。MegaScale-Infer 是代表性工作之一，它将 Attention 与 FFN 分离，并通过 microbatch 流水和多对多通信提高 MoE 推理吞吐。[MegaScale-Infer](https://arxiv.org/abs/2504.02263)

AFD 还改变了专家 batch 的形成方式。在传统同步执行中，一个 global batch 经过 router 后被拆分到不同专家，expert batch 的边界主要由这个 global batch 决定。在 AFD 中，多个 Attention worker 和多个流水阶段可能在不同时间向同一个专家发送 token。同一个 block–expert 因而接收到一个持续的异步 token 流，而不是一次封闭的 global batch。

例如，某个专家先收到 4 个 token，0.2 ms 后又收到 6 个 token。专家侧可以立即执行前 4 个 token，也可以等待 0.2 ms，将两次到达合成一个包含 10 个 token 的 batch。AMoE 使用逐 block–expert 微队列聚合异步到达的 token，并动态形成专家 batch。[AMoE](https://arxiv.org/abs/2505.08944)

### 1.3 Global batch 增大不等于所有 expert batch 都得到改善

增大 Attention 侧的 global batch，不能保证每个专家都获得足够大的 expert batch。router 会按照 token 内容将 global batch 不均匀地拆分到不同专家，因此热门专家和冷门专家的负载增长并不一致。

例如，一个包含 64 个 token 的 global batch 可能给热门专家分配 32 个 token，却只给冷门专家分配 2 个 token。把 global batch 增加到 128 后，热门专家可能收到 64 个 token，冷门专家仍然只收到 4 个 token。结果是：冷门专家仍然难以分摊权重读取成本，热门专家的执行时间却进一步增加。

如果系统以 global batch 为同步边界，其他专家还需要等待最慢的热门专家完成。因此，global batch 增大可能同时造成：

- 冷门专家的 batch 仍然偏小；
- 热门专家执行时间增长；
- 空闲专家等待热门专家；
- global batch 的端到端延迟增加。

所以，专家调度的基本对象不能只是 global batch。系统需要为每个具体 block 中的具体专家维护独立队列，并分别决定这些队列何时释放 batch。

### 1.4 Batching 收益受到端到端 SLO 限制

如果只优化 GPU 吞吐，专家可以尽量等待未来的同专家 token，从而形成更大的 batch。但在线解码通常具有 time per output token（TPOT）或 time between tokens（TBT）目标。等待虽然可能减少 GPU 执行时间，却会消耗 token 的剩余时间。

于是产生基本冲突：

- 立即启动能够减少当前 token 的等待，但可能反复执行低效的小 batch；
- 继续等待能够扩大 batch，但可能使当前 token 或其他受影响 token 违反 SLO。

热门或冷门只描述到达率，不能直接表示 SLO 紧迫程度。一个冷门专家队列即使只有一个 token，也可能因为该 token 已经接近最终 deadline 而必须立即执行。相反，一个热门专家队列中的某些 token 可能仍有较多剩余时间。

因此，调度目标不是无条件增大 batch，而是：

> 在端到端 SLO 可行的范围内，利用每个 block–expert 的等待空间聚合同专家 token。

### 1.5 多层问题处理

一个输出 token 必须依次经过模型的全部 Transformer block。在每个 MoE block 中，token 先经过 Attention 和 router，再被分发给该层选中的多个专家。只有这些专家全部完成并聚合结果后，token 才能进入下一层。

因此，只有 token 进入第一层的时间可以作为外部到达。后续层专家子任务的到达时间由前面所有层的调度共同产生。当前层等待多久、形成什么 batch、占用哪张 GPU，都会改变后续层的到达时间、batch 组合和剩余 SLO。

这个问题同时包含三类耦合：

| 耦合方向 | 含义 |
|---|---|
| 层间依赖 | 同一个 token 必须在当前层全部专家完成后才能进入下一层 |
| 同专家聚合 | 不同 token 在相同 block–expert 上的子任务可以形成 batch |
| 跨层资源竞争 | 同一张 FFN GPU 上来自不同模型层的专家 batch 互相竞争 |

所以，完整调度不能被分解成若干独立的逐层优化问题。某一层减少的 GPU 时间不一定转化为端到端收益；某一层增加的短暂等待也可能使 token 错过下游 batch，从而在后续层产生更大的延迟。

## 2. 完整多层执行模型

### 2.1 系统范围

我们考虑一个部署已经固定的 AFD MoE 解码系统。系统已经确定：

- Attention worker 和 FFN GPU 的数量；
- Transformer block 数量；
- 每个 block 中的专家集合；
- 每个 block–expert 的 GPU 放置；
- 专家副本映射；
- Attention 侧调度方式；
- 通信、聚合和专家执行方式。

本文不联合优化专家 placement、设备扩缩容、prefill 调度、admission control 和网络拓扑。它们被视为上层配置或固定输入。本文的调度变量只控制 FFN GPU 上跨层 block–expert batch 的选择和启动时间。

Attention 侧仍然是完整执行路径的一部分。若 Attention 侧存在排队，其延迟按照已经确定的 Attention 调度规则由 token 到达状态计算；若采用简化模型，则使用 profiling 得到的层间 Attention 与通信时间。无论采用哪一种方式，FFN 调度产生的完成时间都会继续影响后续 Attention 和下一层 FFN 到达。

### 2.2 Token-step

一个请求为了产生下一个输出 token，需要让当前 token representation 经过模型的全部 block。本文把这次完整执行称为一个 token-step，后文简称 token。

一个 token 只有在完成最后一个 block 的全部专家分支、结果聚合和最终输出处理后，才算完成。中间某一个专家分支完成，不代表该 token 已经完成。

如果进一步建模同一个请求的多个连续输出 token，还需要加入自回归依赖：前一个 token-step 完成并产生输出后，下一个 token-step 才能开始。当前离线问题可以在一个有限观察窗口内把这些依赖作为额外前序边加入执行图；它们不会改变下面的多层专家调度定义。

### 2.3 每层的 fork-join 结构

对 token \(i\) 和 block \(l\)，router 选择一个专家集合，记为 \(R_{i,l}\)。如果模型还要求 shared expert，则把 shared expert 也加入这个必须完成的集合。集合中每个专家产生一个独立专家子任务。

例如，Top-2 路由选择专家 \(E_1\) 和 \(E_2\)，则当前层产生两个子任务：

```text
token i 在 block l 的 E₁ 子任务
token i 在 block l 的 E₂ 子任务
```

两个子任务可以进入不同队列、在不同 GPU 上执行，也可以分别和其他 token 形成不同 batch。它们之间不要求同时开始或同时结束。

但是，token 只有在集合 \(R_{i,l}\) 中所有专家子任务都完成后，才能完成 block \(l\) 的汇合。若 \(C_{i,l,e}\) 表示专家 \(e\) 完成 token \(i\) 当前子任务的时间，则该层专家部分的完成时间由最晚完成的已选专家决定：

$$
J_{i,l}=\max_{e\in R_{i,l}} C_{i,l,e}.
$$

这里，\(J_{i,l}\) 是 token \(i\) 在 block \(l\) 的专家汇合时间。下一层还需要等待结果聚合、通信和下一层 Attention 与路由，因此下一层专家子任务的就绪时间是在 \(J_{i,l}\) 的基础上继续计算得到的。

在 Top-1 模型中，集合 \(R_{i,l}\) 只有一个专家，上述 fork-join 自然退化成单分支前序依赖。

### 2.4 Block–expert 队列

每个 block–expert 维护一个逻辑就绪队列，记为 \(Q_{l,e}\)，其中 \(l\) 表示 block，\(e\) 表示该 block 中的专家。

block 2 的 Expert 3 和 block 10 的 Expert 3 是两个不同的 block–expert。它们使用不同权重，因此不能放入同一个普通专家 batch。即使二者位于同一张 GPU 上，也只能作为两个不同队列分别执行。

所有 block–expert 队列在逻辑上持续存在，但后层队列在系统开始时通常为空。一个专家子任务只有在以下条件全部满足后，才进入对应就绪队列：

1. 该 token 的上一层全部专家分支已经完成；
2. 上一层专家结果已经完成聚合；
3. 必要的通信、Attention 和 router 已经完成；
4. 当前层路由已经确定该专家被选中。

因此，后层队列不是不存在，而是由上游 fork-join 完成事件逐步填充。

### 2.5 内生到达时间

对第一层，token 的初始到达时间由请求和 Attention 侧给出。对任何后续层，专家子任务的就绪时间不是独立输入，而是上游调度结果。

例如，token 到达 block 10 的时间取决于：

- block 1 至 block 9 的 batch 分别等待了多久；
- 这些 batch 在共享 GPU 上的执行顺序；
- 每层 Top-K 中最后完成的是哪个专家；
- 每次结果聚合与设备间通信时间；
- 中间 Attention worker 的排队和执行时间。

所以，即使离线调度器已经知道每个 token 的全部未来路由，也不能把后续 block–expert 的实际到达时间预先固定。离线调度器知道执行图，但图中后续节点何时就绪，仍然由它选择的完整调度决定。

### 2.6 FFN GPU 资源竞争

一张 FFN GPU 可以放置来自多个 block 的多个专家。只要这些专家共享同一个不可并行的执行资源，它们的 batch 就不能重叠运行。

当 GPU 空闲时，它可能同时面对以下就绪队列：

- block 2 的某个专家队列；
- block 7 的某个专家队列；
- block 15 的某个专家队列。

因此，GPU 调度不是逐层进行的。调度器必须在该 GPU 上所有已就绪的跨层 block–expert 队列之间选择下一次执行。

### 2.7 专家 batch 执行时间

通过 profiling 可以获得每个 block–expert 在不同 batch size 下的执行时间。记 \(T_{l,e}(b)\) 为 block \(l\) 的专家 \(e\) 执行一个大小为 \(b\) 的 batch 所需的时间。

这里：

- \(l\) 是 block 编号；
- \(e\) 是专家编号；
- \(b\) 是 batch 中的专家子任务数量；
- \(T_{l,e}(b)\) 是对应的非抢占 GPU 执行时间。

一般情况下，\(T_{l,e}(b)\) 不是 token 数量的简单线性函数。小 batch 受到 kernel 启动、权重读取和固定通信成本影响；增大 batch 后，单位 token 成本下降；达到设备饱和区间后，继续增加 batch 的收益减弱。

完整问题允许直接使用 profiling 得到的执行时间表或单调函数。

### 2.8 Full-drain batch release

本文采用 full-drain 语义。某条 block–expert 队列在时间 \(t\) 被选择时，时间 \(t\) 之前已经进入该队列且尚未执行的全部子任务组成当前 batch。batch 启动后新到达的子任务不能加入正在执行的 batch，而是留在队列中等待下一次释放。

因此，full-drain 消除了“从当前就绪 token 中选择哪些成员”的独立决策。调度器在一次事件中统一决定：

- 当前 GPU 选择哪个 block–expert 队列；
- 该队列在什么时间释放；
- 由此形成的 batch size；
- GPU 接下来执行哪个 batch。

调度器也可以暂时不执行某个已经非空的队列。这种等待可能表现为 GPU 主动空闲，也可能表现为先执行同一 GPU 上的另一个 block–expert batch。

### 2.9 两类等待

完整模型需要区分两类等待。

资源等待表示专家子任务已经就绪，但对应 GPU 正在执行其他 batch，因此不能启动。

主动合批等待表示专家子任务已经就绪，而且 GPU 有能力执行该队列，但调度器为了等待未来同 block–expert 子任务而暂不释放。SABR 直接控制的是第二类等待，同时也必须考虑第一类等待带来的下游影响。

### 2.10 End-to-end deadline

每个 token \(i\) 具有最终输出 deadline，记为 \(D_i\)。这个 deadline 对应 TPOT 或 TBT 目标。

token 是否满足 SLO，只根据其完成全部模型层后的最终输出时间判断。不能分别判断每个专家子任务是否“按时”，也不能只看某一层的完成时间。

在 Top-K 路由中，一个较早完成的专家分支可能等待同 token 的其他分支。只要它没有成为该层最后完成的分支，它增加的一部分等待可能不会推迟下一层。这部分可利用时间来自层内汇合关系。但是，一旦该分支晚于其他分支，它就会成为新的汇合瓶颈，并直接推迟所有后续层。

完整离线模型不预先平均分配逐层 deadline。如何在不同层之间使用端到端 slack，是联合调度结果的一部分。任何逐层预算都应当是后续算法根据完整问题构造的近似，而不是原问题的固定输入。

## 3. 多层离线优化问题

### 3.1 离线已知信息

在一个有限观察窗口内，离线调度器知道：

- 全部 token 的第一层到达时间；
- 每个 token 在每个 block 的 Top-K 专家集合；
- block–expert 到 FFN GPU 的放置；
- 所有 batch 执行时间；
- Attention、通信和聚合时间模型；
- 每个 token 的最终输出 deadline。

离线调度器知道未来路由，但后层实际就绪时间仍然由调度产生。

### 3.2 调度输出

离线调度器为每张 FFN GPU 生成一个按时间排列的 batch 执行序列。每个执行项包含：

- 启动时间；
- 被执行的 block–expert 队列；
- 根据 full-drain 规则进入 batch 的专家子任务；
- batch size；
- batch 完成时间。

这些 GPU 执行序列共同决定每个专家分支的完成时间、每层 fork-join 时间、后续层到达时间和最终 token 完成时间。

### 3.3 合法调度约束

一个合法调度必须满足以下约束：

1. **就绪约束：** 专家子任务进入对应 block–expert 队列之前不能执行。
2. **兼容性约束：** 一个 batch 只能包含相同 block–expert 的子任务。
3. **Full-drain 约束：** 队列释放时，所有已经就绪且尚未执行的同队列子任务都进入当前 batch。
4. **设备互斥约束：** 同一 FFN GPU 上互相冲突的两个 batch 不能重叠。
5. **不可抢占约束：** 一个专家 batch 启动后必须连续执行到完成。
6. **层内汇合约束：** token 必须等待当前 block 的全部已选专家分支完成。
7. **层间前序约束：** token 只有在当前 block 汇合及层间处理完成后才能进入下一 block。
8. **完成约束：** token 的最终完成时间由最后一个 block 的汇合及最终输出处理决定。

### 3.4 优化目标

主目标是最大化满足最终输出 deadline 的 token 数量。对于 token \(i\)，只有其最终完成时间不晚于 \(D_i\) 时，它才计入 SLO goodput。

在固定观察窗口内，可以使用词典序的两级目标：

1. 最大化按时完成的 token 数量；
2. 在按时完成数量相同的调度中，最小化所有 FFN GPU 的总 busy time。

第二目标鼓励调度器减少重复的小 batch，让更多同专家子任务分摊固定执行成本。使用词典序可以避免用 batching 收益抵消 SLO 违约。

如果系统只接纳能够保证完成的 token，也可以把所有已接纳 token 的 deadline 作为硬约束，然后最小化 FFN GPU 总 busy time。过载场景下可能不存在让全部 token 按时完成的调度，因此最大化 SLO goodput 是更一般的表述。

### 3.5 决策版本

为了分析计算复杂度，定义以下决策问题：

> 给定一个完整多层 AFD 实例、每个 token 的最终 deadline，以及目标数量 \(K\)，是否存在一个合法调度，使至少 \(K\) 个 token 在各自 deadline 前完成？

NP-hardness 证明使用它的共同 deadline 特例：所有 token 具有相同 deadline，并令 \(K\) 等于 token 总数。此时问题变成：

> 是否存在一个合法的完整多层调度，使所有 token 都在共同最终 deadline 前完成？

如果能够多项式时间最优求解 SLO goodput，就能够回答这个决策问题。

## 4. 完整多层问题的 NP-Hardness

复杂度分析把 token 数、模型层数、专家数和 FFN GPU 数视为问题输入规模的一部分。对于某一个已经完全固定且只包含有限 token 的具体部署，当然可以通过有限枚举求得最优解；NP-hardness 描述的是这些规模随输入增长时，一般问题的最坏情况计算复杂度。

### 4.1 源问题：Job-shop Scheduling

规约来源是经典 Job-shop Scheduling Problem（JSSP）的 makespan 决策版本。

一个 JSSP 实例包含若干 jobs 和若干 machines。每个 job 由一系列必须按顺序执行的 operations 构成。每个 operation 指定一台 machine 和一个处理时间。一台 machine 同一时刻只能执行一个 operation，operation 一旦开始就不能被抢占。

给定共同时间上界 \(D\)，JSSP 决策问题询问：

> 是否存在一个合法调度，使所有 jobs 都在时间 \(D\) 前完成？

该问题是 NP-complete，其优化版本是 NP-hard。[Garey、Johnson 和 Sethi，1976](https://doi.org/10.1287/moor.1.2.117)

### 4.2 规约目标

给定任意一个 JSSP 实例，我们在多项式时间内构造一个多层 Top-K AFD 实例，使得：

> JSSP 实例存在 makespan 不超过 \(D\) 的调度，当且仅当构造出的 AFD 实例存在一个使全部 token 在共同最终 deadline 前完成的调度。

规约中的每个 token 都必须经过完整的多层执行路径。后层专家子任务只能在前层 fork-join 完成后释放，多个层的专家还会竞争相同 FFN GPU。因此，构造保留了完整的层间依赖与跨层设备竞争。

### 4.3 Job 到 token 的映射

对来源 JSSP 中的每个 job，创建一个 AFD token。

若来源 job 具有 \(L\) 道按顺序执行的 operation，则构造的 AFD 模型包含 \(L\) 个对应的 MoE block。来源 job 的第 \(l\) 道 operation 映射为该 token 在 block \(l\) 中的主要专家子任务。

经典 JSSP 可以使用每个 job 具有相同 operation 数量的受限形式。若选用允许 operation 数量不同的定义，也可以在较短 job 的尾部加入无竞争、零处理时间的辅助 operation，使所有 job 具有相同层数。该补齐只增加多项式规模，不改变来源实例的可行性。

### 4.4 Machine 到 FFN GPU 的映射

对来源 JSSP 中的每台 machine，创建一张对应的 FFN GPU。

如果某个来源 operation 必须在 machine \(M_j\) 上执行，则与它对应的主要 block–expert 被放置在 FFN GPU \(G_j\) 上。

不同 block 中的多个专家可以放置在同一张 FFN GPU 上。因此，来源问题中不同 jobs、不同 operation 对同一台 machine 的竞争，准确映射成多个 token 在不同模型层对同一张 FFN GPU 的竞争。

这里，模型层表示 operation 在 job 内部的位置，FFN GPU 表示来源 operation 使用的 machine。二者不是同一个维度。

### 4.5 Operation 到主要专家分支的映射

对来源 job 的每一道 operation，在对应 block 中创建一个主要 block–expert。该专家只接收与该来源 operation 对应的 token 子任务，并被放置到来源 operation 指定的 FFN GPU。

将该专家执行 singleton batch 的时间设置为来源 operation 的处理时间。

由于每个主要 block–expert 在构造中只接收一个对应子任务，它不会与其他子任务形成有效 batch。full-drain 规则自动满足，且不会改变来源 operation 的执行语义。

该构造允许不同主要 block–expert 具有不同 singleton 执行时间。完整问题本来就允许使用按 block–expert profiling 得到的执行时间，因此这是合法的受限实例。

### 4.6 层间前序依赖的映射

来源 job 的第 \(l+1\) 道 operation 只有在第 \(l\) 道 operation 完成后才能开始。

在构造的 AFD 实例中，token 在 block \(l+1\) 的主要专家子任务只有在以下事件完成后才能就绪：

1. block \(l\) 的全部专家分支完成；
2. block \(l\) 的结果汇合；
3. 层间固定处理完成。

在规约的受限实例中，将 Attention、通信和结果聚合的额外处理时间设置为零。于是，block \(l+1\) 的主要专家子任务恰好在 block \(l\) 的主要专家子任务完成后就绪，与来源 JSSP 的 operation precedence 完全一致。

这些零时间只用于复杂度规约。它们是完整非负延迟模型允许的一个特例，不表示真实系统中的通信和 Attention 没有成本。

### 4.7 固定 Top-K 的映射

如果目标 AFD 模型允许 Top-1，则每层只保留上述主要专家分支，层内汇合自然退化成单分支。

如果目标 AFD 模型固定采用 Top-K，且 \(K\geq 2\)，则为每个 token 的每个 block 再创建 \(K-1\) 个辅助专家分支。

这些辅助分支满足：

- 每个辅助分支位于没有其他任务竞争的独占资源；
- 辅助分支的处理时间设置为零，并可以在其就绪时立即完成；
- 下一层仍然必须等待主要分支和全部辅助分支汇合。

在从 JSSP 调度构造 AFD 调度的正向转换中，所有辅助分支都在就绪时立即执行，因此该层的汇合时间由主要专家分支决定。在反向转换中，即使某个 AFD 调度主动推迟辅助分支，只要 token 能在 deadline 前完成，它的主要分支也必然已经在 deadline 前完成，所以仍然能够恢复合法的 JSSP 调度。

因此，构造严格保留了 Top-K 的“全部已选专家完成后才能进入下一层”语义，同时不改变来源主要 operation 的可行时间关系。

辅助专家和辅助资源的数量与来源 operations 数量成线性关系，所以构造规模仍然是多项式的。

### 4.8 Deadline 的映射

为所有构造出的 AFD token 设置相同最终 deadline \(D\)，其中 \(D\) 就是来源 JSSP 的 makespan 上界。

由于规约中的层间额外处理时间为零，AFD token 完成最后一个 block 的时间等于对应来源 job 完成最后一道 operation 的时间。

令决策目标 \(K\) 等于 token 总数。因此，AFD 决策问题要求所有 token 都在 \(D\) 前完成。

完整映射关系如下：

| JSSP | 多层 AFD 实例 |
|---|---|
| 一个 job | 一个完整 token-step |
| job 的第 \(l\) 道 operation | token 在 block \(l\) 的主要专家分支 |
| operation 的指定 machine | 主要专家所在的 FFN GPU |
| operation processing time | singleton 专家 batch 执行时间 |
| job 内 operation 顺序 | token 的层间 fork-join 前序关系 |
| machine 互斥 | FFN GPU batch 互斥 |
| operation 不可抢占 | 专家 batch 不可抢占 |
| makespan 上界 \(D\) | 所有 token 的共同最终 deadline \(D\) |

### 4.9 正向证明

假设来源 JSSP 实例存在一个 makespan 不超过 \(D\) 的合法调度。

对来源调度中的每一道 operation，在相同时间启动构造出的对应主要专家 singleton batch，并保持相同执行时间。

来源调度满足 operation precedence，所以一个 job 的第 \(l+1\) 道 operation 不会早于第 \(l\) 道 operation 完成。构造出的 AFD 调度因而满足层间前序约束。

来源调度还满足 machine 互斥，所以映射到同一 FFN GPU 的两个主要专家 batch 不会重叠。辅助 Top-K 分支在独占资源上提前完成，不会推迟层内汇合。

因此，每个 AFD token 的主要专家路径与对应来源 job 具有完全相同的开始和完成时间。所有来源 jobs 都在 \(D\) 前完成，所以所有 AFD token 也都在共同最终 deadline \(D\) 前完成。

所以：

> 来源 JSSP 实例可行，则构造出的完整多层 AFD 实例可行。

### 4.10 反向证明

假设构造出的完整多层 AFD 实例存在一个合法调度，使所有 token 都在共同最终 deadline \(D\) 前完成。

从该调度中取出全部主要专家 batch，删除辅助 Top-K 分支，并把每个主要专家 batch 转换回对应来源 operation。

AFD 的层间 fork-join 约束保证同一个 token 的主要专家分支按照 block 顺序执行，因此转换后的来源 operations 满足 job precedence。

AFD 的 FFN GPU 互斥约束保证映射到同一来源 machine 的 operations 不会重叠。主要专家 batch 的执行时间与来源 operation processing time 相同，而且二者都不可抢占。

每个 AFD token 都在 \(D\) 前完成，所以每个来源 job 的最后一道 operation 也在 \(D\) 前完成。转换后的来源调度 makespan 不超过 \(D\)。

所以：

> 构造出的完整多层 AFD 实例可行，则来源 JSSP 实例可行。

结合正向和反向可得：

> 来源 JSSP 实例存在 makespan 不超过 \(D\) 的调度，当且仅当构造出的完整多层 AFD 实例存在使全部 token 满足最终 deadline 的调度。

### 4.11 构造复杂度

对每个来源 job 创建一个 token，对每道来源 operation 创建一个主要专家分支。固定 Top-K 时，每个主要分支再增加 \(K-1\) 个辅助分支。

因此，构造出的 token、block、专家分支和资源数量都至多是来源实例规模的多项式。所有映射和执行时间设置也可以在多项式时间内完成。

这是一条多项式时间规约。

### 4.12 决策问题属于 NP

给定一个候选 AFD 调度，可以在多项式时间内检查：

- 每个专家子任务是否在就绪后执行；
- 每个 batch 是否只包含相同 block–expert 的子任务；
- full-drain 规则是否满足；
- 同一 FFN GPU 上的 batch 是否重叠；
- batch 执行时间是否与其 block–expert 和 batch size 一致；
- 每个 token 的层内 Top-K 汇合是否正确；
- 每个 token 的层间前序关系是否满足；
- 至少 \(K\) 个 token 是否在各自 deadline 前完成。

因此，决策问题属于 NP。

### 4.13 复杂度结论

由 JSSP 的 NP-completeness、上述多项式时间规约以及 AFD 决策问题属于 NP，可以得到：

> **定理 1：** 多层离线 SLO-aware expert batch scheduling 的决策版本是 NP-complete，其优化版本是 NP-hard。该结论在以下受限条件下仍然成立：模型放置固定、所有 token 路由已知、所有 token 使用共同最终 deadline、每个 token 必须经过完整多层路径、每层严格满足 Top-K fork-join、采用 full-drain release，并且规约实例中不存在可利用的专家 batching。

该定理表明，仅由完整多层前序关系、Top-K 汇合和跨层 FFN GPU 竞争产生的调度选择，已经足以使最优端到端 SLO 调度成为 NP-hard 问题。允许不同 token 在同 block–expert 上动态合批，会在这个多层基础上继续增加 batch membership 和 release timing 的选择。

## 5. Batch Release 子问题的复杂性

完整多层问题包含另一类独立困难：即使上游调度已经确定，从而某张 FFN GPU 上未来 block–expert 子任务的到达时间已知，仅决定如何等待、合批和释放也可能是困难的。

### 5.1 源问题

考虑单机、带 family setup time 和 release time 的 batch scheduling。每个 job 具有 release time 和 family；只有相同 family 的 jobs 可以放入同一 batch；每次执行一个 family batch 都要支付一次固定 setup time。目标是最小化所有 jobs 的最终完成时间。

Yuan、Liu、Ng 和 Cheng 证明了该问题的强 NP-hardness。[Single machine batch scheduling problem with family setup times and release dates to minimize makespan](https://doi.org/10.1007/s10951-006-8776-2)

### 5.2 到专家 release 的映射

将来源问题映射到完整 AFD 调度在某张 FFN GPU 上的一个条件子问题：

| Family batch scheduling | AFD batch-release 子问题 |
|---|---|
| 一台 machine | 一张 FFN GPU |
| 一个 job | 一个已确定到达时间的专家子任务 |
| job release time | 子任务进入 block–expert 队列的时间 |
| job family | block–expert 队列 |
| 同 family 合批 | 同 block–expert 合批 |
| family setup time | 专家 batch 的固定启动与权重读取成本 |
| batch 顺序 | FFN GPU 上的专家 batch 顺序 |
| makespan 上界 | 该条件子问题的共同完成上界 |

在这个条件子问题中，上游多层调度只负责产生已经给定的到达序列。调度器仍然必须决定哪个 block–expert 先执行，以及是否等待未来同队列子任务。

对于“固定 batch 成本加逐 token 边际成本”的执行时间特例，一个包含 \(b\) 个子任务的专家 batch 执行时间设置为：

$$
T(b)=s+bp.
$$

这里，\(s\) 是每次启动专家 batch 的固定成本，\(p\) 是每个子任务的单位边际处理时间，\(b\) 是 batch 中的子任务数量。来源 family batch 可以直接转换成专家 batch，反向转换也成立。

来源问题通常允许把已经到达的同 family jobs 分到不同 batch，而本文采用 full-drain。对共同 makespan 目标，可以使用以下规范化过程。假设一个 family batch 在时间 \(t\) 启动，但某个在 \(t\) 前已经到达的同 family job 被留到后续 batch。把该 job 从后续 batch 移到当前 batch：当前 batch 增加该 job 的边际处理时间，中间 batch 只会向后移动相同时间，原后续 batch 则减少相同处理时间；执行到原后续 batch 结束时，累计增加的时间被抵消。所有被移动的中间 batch 只会更晚启动，因此不会违反 release time。如果原后续 batch 变空，还可以删除一次 setup。整个调度的最终 makespan 不会增加。

重复这个过程，直到每次启动一个 family 时都包含当时已经到达的全部未完成同 family jobs，即可得到满足 full-drain 的同样可行或更优调度。因此，这个映射也适用于本文的 full-drain release 语义。

所以，该条件 batch-release 子问题至少与来源问题一样困难。

> **定理 2：** 给定由上游多层执行产生的专家子任务到达序列，仅优化一张 FFN GPU 上的 block–expert batch release 和执行顺序，已经是强 NP-hard 的。

定理 1 和定理 2 分别说明两类复杂性：

| 复杂性来源 | 对应结论 |
|---|---|
| 多层前序、Top-K 汇合和跨层 GPU 竞争 | 完整多层问题 NP-hard |
| 异步到达、同专家聚合和 batch release | 条件 release 子问题强 NP-hard |

## 6. 最终问题表述

本文研究的问题可以归纳如下：

> 在固定部署的多层 AFD MoE 解码系统中，每个 token 依次经过全部 Transformer block。每个 block 是一个 fork-join 阶段：token 被路由到一个已选专家集合，每个专家产生一个可以和其他 token 的相同 block–expert 子任务动态合批的分支；只有全部已选专家完成并聚合后，token 才能进入下一层。后续层专家队列的到达时间由前面层的调度内生决定，多个模型层的专家还可能争用同一张 FFN GPU。调度器需要联合决定每张 FFN GPU 上跨层 block–expert batch 的执行顺序和释放时间，使尽可能多的 token 在端到端 deadline 前完成；在 SLO goodput 相同的情况下，进一步最小化 FFN GPU 总执行时间。

这个问题不能通过独立优化每一层得到全局最优结果。当前层的等待会改变本层汇合时间，并进一步改变所有后续层的到达、batch formation、资源竞争和剩余 SLO。完整离线优化需要在全部模型层和全部 token 的执行图上联合选择 batch release 与 GPU dispatch。

完整多层问题的决策版本是 NP-complete，优化版本是 NP-hard；其中仅 batch release 条件子问题也已经是强 NP-hard。这些复杂度结果说明，后续调度方法需要使用低开销的近似、分解或滚动优化，而不能依赖对完整未来执行图进行精确求解。
