# 控制器无关的残差通道(Residual Channel)——方法说明

*即插即用力感知的核心。配套 `paper/plugandplay/` 与记忆
`crosspolicy-plugandplay-result`。公式用 LaTeX;`$$` 块在 VS Code / GitHub
可直接渲染,也可直接复制进论文。*

---

## 1. 问题:原始本体感知为什么不能迁移

力感知要从关节力矩里判断外力。但**执行器力矩 $\boldsymbol{\tau}_{\mathrm{act}}$ 是一锅混合量**:它同时在扛重力、加减速肢体、抗外力,而且**它的分布由控制器(刚度)决定**。

学出来的传感器,是在**某个控制器**下把"什么力矩算正常(无接触)"这条线校准好的。换个控制器(比如 SONIC),这条线就错位:硬控制器无接触时的正常纠偏力矩,被读成接触 → **精确率崩(实测 0.97 → 0.65)、定位崩(0.78 → 0.40)**。

**根因:传感器过拟合了控制器产生力矩的方式。** 解决办法:换一个**携带同样接触信息、但与控制器无关**的输入通道。

---

## 2. 运动方程

浮动基刚体动力学(广义坐标 $\mathbf{q}$,含基座 6 自由度 + 关节):

$$
\mathbf{M}(\mathbf{q})\,\ddot{\mathbf{q}} \;+\; \mathbf{c}(\mathbf{q},\dot{\mathbf{q}})
\;=\; \boldsymbol{\tau}_{\mathrm{act}}
\;+\; \boldsymbol{\tau}_{\mathrm{passive}}
\;+\; \boldsymbol{\tau}_{\mathrm{con}}
\;+\; \boldsymbol{\tau}_{\mathrm{ext}}
$$

各项:

| 符号 | 含义 | 怎么得到 |
|---|---|---|
| $\mathbf{M}(\mathbf{q})$ | 质量矩阵(**满阵**,编码所有关节耦合) | 已知模型 |
| $\mathbf{c}(\mathbf{q},\dot{\mathbf{q}})$ | 科氏 + 离心 + 重力(偏置力,MuJoCo 的 `qfrc_bias`) | 已知模型 |
| $\boldsymbol{\tau}_{\mathrm{act}}$ | 执行器力矩 | 实测(真机=电流估 $\tau=K_t i$) |
| $\boldsymbol{\tau}_{\mathrm{passive}}$ | 被动力(关节阻尼/弹簧) | 已知模型 |
| $\boldsymbol{\tau}_{\mathrm{con}}$ | 约束力(地面反作用力,GRF) | 仿真直接有;真机需估 |
| $\boldsymbol{\tau}_{\mathrm{ext}}$ | **外部接触广义力(我们要的)** | 未知 |

外力和接触点的关系(雅可比转置):

$$
\boldsymbol{\tau}_{\mathrm{ext}} \;=\; \mathbf{J}_c^{\!\top}(\mathbf{q})\,\mathbf{F}_{\mathrm{ext}}
$$

一个外力 $\mathbf{F}_{\mathrm{ext}}$ 经 $\mathbf{J}_c^{\!\top}$ **传到接触点到基座之间的所有关节**——这就是定位靠的"跨关节签名"。

---

## 3. 残差:把外力解出来

对运动方程解 $\boldsymbol{\tau}_{\mathrm{ext}}$:

$$
\boxed{\;
\hat{\boldsymbol{\tau}}_{\mathrm{ext}} \;=\;
\mathbf{M}(\mathbf{q})\,\ddot{\mathbf{q}}
\;+\; \mathbf{c}(\mathbf{q},\dot{\mathbf{q}})
\;-\; \boldsymbol{\tau}_{\mathrm{act}}
\;-\; \boldsymbol{\tau}_{\mathrm{passive}}
\;-\; \boldsymbol{\tau}_{\mathrm{con}}
\;}
$$

**含义:用已知物理算出"要产生当前这个运动、该有多大力矩"($\mathbf{M}\ddot{\mathbf{q}}+\mathbf{c}$),减去电机实际使的力矩和已知的被动/地反力,剩下的必然是外力。**

这不是新传感器——用的是**同样的测量($\mathbf{q},\dot{\mathbf{q}},\boldsymbol{\tau}_{\mathrm{act}}$)+ 已知模型($\mathbf{M},\mathbf{c}$)**。区别只在于**物理是网络自己隐式学,还是我们显式替它算**。我们把 $\hat{\boldsymbol{\tau}}_{\mathrm{ext}}$(29 维,窗口 6 帧)作为一个新输入通道喂给同一个 MLP。

---

## 4. 为什么控制器无关(关键性质)

**无接触时** $\boldsymbol{\tau}_{\mathrm{ext}}=\mathbf{0}$,运动方程给出:

$$
\mathbf{M}\ddot{\mathbf{q}} + \mathbf{c}
- \boldsymbol{\tau}_{\mathrm{act}} - \boldsymbol{\tau}_{\mathrm{passive}} - \boldsymbol{\tau}_{\mathrm{con}}
\;=\; \mathbf{0}
\qquad\Longrightarrow\qquad
\hat{\boldsymbol{\tau}}_{\mathrm{ext}} \equiv \mathbf{0}
$$

**不管控制器使多大力矩:软控制器使小 $\boldsymbol{\tau}_{\mathrm{act}}$、硬控制器使大 $\boldsymbol{\tau}_{\mathrm{act}}$——只要没外力,那个力矩就正好平衡物理,残差都恒为 0。**

- **无接触:残差 ≈ 0(对所有控制器)** → 传感器的"正常线"永远钉在 0 → 不误报。
- **有接触:残差 $=\boldsymbol{\tau}_{\mathrm{ext}}=\mathbf{J}_c^{\!\top}\mathbf{F}$**(同一个物理量,与控制器无关)→ 同样的签名 → 迁移。

对比:原始本体感知把"正常线"校准到某个控制器的力矩水平上,换控制器就错位。残差把它统一钉在物理零点。

---

## 5. 顺带解决"腿"(减掉地反力)

腿难在:脚踩地时,**外力和地面反作用力 $\boldsymbol{\tau}_{\mathrm{con}}$ 混在腿的关节力矩里**(stance masking)。残差里**显式减掉了 $\boldsymbol{\tau}_{\mathrm{con}}$** → 正好去掉这一掩盖项 → 腿现形。

- 前提:要有 $\boldsymbol{\tau}_{\mathrm{con}}$(地反力)估计。
- 上半身(手臂/躯干)无地面接触,$\boldsymbol{\tau}_{\mathrm{con}}\approx 0$,**不需要它就干净**。

---

## 6. 三个问题,一个通道

$$
\hat{\boldsymbol{\tau}}_{\mathrm{ext}} =
\underbrace{\mathbf{M}\ddot{\mathbf{q}} + \mathbf{c}}_{\text{减掉自身惯性/重力} \Rightarrow \text{动态运动}}
\;\underbrace{-\;\boldsymbol{\tau}_{\mathrm{act}}}_{\text{减掉控制器} \Rightarrow \text{即插即用}}
\;-\;\boldsymbol{\tau}_{\mathrm{passive}}
\;\underbrace{-\;\boldsymbol{\tau}_{\mathrm{con}}}_{\text{减掉地反力} \Rightarrow \text{腿}}
$$

| 问题 | 残差减掉谁 |
|---|---|
| 换控制器 | $\boldsymbol{\tau}_{\mathrm{act}}$ → 控制器无关 |
| 动态运动(惯性/科氏淹没外力) | $\mathbf{M}\ddot{\mathbf{q}}+\mathbf{c}$ → 去掉自身动力学 |
| 腿(stance masking) | $\boldsymbol{\tau}_{\mathrm{con}}$ → 去掉地反力 |

---

## 7. 真机怎么算(各项来源)

| 项 | 仿真 | 真机 |
|---|---|---|
| $\mathbf{M},\mathbf{c},\boldsymbol{\tau}_{\mathrm{passive}}$ | 模型直接算 | 模型直接算(需准确的动力学参数) |
| $\boldsymbol{\tau}_{\mathrm{act}}$ | 精确施加力矩 | **电流估** $\tau = K_t\, i$(更噪) |
| $\ddot{\mathbf{q}}$ | 直接读 | **差分 $\dot{\mathbf{q}}$**(噪声大) |
| $\boldsymbol{\tau}_{\mathrm{con}}$(GRF) | `qfrc_constraint` 直接有 | **需估**:脚底 F/T,或浮动基动量观测器(MOB) |

**唯一仿真特权项是 $\boldsymbol{\tau}_{\mathrm{con}}$。** Unitree G1 **没有脚底力传感器**,所以真机上 $\boldsymbol{\tau}_{\mathrm{con}}$ 要用**浮动基动量观测器(MOB,见 MOB-Net)**估。

### 动量观测器形式(避开噪声大的 $\ddot{\mathbf{q}}$)

真机上不想直接用 $\ddot{\mathbf{q}}$,可用广义动量 $\mathbf{p}=\mathbf{M}\dot{\mathbf{q}}$ 的观测器:

$$
\mathbf{r} \;=\; \mathbf{K}_I\!\left(\mathbf{p}
- \int_0^t\!\big(\boldsymbol{\tau}_{\mathrm{act}} + \mathbf{C}^{\!\top}\dot{\mathbf{q}} - \mathbf{g} + \mathbf{r}\big)\,\mathrm{d}t\right)
\;\xrightarrow{\;t\to\infty\;}\; \hat{\boldsymbol{\tau}}_{\mathrm{ext}}
$$

$\mathbf{r}$ 收敛到外部广义力,**不需要 $\ddot{\mathbf{q}}$**(只用 $\dot{\mathbf{q}}$ 和力矩),对真机更友好。

---

## 8. 实测效果(MuJoCo,7 控制器留一)

| | 原始本体感知 | **残差** |
|---|---|---|
| 跨控制器最坏 regAcc | 0.417 | **0.905** |
| 平均 regAcc | 0.653 | **0.931** |
| 精确率 | 崩到 0.53 | **0.97–0.99** |
| 手臂召回 | 0.80 | **0.98** |
| 腿召回 | 0.56 | **0.89** |

**可实现性**(去掉特权 $\boldsymbol{\tau}_{\mathrm{con}}$):手臂 0.94(纯本体感知完全可实现),腿 0.62;给一个带噪的 GRF 估计,腿平缓恢复(25% 误差 → 0.79)。

---

## 9. 局限(诚实)

1. **"不同控制器"目前=缩放 kp/kd**(温和代理);真换一个网络(SONIC/BeyondMimic)是更大漂移,理论上残差照样无关(只读实测力矩+运动),但**未验证**。
2. **$\boldsymbol{\tau}_{\mathrm{con}}$ 真机要 MOB 估**,G1 无脚底力传感器;腿的精度取决于 MOB 质量。
3. **动态运动**($\ddot{\mathbf{q}}$ 噪声大)未测;**真机**($\boldsymbol{\tau}_{\mathrm{act}}$ 电流估更脏、踝部并联连杆未建模)未上。

---

## 一句话

**原始本体感知把力矩直接丢给网络,让它隐式学一个"绑定控制器的近似残差";我们用运动方程显式算残差 $\hat{\boldsymbol{\tau}}_{\mathrm{ext}}=\mathbf{M}\ddot{\mathbf{q}}+\mathbf{c}-\boldsymbol{\tau}_{\mathrm{act}}-\boldsymbol{\tau}_{\mathrm{passive}}-\boldsymbol{\tau}_{\mathrm{con}}$——无接触时对任何控制器恒为 0,有接触时等于 $\mathbf{J}_c^{\!\top}\mathbf{F}$,所以换控制器不用重训(即插即用),而且同一式子顺带去掉了自身动力学(动态)和地反力(腿)。**
