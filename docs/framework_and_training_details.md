# 整体框架与训练细节（以当前代码为准）

> 本文档回答:我们现在整套系统是怎么搭的——分几层控制、每一层是什么模型、
> 每个模型的输入/观测/输出各是什么(逐维度)、怎么训练的。
> 全部内容对照当前代码(branch `wholebodycontact`, 2026-07-07)逐行核对,
> 关键事实附 `file:line` 引用。
>
> 结果与实验分析见 `docs/overnight_experiment_report.md`;
> 方法调研与路线图见 `docs/blind_maze_research_and_method.md`。

---

## 0. 总览:两套系统

当前仓库里有**两条并行的技术线**,共享同一个北极星目标(真实 G1 纯接触盲走出迷宫):

| 系统 | 仿真器 | 作用 | 状态 |
|---|---|---|---|
| **① mjlab 盲走迷宫管线** (`mjlab_maze/`) | MuJoCo-Warp (mjlab) | 北极星的端到端仿真验证:行走 + 接触感知 + 导航闭环 | ✅ 闭环已通,批量成功率 17–25%,瓶颈=低层无接触训练 |
| **② Isaac GentleHumanoid 分层栈** (主仓库) | Isaac Lab | 分层柔顺控制(HL→冻结LL→PD)+ Plan A 静态力感知 | Plan A 完成;Plan B(力估计喂 HL)未开始 |

两套系统的"控制分层"结构不同:

- **mjlab 线是三层**:Pledge 导航状态机(规则,50 Hz)→ 速度指令 → RL 行走策略(50 Hz)→ PD 关节位置控制(200 Hz)。感知(ContactGRU)是旁路读取,不直接控制。
- **Isaac 线也是三层**:高层 RL 策略(50 Hz,输出 root/腕部残差指令)→ 冻结低层运动跟踪 RL 策略(50 Hz,输出 29 关节位置偏移)→ PD 隐式执行器(200 Hz)。Plan A 力传感器同样是旁路监督模型。

下面分两部分逐层展开。

---

# 第一部分:mjlab 盲走迷宫管线(`mjlab_maze/`)

## 1. 架构与数据流

每个控制步(50 Hz,`CTRL_DT=0.02`,`nav_run.py:27`)的闭环:

```
仿真状态
 ├─→ 行走策略 obs (99维) ──→ 行走策略 ──→ 29维关节动作 ──→ PD(200Hz) ─→ 物理
 ├─→ 本体感知 96维 ─→ ContactGRU(H=50) ─→ 检测门控方位 ─→ 12世界bin ─┐
 │    (或: ContactSensor 真值 → gt_sectors → 12世界bin)              │
 └─→ pose2d (x,y,yaw) ────────────────────────────────────────────┤
                                                                    ▼
                                    Pledge 状态机 ──(vx,vy,wz)──→ 覆写行走策略的 twist 指令
```

关键接线(`nav_run.py:62-90`):每步先取策略动作 `act=policy(obs)`;取扇区
(真值 `mi.gt_sectors(force_thresh=3.0)` 或学习传感器
`est.world_bins(fx.proprio(), yaw)`);逐 env 调 `PledgeController.step()` 得
`(vx,vy,wz)`;`mi.set_cmd_batch(cmds)` 直接写进指令项的
`vel_command_b[:,0:3]`(`maze_env.py:131-139`);再 `env.step(act)` 推进物理。
环境自身的指令重采样被设成 `(1e6,1e6)` 失效(`maze_env.py:100`),保证导航器独占指令。

终止判定(`nav_run.py:92-113`):成功 = `mi.success()`(越过打开的东出口,
x 过线且 |y−exit_y|<1.2);摔倒 = 骨盆高度 <0.35 m(`maze_env.py:176-179`);
越界(穿墙)= 位置超出迷宫包围盒;其余到时限 = 超时。

## 2. 模型 ①:G1 行走策略(mjlab velocity, rsl_rl PPO)

任务 `Mjlab-Velocity-Flat-Unitree-G1`,从零训练。
训练 dump:`mjlab_maze/logs/g1_velocity/2026-07-05_00-41-09/params/{env,agent}.yaml`。
G1 共 **29 个驱动关节**。

### 2.1 观测(actor 组,共 99 维,按序拼接)

`concatenate_terms=true`,训练时加均匀噪声(play 时关闭):

| # | 项 | 来源 | 维度 | 噪声(±均匀) |
|---|---|---|---|---|
| 1 | `base_lin_vel` | 传感器 `robot/imu_lin_vel` | 3 | 0.5 |
| 2 | `base_ang_vel` | `robot/imu_ang_vel` | 3 | 0.2 |
| 3 | `projected_gravity` | 重力在基座系投影 | 3 | 0.05 |
| 4 | `joint_pos` | 关节位置(相对默认位形) | 29 | 0.01 |
| 5 | `joint_vel` | 关节速度 | 29 | 1.5 |
| 6 | `actions` | 上一步动作 | 29 | — |
| 7 | `command` | twist 指令 `[vx,vy,wz]`(基座系) | 3 | — |

**共 3+3+3+29+29+29+3 = 99 维。** 无逐项缩放(`scale: null`),无历史堆叠;
obs 归一化由 rsl_rl 的 running normalizer 做(`obs_normalization: true`)。

**critic 组(非对称 actor-critic)**:同上 7 项 + 4 个特权项
(`foot_height`、`foot_air_time`、`foot_contact`、`foot_contact_forces`,
均来自 `feet_ground_contact` 传感器),critic 无噪声。

### 2.2 动作(29 维)

`JointPositionAction`:**目标关节位置 = 默认位形 + action × 每关节 scale**
(`use_default_offset: true`)。scale = `0.25 × effort_limit / stiffness`
(`g1_constants.py:278-286`),实际值:

| scale | 关节 |
|---|---|
| 0.4386 | elbow, shoulder pitch/roll/yaw, wrist_roll, waist_pitch/roll, ankle_pitch/roll |
| 0.5476 | hip_pitch, hip_yaw, waist_yaw |
| 0.3507 | hip_roll, knee |
| 0.0745 | wrist_pitch, wrist_yaw |

默认位形是屈膝关键帧(hip_pitch −0.312, knee 0.669, ankle_pitch −0.363,
elbow 0.6 等),出生高度 z=0.76 m。

### 2.3 底层 PD 控制(第三层控制)

物理 200 Hz(`timestep 0.005`),控制 50 Hz(`decimation=4`)。
`BuiltinPositionActuatorCfg` 位置执行器,增益由 `Kp=armature·ω²`(ω=2π·10 Hz)、
`Kd=2ζ·armature·ω`(ζ=2.0)推出,编译后实际值:

| 电机组 | 关节 | Kp | Kd | 力矩上限(N·m) |
|---|---|---|---|---|
| 5020 | elbow, shoulder×3, wrist_roll | 14.25 | 0.907 | 25 |
| 7520-14 | hip_pitch, hip_yaw, waist_yaw | 40.18 | 2.558 | 88 |
| 7520-22 | hip_roll, knee | 99.10 | 6.309 | 139 |
| 4010 | wrist_pitch/yaw | 16.78 | 1.068 | 5 |
| 腰/踝 (5020×2) | waist_pitch/roll, ankle_pitch/roll | 28.50 | 1.814 | 50 |

### 2.4 训练(PPO, rsl_rl)

- **网络**:actor MLP (512,256,128) ELU,高斯策略 init_std=1.0;critic 同尺寸。无 RNN。
- **PPO**:lr 1e-3(adaptive, desired_kl=0.01),γ=0.99,λ=0.95,clip 0.2,
  entropy 0.01,5 epochs × 4 minibatch,rollout 24 步 × **4096 envs**
  (≈98k 步/迭代),**3000 迭代 ≈ 2.95 亿环境步**,一晚训完(reward 67.5,回合长度 998/1000)。
- **奖励**(权重):跟踪线速度 +2.0 / 角速度 +2.0(exp 核),直立 +1.0,
  姿态(站/走/跑三档 std)+1.0;惩罚:躯干角速度 −0.05、角动量 −0.02、
  关节限位 −1.0、动作变化率 −0.1、足底打滑 −0.1、抬脚高度 −2.0 /
  摆动高度 −0.25(目标 0.1 m)、自碰撞 −1.0。
- **域随机化**:每 1–3 s 随机推一次(速度踢 ±0.5 m/s,角速度踢 ±0.5–0.78 rad/s)、
  足底摩擦 [0.3,1.2]、编码器偏置 ±0.015 rad、躯干质心偏移 ±0.025/0.03 m。
- **指令分布**:vx,vy ∈ [−1,1]、wz ∈ [−0.5,0.5] 起步,带课程扩到 vx [−2,3];
  10% 站立 env、30% 朝向控制 env。
- 回合 20 s,摔倒判定 = 倾角 >70°。
- **重要**:训练时**没有任何墙壁/持续接触**——这正是当前批量失败的根源
  (贴墙缠斗摔倒),下一步的接触课程重训就是往这里加墙体和持续侧推事件。

## 3. 模型 ②:接触传感器 ContactGRU(监督学习,非 RL)

作用:**只用本体感觉**估计"是否在碰墙、墙在哪个方向、多大力"。部署时替代
接触真值传感器。

### 3.1 输入:96 维本体感觉(`features.py:35-46`)

按序拼接,**无任何缩放**(归一化在训练/部署时用数据集统计量做):

| # | 分量 | 维度 | 来源 |
|---|---|---|---|
| 1 | 关节位置 q | 29 | `robot.data.joint_pos` |
| 2 | 关节速度 dq | 29 | `robot.data.joint_vel` |
| 3 | 执行器力矩 τ | 29 | `sim.data.actuator_force` |
| 4 | IMU 角速度 | 3 | `robot/imu_ang_vel` |
| 5 | IMU 重力方向(upvector) | 3 | `robot/imu_upvector` |
| 6 | IMU 线加速度 | 3 | `robot/imu_lin_acc` |

**刻意排除**:速度指令、动作/控制器输入(SixthSense 的可部署性原则)、
基座位置与 yaw(部署时导航层再用 IMU yaw 转世界系)。

### 3.2 标签(来自仿真接触传感器真值,`features.py:49-63`)

接触传感器 `wall_contact`(`maze_env.py:88-97`):主体 = 机器人 **24 个连杆**
(骨盆/躯干/肩×3/肘/腕×3/髋×3/膝,**刻意排除双脚**——脚碰墙会传导到膝/髋,
且避免槽位爆炸);对象 = **仅 maze_wall 几何体**(脚踩地永不污染标签);
输出世界系"机器人→墙"净力。

逐 env 计算三个标签:
1. **contact(bool)**:各连杆平面力(去掉 z)范数 >0.5 N 视为激活,把激活连杆
   的平面力**矢量求和**得净力 `fnet`,`mag=‖fnet‖`;`contact = mag > 3.0 N`。
2. **az_r(rad)**:净力世界方位 `az_w=atan2(fy,fx)` 减去机器人 yaw,包到 (−π,π]
   —— **机器人系方位角**(可部署;世界系由导航层加回 yaw 恢复)。
3. **mag(N)**:净力平面幅值。

### 3.3 数据采集(`collect_sensor_data.py`)

- 冻结的行走策略在 4×4 迷宫里**随机游走自然撞墙**:指令每 3 s 重采样,
  vx ∈ [−0.3,0.7](前向偏置,让机器人"巡航撞墙"),vy ∈ ±0.1,wz ∈ ±0.7。
- 32 env × 12000–15000 步 × 2 个迷宫种子;回合 10000 s(摔倒才 reset,
  reset 记入 `reset` 标志用于切断训练窗口)。
- 存 HDF5:`X (T,N,96) f32`、`contact/reset (T,N) bool`、`az_r/mag (T,N) f32`。
- 实测接触率 ~22%(天然 1:4 正负比),贴墙平均力 ~27 N(**持续接触**——
  这正是文献没人做过的分布,也是指标远超瞬时推力设置的原因)。

### 3.4 模型结构(`sensor_model.py:13-24`)

```
输入 [B, H=50, 96](z-score 归一化)
  → GRU(96→128, 1 层, batch_first) → 取最后时刻隐状态 z(128)
      ├─ det 头: Linear(128,1) → 接触概率 logit(部署时 sigmoid)
      ├─ az  头: Linear(128,2) → (sin, cos) 机器人系方位
      └─ mag 头: Linear(128,1) → 力幅值(训练目标除以 50 N)
```
约 9 万参数。**H=50 帧窗口(=1 秒 @50 Hz)**,是调研中收敛的窗口长度
(SixthSense/RMA/MOB-Net 同款)。

### 3.5 训练(`train_sensor.py`)

- **窗口索引**:所有"窗口内无 reset"的 (t,e) 对(用 reset 累积和判断)。
- **划分**:按 env 划分,25% env 留出为验证(无窗口泄漏)。
- **归一化**:训练子样本(≤10 万窗)的 mean/std,存进 ckpt。
- **损失**:`BCEWithLogits(det, pos_weight=min(负/正, 8))`
  `+ 2.0 × MSE(az_sin_cos)`(仅接触帧)
  `+ 0.5 × SmoothL1(mag/50)`(仅接触帧)。
- **增强**:归一化后加高斯噪声 σ=0.01。
- **优化**:AdamW lr 1e-3,wd 1e-6,8 epochs,batch 4096,按最优验证 F1 存 ckpt。
  分钟级训完。
- **指标**:det P/R/F1、误报率、方位误差(接触帧平均绝对包角误差)、12-bin 准确率。
- **结果**:det F1 0.949 / 方位 8.3° / 误报 0.2%;跨迷宫(`eval_cross_maze.py`,
  换迷宫种子全量评估)F1 0.943。

### 3.6 部署封装 RollingEstimator(`sensor_model.py:27-63`)

在线维护每 env 的 H=50 帧滚动缓冲(首帧重复填充,之后 `torch.roll`);
每步:归一化 → GRU 前向 → `p=sigmoid(det)`;
`az_w = atan2(sin,cos) + yaw`(IMU yaw 转回世界系);
`bin = ((deg+15)%360)//30` → **12 个 30° 世界 bin,每步至多点亮 1 个**
(`p > det_thresh=0.5` 才点亮)。去抖不在这里做,在导航层做。

## 4. 第三个"控制器":Pledge 导航状态机(规则,非学习)

`pledge.py`——仿真器无关的纯反应式沿墙控制器。

### 4.1 输入 / 输出

- **输入**(`step(raw_world_bins, yaw, pos)`):12 个世界系 30° 方位 bin(bool,
  bin b 覆盖 b·30°±15°)、yaw(rad)、位置 (x,y)(仅用于卡死看门狗)。
- **输出**:`(vx, vy, wz, state)`。vy 恒 0;vx ∈ {0.7(v_walk), 0.3(v_slow), 0, −0.3};
  wz 上限 0.7(原地转)/ 0.35(航向修正)/ 0.45(外角绕弯)。

### 4.2 关键设计

- **为什么 12×30° 世界 bin**:轴对齐墙的方位(0/90/180/270°)恰落在 bin **中心**
  (60° bin 会正好落在边界上=抖动);**去抖必须在世界系做**——墙不随机器人转,
  机器人转身时机器人系扇区会乱跳。
- **k-of-n 去抖**(`Debounce`,作用在 12 世界 bin 上):k_on=3 帧确认接触
  (把 15% 单帧误报压到 ~0.3%),k_off=10 帧确认丢墙。无 EMA(注释提到但未实现)。
- 去抖后再按当前 yaw 旋转成 6×60° 机器人系扇区
  (FRONT/FL/BL/BACK/BR/FR)供状态机用。

### 4.3 状态机(EXPLORE → ACQUIRE → FOLLOW → DEPART → EXPLORE + STUCK)

| 状态 | 行为 | 转出条件 |
|---|---|---|
| EXPLORE | 朝首选航向 φ0 直走(vx=0.7,航向 P 控制 ±0.35) | 前方/前侧碰墙→记住墙方位,目标 yaw=墙方位−90°(墙放左手),进 ACQUIRE;卡死→STUCK |
| ACQUIRE | **开环 yaw-servo 原地右转**(0,0,−0.7)转到记忆方位——转身中接触会丢失,故不依赖接触;倒车顶墙曾是头号摔倒原因,故此态零平移 | 与目标 yaw 差 <12°→FOLLOW;卡死→STUCK |
| FOLLOW | 左手贴墙:前方有墙→原地右转;左前/左后有墙→前进微离墙(0.7,0,−0.05);丢墙(外角)→慢速左弧线绕角(0.3,0,+0.45) | 贴墙 >90 s→随机踢 φ0(±75°)回 EXPLORE(RAMBLER);丢墙 >8 s→回 EXPLORE;Pledge 离墙条件(见下);卡死→STUCK |
| DEPART | 沿 φ0 直走一步 | 立即回 EXPLORE;遇墙回 ACQUIRE |
| STUCK | 倒车右转(−0.3,0,−0.7)1.5 s | 回 EXPLORE |

- **卡死看门狗**:4 s 内位移 <0.15 m 判卡死(所有状态生效)。
- **Pledge 计数器**:贴墙期间累计 yaw 变化 `turn_acc`;经典 Pledge 离墙条件是
  `|turn_acc|<tol 且 |yaw−φ0|<tol`。当前 **`yaw_tol=0` 即离墙被禁用**——
  完美迷宫(无环、墙全连外界)纯左手规则已保证完备,离墙只在非树迷宫需要。
- φ0 初始化为 0(指南针假设:出口在东/+x)。
- 常数注释里写明了实测折扣:策略只跟得上 ~70% 的 v_walk、~64% 的 w_turn。

### 4.4 2D 运动学验证(`test_pledge_2d.py`)

独轮车模型(半径 0.35 m + 0.05 m 接触壳,DT=0.02),几何接触 + 滑动碰撞响应;
噪声模式:真 bin 15% 漏检、假 bin 共 15%/帧误报。结果:4×4 干净 96% / 带噪 98%,
6×6 87–90%——验证控制器与去抖链本身没问题。

## 5. 迷宫环境(`maze_gen.py` / `maze_env.py`)

- **生成**:递归回溯 DFS 完美迷宫(单连通、无环),每格 2.0 m,墙厚 0.3 m
  (0.1 m 薄墙会被步态冲击穿透!),墙高 1.6 m(过肩,让手臂/躯干接触;
  1.0 m 时接触分布劣化)。右上角东墙打开为出口。
- **注入**:`SceneCfg.spec_fn` 闭包在场景编译时向 MuJoCo spec 添加静态 box 几何体
  (`maze_wall_i`)。`shifted_maze` 把起点格中心平移到世界原点(机器人出生点)。
- 环境用 play 模式构建 + 超长回合;录像用固定俯视上帝相机(1280×960)。

---

# 第二部分:Isaac GentleHumanoid 分层栈(主仓库)

## 6. 控制层级(三层)

```
高层 RL 策略 (root_ppo, 50 Hz)
   │  输出:root/腕部残差指令 (5/12/17 维)
   ▼
冻结低层运动跟踪策略 (student PPO, 50 Hz)
   │  输出:29 维关节位置偏移
   ▼
PD 隐式执行器 (200 Hz, decimation=4)   [Kp 14–99, Kd 0.9–6.3,与 mjlab 线同一组 G1 增益]
```

`HierarchicalRootCommand`(`action.py:158`)把两层粘起来:每控制步把 HL 动作
解码成指令 → 写进指令管理器/低层的 obs → 查询冻结低层得关节动作 → 交给
`JointPosition` 执行。

## 7. 模型 ③:低层运动跟踪策略(teacher–student PPO)

冻结使用的 run:`gentle_finetune_3point_amass_limmt_full_stiff30`
(本地缓存名 `3kp_amass_limmt_full_stiff600`)。
**"3kp"= 3 keypoints(root+两腕三点参考),不是增益;"stiff"= 关掉导纳的
硬跟踪变体;"30"= 30 N 扰动上限。**

### 7.1 跟踪什么

AMASS 蒸馏的 3 点参考,通过两个观测项进入策略:
- `command`(6 维):`[root高度(1), 目标线速度_b xy(2), 目标朝向_b xy(2), 安全力限(1)]`
  (`motion_tracking.py:2396-2443`)。
- `root_and_wrist_6d`(12 维):左右腕位置_b(3+3)+ 左右腕轴角(3+3)。

### 7.2 观测(G1 29 DOF)

**`policy` 组(student 可见,≈257 维)**:

| 项 | 维度 |
|---|---|
| boot_indicator_state | 1 |
| command | 6 |
| root_and_wrist_6d | 12 |
| root_ang_vel(当前帧) | 3 |
| projected_gravity(当前帧) | 3 |
| joint_pos 5 帧历史 | 145 |
| prev_actions ×3 | 87 |

**`priv` 组(teacher 特权,经编码器压成 256 维 latent)**:目标位姿/速度/相对四元数/
目标重力、`force_priv`、身体高度(骨盆/躯干/双踝)、踝接触力、root 线速度(EMA)、
9 帧角速度/重力/关节历史、当前与目标关键点、applied_action/torque 等。
**`joint_target` 组(teacher 直连)**:目标关节位置 29。
**`priv_critic`**:累计误差。

### 7.3 网络与训练(`ppo.py`)

- 特权编码器 MLP[256]→256 latent;student 状态估计器 MLP[512,256]→256 +
  关节预测器 MLP[512,256]→29(从 `policy` 组预测特权 latent 与目标关节)。
- **Actor MLP[512,512,256]**(Mish + LayerNorm),输出 29 维高斯;
  teacher 吃 `[policy, priv_feature, joint_target]`,student 吃
  `[policy, priv_pred, priv_joint]`。Critic MLP[512,512,256]。
- PPO:lr 1e-4,5 epochs × 8 minibatch,clip 0.2,desired_kl 0.01,
  entropy 0.002→0.0005,γ=0.99 λ=0.95,phase=finetune。
- **冻结部署时强制 `phase="adapt"`**:只用 `policy` 观测 + 自预测 latent 跑
  student actor(`frozen_low_level.py:177-208`),VecNorm 用训练统计量(eval)。

### 7.4 动作与 PD

`JointPosition`:`目标 = 默认关节位置 + action × 每关节 scale`
(elbow/shoulder/wrist 1.0,hip_pitch/knee/ankle 0.5,hip_roll/yaw/waist 0.25);
raw clamp ±10,一阶低通 α≈0.9,通信延迟 ≤4 物理步。
PD 增益与 mjlab 线相同(同一组 G1 常数,见 §2.3 表)。
时序:控制 50 Hz / 物理 200 Hz / decimation 4。

## 8. net_pull 外力探针(Plan A 的数据来源)

`MotionTrackingCommand_impedance` 的 `external_force_mode="net_pull"`
(`motion_tracking.py:1378`):

- **选体**:从 `net_pull_apply_pattern` 匹配候选连杆,每回合/相位起点均匀抽 1 个。
  Plan A 采集配置(`cfg/wbc/collect.yaml:19-31`)用 **12 个连杆**(即 one-hot 顺序):
  torso, pelvis, L/R shoulder_yaw, L/R elbow, L/R wrist_roll, L/R knee, L/R ankle_roll。
- **方向**:`net_pull_xy_only=false` → 全 3D 各向同性单位向量(默认只水平)。
- **幅值**:U[10,40] N(<5 N 在本体感知死区)。
- **相位状态机**(50 Hz 步数,均匀采样):REST U[20,60] → RAMP-UP U[10,30] →
  HOLD U[40,100] → RAMP-DOWN U[10,30] → 循环。REST 段提供负样本。
- **施加**:每物理子步把世界系力写到所选连杆的 wrench 缓冲
  (`force_apply_net_pull`, `motion_tracking.py:2286-2316`),再对躯干做净
  力/矩限幅。

**真值观测 `net_pull_force_priv`(27 维,`motion_tracking.py:2587-2605`)**:

```
[接触点_b(3) | 力_b/Fmax(3) | 力_w/Fmax(3) | 连杆one-hot(12) | 相位one-hot(4) | 相位计时/250(1) | 幅值/Fmax(1)]
```

相位序:0 rest / 1 ramp_up / 2 hold / 3 ramp_down。

## 9. 模型 ④:Plan A 力传感器 v3(监督 MLP,`wbc_train_v3.py`)

### 9.1 输入:每帧 320 维 × 窗口 W(最佳 W=6 → 网络入 1920 维)

`wbc_input_` 组(`collect.yaml:44-50`),按序拼接:

| 项 | 含义 | 维度 |
|---|---|---|
| applied_torque | PD 实际施加力矩(最强外力线索) | 29 |
| joint_pos_history [0..4] | 关节位置 5 帧滞后 | 145 |
| applied_action | 指令关节目标(→跟踪误差) | 29 |
| root_ang_vel_history [0..4] | IMU 陀螺 5 帧 | 15 |
| projected_gravity_history [0..4] | IMU 倾角 5 帧 | 15 |
| prev_actions ×3 | 已知自身输入(供扣除) | 87 |

**合计 320 维/帧。** 组名尾缀 `_` 使其绕过 VecNorm(`helpers.py:128-141`),
存原始值;归一化训练时做(前 20 万训练帧的 mean/std,W 次平铺,逐 batch 应用,
避免物化第二份 8 GB 拷贝)。

标签 = 上节 27 维 `net_pull_force_priv`(`wbc_label_` 组)。

### 9.2 数据

冻结 stiff 低层策略(不柔顺——柔顺会"卸掉"信号;且它不消费 force_priv,
无输入维冲突)+ net_pull 全身探针。真实数据集 `data/wbc/wbc_train.h5`:
**T=50001 × E=128 = 640 万样本**,Fmax=40 N,HDF5 平铺 `[N,320]`
(行→(env,时间) 在训练器里按 E=128 重构成 `[T,E,320]` GPU 常驻网格)。

### 9.3 模型结构:共享干 + 四头

```
输入 [B, W×320] → MLP 干 [512,512,256](Linear+LayerNorm+ELU)
  ├─ det 头 Linear(256,1):接触有无(logit)
  ├─ loc 头 Linear(256,K):定位(K=5 区域 或 12 连杆;+1 "none" 类)
  ├─ dir 头 Linear(256,3):力单位方向(基座系)
  └─ mag 头 Linear(256,1):力幅值(/Fmax 归一)
```

区域映射(12 连杆→5 区域):torso/pelvis→trunk;shoulder/elbow/wrist→同侧 arm;
knee/ankle→同侧 leg。

### 9.4 损失与训练

- det:**focal BCE**(γ=1.5,pos_weight=负/正,cap 4.0),全帧。
- loc:CE;dir:1−cos;mag:SmoothL1——**三者仅在接触帧上**(mag>0.05 判接触)。
- 总损失 = det + 1.0·loc + 1.0·dir + 1.0·mag。
- AdamW lr 1e-3 wd 1e-5,cosine 退火,40 epochs,batch 16384;
  val = env 0..15 留出(共 128 env);选型分数 = active_acc + det_f1。
- 网格常驻 GPU(~9 GB),1 epoch ≈1 s,12 组 sweep 22.7 min(4090)。

### 9.5 结果与关键教训

最佳 `w6_regions`(W=6,5 区域):active_acc 0.478,det F1 0.587
(rec 0.86/prec 0.45),force_cos 0.349,mag_rel_err 0.77。
臂部召回 ~0.7(干净),腿 0.23–0.26 / 躯干 0.32(弱)——正是可观测性预测的模式。
Sweep 结论:窗口 W1→W6 提升明显后饱和;**区域粒度是最大杠杆**(0.33→0.48);
**标签清洗有害**(保留 ramp 帧训练);
**部署必须用 det 头门控 mag/dir 头**(无接触帧上 mag 头会输出 40–80 N 垃圾)。
这个 0.587 后来被 mjlab 持续接触数据的 0.949 大幅超越——差距在**数据分布**
(瞬时点推 vs 行走持续贴墙),不在模型。

## 10. 模型 ⑤:高层策略(root_ppo 家族)

入口 `bash/train_hl.sh`(`ALGO=root_ppo`,`TASK=G1/G1_hl_ee_compliance`),
低层冻结(`FrozenLowLevelPolicy`,requires_grad=False,MODE 探索)。

### 10.1 HL 指令解码(`HierarchicalRootCommand`)

- **root 指令(5 维)**:`tanh(a)×scale [0.15,0.5,0.5,0.6,0.6]` →
  `[root高度=0.79+Δ, vx, vy, heading_xy]`,写入低层 `policy` obs 的 1:7 切片
  (覆盖其 6 维 command 块)。
- **EE 指令(12 维,`G1_hl_ee_compliance` 用)**:参考腕 6D 位姿 +
  `tanh(a)×0.5` 残差,双腕各 (pos 3 + rot 3)。
- 组合变体:root+EE 17 维、root+feet 11 维。

### 10.2 观测与训练

- **`G1_hl_ee_compliance`(单层特权 teacher)**:obs `hl_policy` 含
  boot/command/**net_pull_force_priv(特权真值力)**/身体高度/踝接触力/
  root 线速度/9 帧历史/关键点/applied_action/torque/足位/腕参考/柔顺状态/
  prev_high_actions;输出 12 维腕残差。奖励主项
  `ee_force_compliance_tracking`(w=5.0,虚拟弹簧 stiffness 200 N/m,
  max_offset 0.25 m,死区 5 N)+ 运动正则。外力:net_pull_ee
  (仅 hand_mimic,5–30 N)。
- **网络(root_ppo)**:actor/critic 均 MLP[512,256],lr 3e-4,clip 0.2,
  entropy 0.005→0.001。
- **teacher–student 变体(root_student_ppo)**:`hl_policy` 只留本体感知,
  特权项(含 net_pull_force_priv)进 `hl_priv` → 编码 128 latent;
  student 用估计器 MLP[512,256]→128 从本体感知预测 latent。
- 其他任务:`G1_hl_root_hold`(root 保持)、`G1_hl_force_resist_*`
  (20–60 N 抗扰,root 弹簧 stiffness 400)。

**Plan B(未开始)**:把模型 ④ 的输出(det 门控的 区域 one-hot+方向+幅值)
替换 HL obs 里的特权 `net_pull_force_priv`,冻结低层训 HL 柔顺——
即把上面 student 变体里的"估计器"换成真正的本体感知力传感器。

---

# 附录

## A. 五个模型一页速查

| # | 模型 | 类型 | 输入 | 输出 | 训练 |
|---|---|---|---|---|---|
| ① | mjlab G1 行走 | RL (PPO) | 99 维(v/ω/重力/q/dq/上步动作/twist 指令) | 29 维关节位置偏移 | rsl_rl,4096 env × 3000 iter,奖励=速度跟踪+姿态−正则 |
| ② | ContactGRU | 监督 | [50,96](q/dq/τ/IMU×3) | 接触 p + (sin,cos) 方位 + 幅值/50 | BCE(pw≤8)+2·MSE+0.5·SmoothL1,8 ep,分钟级 |
| ③ | Isaac 低层跟踪 | RL (PPO, T-S) | policy 257 维 + latent 256 + 关节预测 29 | 29 维关节位置偏移 | teacher 特权→student 蒸馏,MLP[512,512,256] |
| ④ | WBC 力传感器 v3 | 监督 | W×320(τ/q 史/动作/IMU 史) | det + 5 区域 + 3D 方向 + 幅值 | focal BCE+CE+cos+SmoothL1,40 ep,GPU 常驻网格 |
| ⑤ | HL root_ppo | RL (PPO) | hl_policy(含特权力真值) | 5/12/17 维 root/腕残差指令 | 冻结低层,MLP[512,256],lr 3e-4 |
| — | Pledge 导航 | 规则状态机 | 12 世界 bin + yaw + (x,y) | (vx,vy,wz) | 无训练;2D 运动学验证 96–98% |

## B. 两条线的坐标系与门控约定(易错点)

- 去抖在**世界系** 12×30° bin(墙不随机器人转;30° 让轴对齐墙落 bin 中心)。
- 传感器方位标签是**机器人系**(可部署),部署时 +IMU yaw 恢复世界系。
- mag/dir 头只在接触帧有监督 → **部署一律用 det 头门控**。
- 采集要用 **stiff** 低层(柔顺会洗掉本体感知信号)。
- ContactGRU 特征不含指令/动作;WBC v3 特征含 applied_action/prev_actions
  ——前者为跨控制器可部署,后者为单控制器最大化信息,两种取舍并存。

## C. 复现命令

见 `docs/overnight_experiment_report.md` §9(mjlab 线)与 README(Isaac 线:
`bash/train.sh` 低层、`bash/train_hl*.sh` 高层、
`scripts/collect_force_data.py` → `scripts/wbc_sweep_v3.py` Plan A)。
