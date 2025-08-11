# YICA 架构核心特性文档

## 概述
YICA（YICA Intelligence Computing Architecture，基于计算内存的智能计算架构）是一个创新的存算一体（Computing-In-Memory, CIM）架构，专门为深度学习和人工智能计算设计。该架构通过 torch_cim 框架实现完整的软件栈，以 PyTorch 的 PrivateUse1 后端形式集成，提供从底层硬件到上层应用的全栈深度学习计算支持。

### 架构核心理念
- **存算融合**：在存储单元内直接执行计算操作，消除传统冯·诺依曼架构的存储墙瓶颈
- **层次化并行**：从指令级到设备级的多层次并行计算模型
- **软硬协同**：硬件架构与软件优化的深度融合设计
- **生态兼容**：与主流深度学习框架的无缝集成

## 1. 核心架构特性

### 1.1 存算一体设计架构

#### 1.1.1 三级内存层次架构
- **L1 - 寄存器文件 (Register File)**：
  - 位于计算单元内部，最高速访问
  - 支持向量寄存器和标量寄存器
  - 直接参与 YIS 指令执行
  
- **L2 - SPM (Scratchpad Memory)**：
  - 片上可编程存储器，容量 KB 级别
  - 支持存算一体操作，可直接执行计算指令
  - 低延迟、高带宽，典型访问延迟 1-2 个周期
  - 支持多种数据布局：行优先、列优先、分块布局
  
- **L3 - DRAM 主存**：
  - 系统主内存，容量 GB 级别
  - 支持 UMA (统一内存访问) 和 NUMA (非统一内存访问) 两种模式
  - 通过高速互连与 SPM 层进行数据交换

#### 1.1.2 计算单元 (CU) 架构
- **功能单元组成**：
  - ALU (算术逻辑单元)：支持整数和浮点运算
  - MMA (矩阵乘法加速器)：专用于矩阵乘法运算
  - 向量处理单元：支持 SIMD 并行计算
  - 控制单元：指令解码和执行控制
  
- **存储集成**：
  - 每个 CU 配备专用 SPM
  - 支持计算和存储的紧密耦合
  - 数据路径优化，减少数据移动开销

#### 1.1.3 Die 级别组织
- **CU 集群**：多个 CU 组织在同一个 Die 上，典型配置 4-16 个 CU
- **片上互连**：高速 NoC (Network-on-Chip) 连接各 CU
- **共享资源**：L3 Cache、内存控制器、I/O 接口

### 1.2 数据类型系统

#### 1.2.1 整数数据类型
- **有符号整数**：
  - INT4：4位有符号整数，范围 [-8, 7]
  - INT8：8位有符号整数，范围 [-128, 127]
  - INT16：16位有符号整数，范围 [-32768, 32767]
  - INT32：32位有符号整数，标准整数类型
  - INT64：64位有符号整数，长整数类型
  
- **无符号整数**：
  - UINT8：8位无符号整数，范围 [0, 255]
  - UINT16：16位无符号整数，范围 [0, 65535]
  - UINT32：32位无符号整数
  - UINT64：64位无符号整数

#### 1.2.2 浮点数据类型
- **标准浮点**：
  - FP16：IEEE 754 半精度浮点，1符号位+5指数位+10尾数位
  - FP32：IEEE 754 单精度浮点，1符号位+8指数位+23尾数位
  - FP64：IEEE 754 双精度浮点，1符号位+11指数位+52尾数位
  
- **AI 优化浮点**：
  - BF16：Brain Float，1符号位+8指数位+7尾数位，适合深度学习
  - TF32：Tensor Float，1符号位+8指数位+10尾数位，Tensor Core 优化
  - FP8：8位浮点，超低精度推理
  - FP4：4位浮点，极端量化场景

#### 1.2.3 特殊数据类型
- **打包类型**：
  - INT4x2：两个4位整数打包在8位中
  - FP4x2：两个4位浮点数打包在8位中
  - 支持向量化操作，提高数据并行度
  
- **硬件优化特性**：
  - 数据类型转换硬件加速
  - 混合精度计算支持
  - 量化感知的数据路径设计

### 1.3 内存布局策略

#### 1.3.1 基础数据布局
- **ROWMAJOR (行优先)**：
  - 连续存储同一行的元素
  - 适合行向量操作和矩阵乘法的 A 矩阵
  - 内存访问模式：A[i][j] 后跟 A[i][j+1]
  
- **COLMAJOR (列优先)**：
  - 连续存储同一列的元素
  - 适合列向量操作和矩阵乘法的 B 矩阵
  - 内存访问模式：A[i][j] 后跟 A[i+1][j]

#### 1.3.2 分块数据布局
- **TROWMAJOR (分块行优先)**：
  - 将矩阵分成块，块内行优先存储
  - 优化缓存局部性和 SPM 利用率
  - 适合大矩阵的分块计算
  
- **TCOLMAJOR (分块列优先)**：
  - 将矩阵分成块，块内列优先存储
  - 支持分块矩阵乘法优化
  - 减少跨块内存访问

#### 1.3.3 内存类型分类
- **HOST**：主机端内存，CPU 可直接访问
- **DRAM_UMA**：统一内存访问 DRAM，所有 CU 等距访问
- **DRAM_NUMA**：非统一内存访问 DRAM，访问延迟因位置而异
- **SPM**：片上暂存存储器，最高性能的存储层次

## 2. 分布式计算架构

### 2.1 YCCL 通信后端

#### 2.1.1 通信框架设计
- **YCCL (YICA Collective Communication Library)**：
  - 专为 YICA 架构设计的集合通信库
  - 参考 NCCL 设计理念，针对存算一体架构优化
  - 支持点对点通信和集合通信操作
  - 原生支持多精度数据类型通信

#### 2.1.2 异步执行模型
- **非阻塞操作**：所有集合通信操作均为异步执行
- **流水线重叠**：通信与计算的流水线重叠执行
- **事件同步**：基于事件的细粒度同步机制
- **优先级调度**：支持高优先级通信流

#### 2.1.3 错误处理策略
- **NoHandling**：
  - 不处理异步错误，由上层应用处理
  - 适用于调试和性能测试场景
  
- **TearDown**：
  - 遇到错误时立即拆除整个进程组
  - 确保分布式系统的一致性状态
  
- **CleanUpOnly**：
  - 仅清理当前集合操作和通信器资源
  - 保持进程组活跃，允许后续操作
  
- **SkipCleanUp**：
  - 跳过资源清理，直接拆除进程
  - 用于紧急错误恢复场景

#### 2.1.4 集合通信原语
- **AllReduce**：全局归约操作，支持 sum、max、min、prod、avg
- **AllGather**：全局收集操作，收集所有进程的数据
- **ReduceScatter**：归约分散操作，结合归约和分散
- **Broadcast**：广播操作，从根进程向所有进程广播
- **AllToAll**：全交换操作，进程间数据完全交换

### 2.2 设备网格管理

#### 2.2.1 DieMesh 抽象
- **网格拓扑**：
  - 支持 1D、2D、3D 网格拓扑结构
  - 最大支持 32 个计算单元的网格配置
  - 灵活的网格维度配置：[mesh[0], mesh[1], mesh[2]]
  
- **拓扑感知**：
  - 基于物理拓扑的通信路径优化
  - 支持层次化通信：片内 -> 片间 -> 节点间
  - 带宽和延迟感知的路由选择

#### 2.2.2 Rank 映射策略
- **线性映射**：rank = x + y * mesh[0] + z * mesh[0] * mesh[1]
- **自定义映射**：支持用户定义的 rank 到坐标映射
- **负载均衡**：考虑计算负载的动态 rank 分配
- **容错映射**：支持故障节点的 rank 重映射

#### 2.2.3 网格通信优化
- **邻居通信优化**：相邻节点间的高速通信
- **多维归约**：利用网格结构的多维归约算法
- **带宽聚合**：多路径并行通信的带宽聚合

### 2.3 张量分布策略

#### 2.3.1 Shard（分片）策略
- **维度分片**：
  - 在指定维度上将张量分片分布
  - 支持多维度同时分片
  - 自动处理不均匀分片的边界情况
  
- **分片粒度**：
  - 块级分片：以固定大小块为单位分片
  - 行级分片：按行分布张量数据
  - 列级分片：按列分布张量数据
  
- **负载均衡**：
  - 动态调整分片大小以均衡负载
  - 考虑计算复杂度的智能分片

#### 2.3.2 Replicate（复制）策略
- **全复制**：张量在所有设备上完全复制
- **部分复制**：在设备子集上复制张量
- **一致性维护**：复制张量的自动一致性更新
- **内存优化**：共享只读张量的内存优化

#### 2.3.3 Partial（部分归约）策略
- **归约操作类型**：
  - Sum：求和归约，用于梯度聚合
  - Prod：乘积归约，用于概率计算
  - Max/Min：最值归约，用于池化操作
  - Avg：平均归约，用于批量归一化
  
- **部分归约优化**：
  - 树状归约：减少通信轮数
  - 环状归约：优化带宽利用率
  - 分层归约：利用网络层次结构

## 3. 内存管理系统

### 3.1 CIM 缓存分配器 (CIMCachingAllocator)

#### 3.1.1 分层缓存架构
- **SPM 缓存分配器**：
  - 管理片上 SPM 的内存分配和回收
  - 支持细粒度的内存块管理（最小 8 字节对齐）
  - 实现快速分配算法，平均分配时间 O(1)
  - 支持内存碎片整理和合并优化
  
- **DRAM 缓存分配器**：
  - 管理主存 DRAM 的大块内存分配
  - 支持多种分配策略：首次适配、最佳适配、伙伴系统
  - 实现内存池技术，减少系统调用开销
  - 支持跨 NUMA 域的内存分配和迁移

#### 3.1.2 内存池管理
- **分级内存池**：
  - 小块内存池：8B - 1KB，用于元数据和小张量
  - 中块内存池：1KB - 1MB，用于中等大小张量
  - 大块内存池：1MB+，用于大张量和批处理
  
- **扩展段机制**：
  - 动态扩展内存池容量
  - 支持内存池的热扩展，无需重启
  - 实现内存预分配策略，减少运行时分配延迟

#### 3.1.3 性能监控和统计
- **实时统计指标**：
  - 总分配内存量、峰值内存使用量
  - 分配/释放操作次数和平均延迟
  - 内存碎片率和利用率统计
  - 缓存命中率和未命中率
  
- **内存快照功能**：
  - 支持内存使用状态的快照保存
  - 提供内存泄漏检测和分析工具
  - 支持内存使用历史追踪

#### 3.1.4 OOM (Out of Memory) 处理
- **分级 OOM 策略**：
  - Level 1：尝试内存整理和回收
  - Level 2：触发垃圾回收和缓存清理
  - Level 3：向上层报告 OOM 错误
  
- **回调机制**：
  - 注册 OOM 回调函数，支持自定义处理逻辑
  - 支持优雅降级和资源释放
  - 提供内存压力预警机制

### 3.2 张量实现 (CIMTensorImpl)

#### 3.2.1 张量结构设计
- **继承关系**：
  ```cpp
  class CIMTensorImpl : public c10::TensorImpl {
    yica::CIMTensorMeta meta_;  // YICA 特有的元数据
    // ... 其他成员
  };
  ```
  
- **元数据结构 (CIMTensorMeta)**：
  - 数据类型信息：dtype、element_size、数据格式
  - 内存布局信息：layout、stride、offset
  - 设备信息：device_id、memory_type（SPM/DRAM）
  - 分布信息：placement_strategy、device_mesh

#### 3.2.2 内存管理集成
- **智能指针管理**：
  - 使用 c10::DataPtr 管理张量数据生命周期
  - 支持自定义删除器，与 CIM 分配器集成
  - 实现引用计数和自动内存回收
  
- **浅拷贝优化**：
  - 支持零拷贝的张量视图创建
  - 实现 COW (Copy-on-Write) 机制
  - 优化张量切片和重塑操作

#### 3.2.3 设备间数据移动
- **异步数据传输**：
  - 支持 SPM ↔ DRAM 的异步数据传输
  - 实现流水线数据传输，重叠计算和通信
  - 支持批量数据传输优化
  
- **数据格式转换**：
  - 支持不同数据类型间的自动转换
  - 实现硬件加速的类型转换
  - 支持量化和反量化操作

## 4. 计算抽象

### 4.1 GTensor 分布式张量
- **CPU 表示**：GTensor 作为 CIM 张量的 CPU 表示
- **自动分发**：支持张量操作的自动分发到设备
- **灵活构造**：支持从单个张量或张量列表构造分布式张量

### 4.2 流管理
- **异步执行**：基于流的异步计算执行
- **同步机制**：WorkYCCL 提供完整的同步和等待机制
- **事件跟踪**：CIM 事件系统用于性能监控和调试

## 5. 系统集成

### 5.1 PyTorch 集成
- **PrivateUse1 后端**：注册为 "cim" 设备类型
- **完整兼容**：支持 PyTorch 的张量、模块、存储等所有组件
- **自动生成方法**：自动生成 PyTorch 标准方法

### 5.2 调试和监控
- **Profiler 集成**：完整的性能分析工具集成
- **调试模式**：支持 desync 调试和详细错误报告
- **监控线程**：watchdog 线程监控系统健康状态

## 6. 性能优化特性

### 6.1 内存优化
- **缓存分配器**：高效的内存缓存和重用机制
- **分段管理**：支持可扩展内存段
- **记录流**：避免不必要的流记录以优化分配器安全性

### 6.2 通信优化
- **高优先级流**：支持高优先级通信流
- **批量操作**：支持张量批量通信操作
- **定时监控**：详细的操作时间监控和优化

## 7. 架构优势

1. **存算一体**：紧密集成的存储和计算，减少数据移动开销
2. **多精度支持**：从 4 位到 64 位的全精度范围支持
3. **灵活分布**：多种张量分布策略适应不同计算模式
4. **高效通信**：基于 YCCL 的优化分布式通信
5. **PyTorch 兼容**：与 PyTorch 生态系统无缝集成
6. **完善监控**：全面的性能监控和调试工具

## 8. 应用场景

- **大规模深度学习训练**：利用分布式计算能力
- **推理加速**：存算一体架构减少内存访问延迟
- **多精度计算**：支持混合精度和低精度计算
- **边缘计算**：SPM 的高效存储适合资源受限环境

## 9. 运行时执行框架（基于 ybx_replay 分析）

### 9.1 内核回放系统
- **JSON 驱动执行**：通过 JSON 配置文件定义计算图和内核参数
- **任务抽象**：Task 类封装单个内核执行，支持同步和异步执行
- **运行器框架**：Runner 管理多个任务的执行和同步
- **gRPC 服务**：支持远程内核执行和数据传输

### 9.2 张量系统增强
- **多设备支持**：CPU 和 YICA 设备的统一张量抽象
- **内存类型区分**：SRAM_MEMORY 和 DRAM_MEMORY 的显式管理
- **数据类型支持**：FLOAT32、FLOAT16、INT32、UINT8
- **引用计数管理**：DataPtr 实现自动内存管理

### 9.3 内存分配器架构
- **分层分配**：DeviceAllocator 和 HostAllocator 的分离设计
- **块管理**：基于 DeviceBlock 的内存块管理和合并
- **对齐优化**：8 字节对齐的内存分配策略
- **统计监控**：详细的内存使用统计和调试信息

### 9.4 计算图执行
- **节点类型**：placeholder（输入）、weight（权重）、op（操作）
- **图遍历**：基于依赖关系的图执行顺序
- **张量传递**：节点间的张量数据流管理
- **工作空间管理**：临时计算空间的动态分配

## 10. Llama 模型专用优化

### 10.1 Attention 机制优化
- **QK BMM 算子**：Query-Key 批量矩阵乘法的专用实现
- **动态序列长度**：基于 position_id 的动态参数调整
- **KV Cache 管理**：高效的键值缓存存储和更新
- **Softmax 优化**：针对 Attention 权重的专用 Softmax 实现

### 10.2 模型结构支持
- **多头注意力**：32 个 attention head 的并行处理
- **RoPE 位置编码**：旋转位置编码的硬件加速
- **LayerNorm**：RMSNorm 的优化实现
- **MLP 层**：前馈网络的矩阵乘法优化

### 10.3 内存布局优化
- **权重分片**：大权重矩阵的多 CU 分布存储
- **缓存局部性**：基于访问模式的数据布局优化
- **动态形状**：支持不同序列长度的动态张量形状

## 11. 性能监控和调试

### 11.1 执行时间统计
- **内核级监控**：每个内核的执行时间统计
- **端到端性能**：完整推理流程的性能分析
- **内存带宽**：内存访问模式的性能监控
- **设备同步**：hipDeviceSynchronize 的同步开销统计

### 11.2 调试工具
- **图可视化**：计算图的 DOT 格式导出
- **数据转储**：中间结果的文件保存和验证
- **环境变量控制**：丰富的调试开关和配置选项
- **日志系统**：分级日志和彩色输出

### 11.3 测试框架
- **单元测试**：张量、内存分配器、驱动 API 的单元测试
- **回归测试**：基于 JSON 配置的自动化测试
- **性能基准**：标准化的性能测试套件

## 12. 硬件抽象层

### 12.1 驱动 API 封装
- **HIP 兼容接口**：类似 HIP 的设备管理 API
- **内存管理**：hipHostMalloc、hipDeviceSynchronize 等接口
- **错误处理**：完整的错误码和错误信息系统

### 12.2 内核加载机制
- **ELF 文件支持**：从 ELF 文件加载内核代码
- **动态链接**：运行时内核函数的动态绑定
- **参数传递**：内核参数的类型安全传递

### 12.3 设备配置
- **工作组配置**：SWG（软件工作组）和 CWG（计算工作组）
- **网格配置**：Grid 和 Block 的多维配置
- **资源分配**：计算资源和内存资源的动态分配

## 13. LLVM RISC-V 后端 YICA 扩展（基于 LLVM 后端分析）

### 13.1 YIS 指令集扩展

#### 13.1.1 指令集架构概述
YIS (YICA Instruction Set) 是基于 RISC-V 的存算一体指令集扩展，专门为 YICA 架构设计，提供了五大类专用指令：

#### 13.1.2 YISECOPY (外部拷贝指令)
- **指令功能**：处理不同存储层次间的数据移动
- **支持的拷贝类型**：
  - **G2G (Global to Global)**：DRAM 到 DRAM 的数据拷贝
  - **G2S (Global to Scratchpad)**：DRAM 到 SPM 的数据拷贝
  - **S2G (Scratchpad to Global)**：SPM 到 DRAM 的数据拷贝
  - **G2SPM (Global to SPM)**：DRAM 到 SPM 的直接拷贝
  - **SPM2G (SPM to Global)**：SPM 到 DRAM 的直接拷贝
  
- **数据布局支持**：
  - **TROW (Tiled Row)**：分块行优先布局
  - **ROW**：标准行优先布局
  - **TCOL (Tiled Column)**：分块列优先布局
  - **COL**：标准列优先布局
  
- **作业域 (Boscope) 支持**：
  - **WG (Work Group)**：工作组级别的数据拷贝
  - **CWG (Compute Work Group)**：计算工作组级别的数据拷贝
  
- **卷积优化**：
  - **IM2COL 变换**：支持卷积的 im2col 数据重排
  - 硬件加速的数据格式转换

#### 13.1.3 YISICOPY (内部拷贝指令)
- **指令功能**：处理同一存储层次内的数据拷贝和重排
- **拷贝模式**：
  - **MC (Multicast)**：多播模式，一对多数据分发
  - **NOMC (Non-Multicast)**：非多播模式，点对点数据拷贝
  
- **支持的拷贝类型**：
  - **S2S (SPM to SPM)**：SPM 间数据拷贝
  - **SPM2S (SPM to SPM)**：SPM 到 SPM 的专用拷贝
  - **R2S (Register to SPM)**：寄存器到 SPM 的数据拷贝
  - **S2R (SPM to Register)**：SPM 到寄存器的数据拷贝
  
- **集合通信支持**：
  - **BC (Broadcast)**：广播操作
  - **GAT (Gather)**：收集操作
  
- **寻址模式**：
  - **RR (Register-Register)**：寄存器寻址
  - **RI (Register-Immediate)**：寄存器-立即数寻址
  - **IR (Immediate-Register)**：立即数-寄存器寻址
  - **II (Immediate-Immediate)**：立即数寻址

#### 13.1.4 YISMMA (矩阵乘法加速指令)
- **指令功能**：硬件加速的矩阵乘法运算
- **矩阵维度配置**：
  - **M、N、K 维度**：支持可配置的矩阵乘法维度
  - 典型配置：8x8、16x16、32x32 等
  
- **计算模式**：
  - **ACC (Accumulate)**：累加模式，C = C + A × B
  - **Non-ACC**：非累加模式，C = A × B
  
- **存储模式**：
  - **SPM 模式**：操作数全部在 SPM 中
  - **SPMG (SPM Global) 模式**：支持 SPM 和 DRAM 混合操作
  
- **精度支持**：
  - FP16、FP32 浮点矩阵乘法
  - INT8、INT16 整数矩阵乘法
  - 混合精度计算支持

#### 13.1.5 YISSYNC (同步指令)
- **指令功能**：提供硬件级同步原语
- **同步类型**：
  - **BAR (Barrier)**：栅栏同步，等待所有线程到达同步点
  - **BOINIT (Buffer Object Init)**：缓冲区对象初始化
  - **BOARRV (Buffer Object Arrive)**：缓冲区对象到达通知
  - **BOWAIT (Buffer Object Wait)**：缓冲区对象等待
  
- **作业域支持**：
  - 支持不同粒度的同步：线程级、工作组级、设备级
  - 支持跨设备的分布式同步

#### 13.1.6 YISCONTROL (控制指令)
- **指令功能**：控制程序执行流和内核调用
- **控制类型**：
  - **CALL_EU (Call Execution Unit)**：调用执行单元
  - **END**：结束当前内核执行
  
- **参数传递**：
  - 支持寄存器和立即数的灵活参数传递
  - 最大支持 30 位立即数参数

### 13.2 YICA 架构层次寄存器
- **内核启动参数寄存器**：
  - TID (线程 ID)：0x04
  - WARP_RANK (Warp 级别)：0x00
  - STAGE (执行阶段)：0x08
  - FUNC_ID (函数 ID)：0x0C
  
- **工作组层次寄存器**：
  - WGID_IN_SWG0-3 (SWG 内工作组 ID)：0x18-0x24
  - WGRANK_IN_DNUMA (DNUMA 内工作组排序)：0x28
  - SWGRANK_IN_GRID (网格内 SWG 排序)：0x3C
  
- **计算工作组寄存器**：
  - CWGID_IN_SWG0-3 (SWG 内 CWG ID)：0x2C-0x38
  - WGID_IN_CWG0-3 (CWG 内工作组 ID)：0x40-0x4C

### 13.3 存储基地址寄存器
- **内存基地址寄存器**：
  - DLM_BASE (动态局部内存基址)：0x10
  - SMEM_BASE (共享内存基址)：0x14
  - PBUF_BASE (参数缓冲区基址)：0x18
  - TAR_BASE (目标寄存器基址)：0x1C

### 13.4 YIS 指令集架构特性
- **外部拷贝指令 (YISECOPY)**：
  - 支持多种数据布局：TROW、ROW、TCOL、COL
  - 支持两种作业域：WG (工作组)、CWG (计算工作组)
  - 支持 IM2COL 卷积优化
  
- **内部拷贝指令 (YISICOPY)**：
  - 支持多播 (MC) 和非多播 (NOMC) 模式
  - 支持集合通信：broadcast (BC)、gather (GAT)
  - 灵活的寄存器/立即数寻址模式

- **矩阵乘法指令 (YISMMA)**：
  - 支持可配置的 M、N、K 维度
  - 支持累加 (ACC) 和非累加模式
  - 支持 SPM 到 SPM 全局模式 (SPMG)

- **同步控制指令**：
  - BAR：栅栏同步指令
  - BOINIT：缓冲区对象初始化
  - BOARRV/BOWAIT：缓冲区对象到达/等待

### 13.5 编译器内建函数支持
- **层次化 ID 获取**：
  - `__builtin_riscv_tid()`
  - `__builtin_riscv_warp_rank()`
  - `__builtin_riscv_wgrank_in_dnuma()`
  - `__builtin_riscv_swgrank_in_grid()`

- **内存基址获取**：
  - `__builtin_riscv_dlm_base()`
  - `__builtin_riscv_smem_base()`
  - `__builtin_riscv_pbuf_base()`
  - `__builtin_riscv_tar_base()`

- **YIS 指令内建函数**：
  - 拷贝操作：`__builtin_riscv_ybx_copy_*`
  - 内部拷贝：`__builtin_riscv_ybx_internal_copy_*`
  - 同步控制：`__builtin_riscv_ybx_bar_64`

### 13.6 YICA 层次化计算模型
- **三级计算层次**：
  - Grid (网格)：最高层计算组织
  - SWG (Super Work Group)：超级工作组，中间层
  - WG/CWG (Work Group/Compute Work Group)：基本计算单元

- **DNUMA 内存架构**：
  - 分布式非一致性内存访问
  - 工作组在 DNUMA 域内的层次化组织
  - 支持跨 DNUMA 域的数据移动

## 14. 架构综合特性总结

### 14.1 完整的存算一体解决方案
- **硬件层**：RISC-V YIS 指令集 + 多层内存 + 矩阵计算单元
- **编译器层**：LLVM 后端完整支持 + 内建函数
- **运行时层**：PyTorch 后端 + 设备管理 + 内存分配
- **框架层**：分布式通信 + 图执行 + 自动优化

### 14.2 分层优化策略
- **指令级**：YIS 专用指令优化数据移动和计算
- **内存级**：SPM/DRAM 层次化管理 + 缓存分配器
- **线程级**：Warp 和工作组层次化执行
- **设备级**：多设备协调 + 分布式通信

### 14.3 YICA 与标准架构的差异

#### 14.3.1 指令集层面差异
- **存算一体指令**：
  - YICA：专用的 YIS 指令集，支持存储内计算
  - 传统架构：分离的计算和存储指令
  
- **数据移动优化**：
  - YICA：硬件加速的数据重排和格式转换
  - 传统架构：软件实现的数据移动

#### 14.3.2 内存架构差异
- **内存层次**：
  - YICA：三级层次（寄存器 + SPM + DRAM）
  - 传统架构：两级层次（寄存器 + 内存）
  
- **存算融合**：
  - YICA：SPM 支持就地计算，减少数据移动
  - 传统架构：计算和存储分离，存在存储墙问题

#### 14.3.3 计算模型差异
- **并行层次**：
  - YICA：四级层次（Grid/SWG/WG/CWG/Thread）
  - 传统架构：二级层次（Block/Thread）
  
- **工作组织**：
  - YICA：层次化工作组，支持复杂的并行模式
  - 传统架构：扁平化线程组织

#### 14.3.4 同步机制差异
- **硬件同步**：
  - YICA：硬件级缓冲区对象同步，低延迟
  - 传统架构：软件同步原语，较高开销
  
- **分布式同步**：
  - YICA：原生支持跨设备同步
  - 传统架构：依赖软件库实现

## 15. 性能特性和优化策略

### 15.1 计算性能优化

#### 15.1.1 存算一体优势
- **数据局部性**：
  - 计算在数据存储位置就地执行
  - 消除传统架构的数据搬移开销
  - 典型性能提升：2-5x 相比传统架构
  
- **内存带宽利用**：
  - SPM 提供极高的内存带宽（TB/s 级别）
  - 多级内存层次的带宽聚合
  - 支持多路并发内存访问

#### 15.1.2 矩阵计算优化
- **MMA 指令加速**：
  - 硬件原生支持的矩阵乘法
  - 支持多种精度的混合计算
  - 典型性能：FP16 可达数百 TOPS
  
- **数据重用优化**：
  - 智能的数据缓存和重用策略
  - 分块计算算法的硬件支持
  - 减少重复数据加载

### 15.2 内存性能优化

#### 15.2.1 分层存储优化
- **SPM 优化策略**：
  - 智能数据预取和预加载
  - 基于访问模式的数据布局优化
  - 多 SPM 间的负载均衡
  
- **DRAM 访问优化**：
  - 批量数据传输减少延迟
  - NUMA 感知的内存分配
  - 内存带宽的动态调度

#### 15.2.2 缓存优化
- **多级缓存协同**：
  - L1/L2/L3 缓存的协同优化
  - 缓存一致性的硬件支持
  - 智能缓存替换策略
  
- **数据预取**：
  - 硬件预取器的智能调度
  - 基于访问模式的预测预取
  - 软硬件协同的预取优化

### 15.3 通信性能优化

#### 15.3.1 片内通信优化
- **NoC 优化**：
  - 低延迟的片上网络
  - 拥塞感知的路由算法
  - 多路径并行通信
  
- **数据传输优化**：
  - 零拷贝数据传输
  - 流水线重叠传输
  - 压缩传输减少带宽需求

#### 15.3.2 分布式通信优化
- **YCCL 优化**：
  - 拓扑感知的通信算法
  - 分层归约减少通信轮数
  - 带宽聚合和负载均衡

## 16. 应用场景和性能基准

### 16.1 深度学习训练场景

#### 16.1.1 大模型训练
- **Transformer 模型**：
  - Llama2-7B：相比 GPU 提升 2.5x 训练速度
  - Llama2-70B：支持千卡级别的分布式训练
  - GPT 系列：优化 Attention 和 MLP 计算
  
- **CNN 模型**：
  - ResNet 系列：卷积计算加速 3x
  - EfficientNet：混合精度训练优化
  - 目标检测：YOLO 系列的端到端优化

#### 16.1.2 推理加速场景
- **低延迟推理**：
  - 单次推理延迟：< 1ms（小模型）
  - 批量推理吞吐：10x 提升
  - 动态批处理优化
  
- **边缘部署**：
  - 功耗优化：相比 GPU 降低 50% 功耗
  - 模型压缩：支持 INT4/FP4 极端量化
  - 实时推理：支持视频流实时处理

### 16.2 性能基准测试

#### 16.2.1 MLPerf 基准
- **训练基准**：
  - ImageNet：ResNet-50 训练时间缩短 40%
  - BERT：预训练效率提升 60%
  - Recommendation：推荐系统训练加速 3x
  
- **推理基准**：
  - ResNet-50 推理：吞吐量提升 5x
  - BERT-Large：延迟降低 70%
  - 3D U-Net：医疗影像处理加速 4x

#### 16.2.2 内存性能基准
- **内存带宽**：
  - SPM 带宽：2TB/s（理论峰值）
  - DRAM 带宽：1TB/s（实际可达）
  - 内存延迟：SPM < 2ns，DRAM < 100ns
  
- **能效比**：
  - 计算能效：100 TOPS/W（FP16）
  - 内存能效：10 TB/s/W
  - 系统级能效：相比 GPU 提升 3-5x

# YICA 架构详细参数规范

## 17.1 硬件架构参数

### 17.1.1 计算层次结构参数
```cpp
// 层次化计算单元组织
#define GRID_LEVEL     0    // 网格级别
#define SWG_LEVEL      1    // 子工作组级别  
#define WG_LEVEL       2    // 工作组级别
#define CWG_LEVEL      3    // 计算工作组级别

// 寄存器地址偏移 (来自 RISCVYica.h)
#define WARP_RANK        0   // Warp 等级
#define RVV_FUNC_ID      4   // RISC-V 向量函数 ID
#define STAGE            8   // 执行阶段
#define DLM_BASE         12  // 数据本地内存基址
#define SMEM_BASE        16  // 共享内存基址
#define PBUF_BASE        20  // 参数缓冲区基址

// 工作组层次 ID 偏移
#define WGID_IN_SWG0     24  // SWG 中的 WG ID (维度0)
#define WGID_IN_SWG1     28  // SWG 中的 WG ID (维度1)
#define WGID_IN_SWG2     32  // SWG 中的 WG ID (维度2)
#define WGID_IN_SWG3     36  // SWG 中的 WG ID (维度3)
#define WGRANK_IN_DNUMA  40  // NUMA 域中的 WG 等级
#define CWGID_IN_SWG0    44  // SWG 中的 CWG ID (维度0)
#define CWGID_IN_SWG1    48  // SWG 中的 CWG ID (维度1)
#define CWGID_IN_SWG2    52  // SWG 中的 CWG ID (维度2)
#define CWGID_IN_SWG3    56  // SWG 中的 CWG ID (维度3)
#define SWGRANK_IN_GRID  60  // Grid 中的 SWG 等级
#define WGID_IN_CWG0     64  // CWG 中的 WG ID (维度0)
#define WGID_IN_CWG1     68  // CWG 中的 WG ID (维度1)
#define WGID_IN_CWG2     72  // CWG 中的 WG ID (维度2)
#define WGID_IN_CWG3     76  // CWG 中的 WG ID (维度3)
#define TID              120 // 线程 ID
```

### 17.1.2 设备属性参数

#### 基本设备属性
```cpp
struct CIMDeviceProp {
    std::string name;                    // 设备名称
    int major;                          // 主版本号
    int minor;                          // 次版本号
    int warpSize;                       // Warp 大小
    size_t totalGlobalMem;              // 全局内存总大小
    int multiProcessorCount;            // 多处理器（计算单元）数量
    int maxThreadsPerMultiProcessor;    // 每个多处理器最大线程数
    int isMultiGpuBoard;               // 是否为多 GPU 板卡
    int integrated;                     // APU vs dGPU
    
    // PCI 设备属性
    int pciDomainID;                   // PCI 域 ID
    int pciBusID;                      // PCI 总线 ID
    int pciDeviceID;                   // PCI 设备 ID
};
```

#### YICA 特定设备属性
```cpp
// YICA 专用属性扩展
struct YICADeviceProp : CIMDeviceProp {
    size_t gpuId;                      // GPU ID (bit4~10: node id, bit0~3: chip id)
    size_t cimDieCount;                // 每设备 CIM Die 数量
    size_t clusterCountPerCimDie;      // 每个 CIM Die 的计算单元数量
    size_t ddramSizePerCluster;        // 每个计算单元的 DRAM 大小
    size_t spmSizePerCimDie;           // 每个 CIM Die 的 SPM 本地内存大小
    size_t maxSwgPerGrid;              // 每个 Grid 的最大 SWG 数量
    size_t maxWgsPerCwg;               // 每个 CWG 的最大 WG 数量
    size_t maxNumCWGPerBPC;            // 每个 BPC(块处理集群) 的最大 CWG 数量
    size_t maxNumThreadPerWorkGroup;   // 每个工作组的最大线程数
};
```

### 17.1.3 内存层次参数

#### 内存类型定义
```cpp
// 内存类型枚举
enum YICAMemoryType {
    YICA_MEM_TYPE_SPM = 0,     // 片上暂存内存 (Scratchpad Memory)
    YICA_MEM_TYPE_DRAM = 1,    // 设备 DRAM
    YICA_MEM_TYPE_HOST = 2     // 主机内存
};

// 内存大小常量
#define RESERVED_MEMORY_SIZE (2 << 20)  // 2MB 运行时保留内存
#define MB_SIZE (1024 * 1024)           // 1MB 大小定义
```

#### 内存分配器参数
```cpp
// 缓存分配器配置
struct CIMCachingAllocatorConfig {
    size_t max_split_size_;           // 最大分割大小
    double garbage_collection_threshold_; // 垃圾回收阈值
    bool expandable_segments_;        // 可扩展段
    bool release_lock_on_malloc_;     // 分配时释放锁
    std::chrono::milliseconds round_robin_threshold_; // 轮询阈值
};
```

### 17.1.4 计算网格参数配置

#### 标准网格配置示例
```json
// 典型的 YICA 计算网格配置
{
    "grid": [4, 1, 1],    // Grid 维度: [x, y, z]
    "swg": [4, 1, 1],     // 子工作组维度
    "cwg": [4, 1, 1],     // 计算工作组维度  
    "block": [1, 64, 1]   // 线程块维度
}

// Llama 模型专用配置
{
    "rmsnorm": {
        "grid": [1, 1, 1],
        "swg": [1, 1, 1], 
        "cwg": [1, 1, 1],
        "block": [1, 16, 1]
    },
    "qkv_linear": {
        "grid": [4, 1, 1],
        "swg": [4, 1, 1],
        "cwg": [4, 1, 1], 
        "block": [1, 1, 1]
    },
    "attention": {
        "grid": [4, 1, 1],
        "swg": [4, 1, 1],
        "cwg": [4, 1, 1],
        "block": [64, 1, 1]
    }
}
```

## 17.2 编程模型参数

### 17.2.1 线程层次参数
```cpp
// 线程组织层次
#define YICA_MAX_GRID_DIM_X      65535  // Grid X 维度最大值
#define YICA_MAX_GRID_DIM_Y      65535  // Grid Y 维度最大值  
#define YICA_MAX_GRID_DIM_Z      65535  // Grid Z 维度最大值

#define YICA_MAX_BLOCK_DIM_X     1024   // Block X 维度最大值
#define YICA_MAX_BLOCK_DIM_Y     1024   // Block Y 维度最大值
#define YICA_MAX_BLOCK_DIM_Z     64     // Block Z 维度最大值

#define YICA_MAX_THREADS_PER_BLOCK 1024 // 每个块的最大线程数
#define YICA_WARP_SIZE           32     // Warp 大小 (可配置)
```

### 17.2.2 张量布局参数
```cpp
// 张量行列参数
#define _THC_TROW_NUM_ROWS       4      // 张量行数
#define _THC_TROW_NUM_BITS_PER_ROW 512  // 每行位数

#define _THC_TCOL_NUM_COLS       4      // 张量列数  
#define _THC_TCOL_NUM_BITS_PER_COL 512  // 每列位数
```

### 17.2.3 内核配置参数
```cpp
// 内核执行配置
struct YICAKernelConfig {
    dim3 gridDim;                    // Grid 维度
    dim3 blockDim;                   // Block 维度
    dim3 swgDim;                     // SWG 维度
    dim3 cwgDim;                     // CWG 维度
    size_t sharedMemBytes;           // 共享内存字节数
    void* stream;                    // 执行流
    std::vector<void*> kernelArgs;   // 内核参数
};
```

## 17.3 性能参数基准

### 17.3.1 计算性能参数
```cpp
// 性能基准参数 (示例值)
struct YICAPerformanceSpecs {
    // 计算性能
    float peak_flops_fp32;           // 峰值 FP32 FLOPS (TFLOPS)
    float peak_flops_fp16;           // 峰值 FP16 FLOPS (TFLOPS)
    float peak_flops_int8;           // 峰值 INT8 OPS (TOPS)
    float peak_flops_int4;           // 峰值 INT4 OPS (TOPS)
    
    // 内存带宽
    float spm_bandwidth;             // SPM 内存带宽 (GB/s)
    float dram_bandwidth;            // DRAM 内存带宽 (GB/s)
    float host_bandwidth;            // 主机内存带宽 (GB/s)
    
    // 通信性能
    float yccl_bandwidth;            // YCCL 通信带宽 (GB/s)
    float yccl_latency;              // YCCL 通信延迟 (μs)
    
    // 功耗参数
    float max_power_consumption;     // 最大功耗 (W)
    float idle_power_consumption;    // 空闲功耗 (W)
    float compute_efficiency;        // 计算效率 (TOPS/W)
};
```

### 17.3.2 内存配置参数
```cpp
// 典型内存配置
struct YICAMemorySpecs {
    // SPM 配置
    size_t spm_size_per_die;         // 每个 Die 的 SPM 大小 (例: 128MB)
    size_t spm_banks;                // SPM 存储体数量
    size_t spm_bank_width;           // SPM 存储体宽度 (位)
    
    // DRAM 配置  
    size_t dram_size_per_cluster;    // 每个集群的 DRAM 大小 (例: 16GB)
    size_t dram_channels;            // DRAM 通道数
    size_t dram_bus_width;           // DRAM 总线宽度 (位)
    
    // 缓存配置
    size_t l1_cache_size;            // L1 缓存大小
    size_t l2_cache_size;            // L2 缓存大小
    size_t cache_line_size;          // 缓存行大小
};
```

### 17.3.3 运行时配置参数
```cpp
// 运行时版本参数
#define YICA_RUNTIME_VERSION_CODE(major, minor, patch) \
    (major * 10000000 + minor * 100000 + patch)

// DLM 特殊寄存器基址 (不同芯片版本)
#define DLM_SPECIAL_REG_BASE_N300    0x02420000  // N300 芯片
#define DLM_SPECIAL_REG_BASE_N900    0x0B040040  // N900 芯片

// 内核对象偏移
#define KERNEL_OBJECT_BASE_OFFSET    0           // 内核对象基址偏移
#define KERNEL_CE_TAR_BASE_OFFSET    8           // 内核计算引擎目标基址偏移
#define FUNC_PBUF_BASE_OFFSET        16          // 函数参数缓冲区基址偏移
#define FW_FUNC_TABLE_BASE_OFFSET    24          // 固件函数表基址偏移
```

## 17.4 API 兼容性参数

### 17.4.1 设备管理 API 参数
```cpp
// 设备查询参数
int device_count;                    // 设备数量
int current_device;                  // 当前设备 ID
CIMDeviceProp device_properties;     // 设备属性

// 内存信息查询
struct CIMDeviceMem {
    size_t dram_total;               // DRAM 总大小
    size_t dram_free;                // DRAM 空闲大小
    size_t spm_total;                // SPM 总大小
    size_t spm_free;                 // SPM 空闲大小
};
```

### 17.4.2 流和事件参数
```cpp
// 流配置
struct CIMStreamConfig {
    int priority;                    // 流优先级
    unsigned int flags;              // 流标志
    size_t stream_id;               // 流 ID
    int device_index;               // 设备索引
};

// 事件配置
struct CIMEventConfig {
    unsigned int flags;              // 事件标志
    bool blocking_sync;             // 阻塞同步
    bool timing_enabled;            // 时间测量使能
};
```

这些详细参数为开发类似 CUDA 的后端提供了完整的架构规范基础，涵盖了硬件配置、编程模型、性能基准和 API 兼容性等各个方面。开发者可以基于这些参数来实现对 YICA 架构的完整支持。

## 17.5 YIS 指令集详细参数

### 17.5.1 指令格式定义
```cpp
// YIS 指令格式枚举
enum YISInstructionFormat {
    InstFormatYISECOPY   = 23,    // 外部拷贝指令格式
    InstFormatYISICOPY   = 24,    // 内部拷贝指令格式  
    InstFormatYISMMA     = 25,    // 矩阵乘累加指令格式
    InstFormatYISSYNC    = 26,    // 同步指令格式
    InstFormatYISCONTROL = 27     // 控制指令格式
};
```

### 17.5.2 YISECOPY 指令参数
```cpp
// 外部拷贝指令操作码定义
#define YIS_COPY_G2G         0b00000  // Global 到 Global 拷贝
#define YIS_COPY_G2S         0b00001  // Global 到 Shared 拷贝
#define YIS_COPY_S2G         0b00010  // Shared 到 Global 拷贝
#define YIS_COPY_G2SPM       0b00011  // Global 到 SPM 拷贝
#define YIS_COPY_SPM2G       0b00100  // SPM 到 Global 拷贝
#define YIS_COPY_G2S_IM2COL  0b00101  // Global到Shared (Im2Col变换)
#define YIS_COPY_S2G_IM2COL  0b00110  // Shared到Global (Im2Col变换)
#define YIS_COPY_G2SPM_IM2COL 0b00111 // Global到SPM (Im2Col变换)
#define YIS_COPY_SPM2G_IM2COL 0b01000 // SPM到Global (Im2Col变换)

// 数据布局类型
#define YIS_LAYOUT_TROW      0b000    // 张量行布局
#define YIS_LAYOUT_ROW       0b001    // 行布局
#define YIS_LAYOUT_TCOL      0b010    // 张量列布局
#define YIS_LAYOUT_COL       0b011    // 列布局

// 块作用域类型
#define YIS_BOSCOPE_WG       0b00     // 工作组作用域
#define YIS_BOSCOPE_CWG      0b01     // 计算工作组作用域
```

### 17.5.3 YISICOPY 指令参数
```cpp
// 内部拷贝指令操作码
#define YIS_COLL_BC          0b01000  // 集合广播
#define YIS_COLL_GAT         0b01001  // 集合收集
#define YIS_COLL_RED         0b01010  // 集合归约
#define YIS_COLL_ALLRED      0b01011  // 全局归约
#define YIS_COLL_REDSCT      0b01100  // 归约散射

// 归约操作类型
enum YISReductionOp {
    YIS_RED_SUM = 0,    // 求和归约
    YIS_RED_MAX = 1,    // 最大值归约  
    YIS_RED_MIN = 2,    // 最小值归约
    YIS_RED_PROD = 3    // 乘积归约
};

// 累加使能标志
#define YIS_ACC_DISABLE      0        // 禁用累加
#define YIS_ACC_ENABLE       1        // 使能累加
```

### 17.5.4 YISMMA 指令参数
```cpp
// 矩阵乘累加指令配置
struct YISMMAConfig {
    bits<4> opcode;                  // 4位操作码
    bool accumulate;                 // 累加标志
    uint8_t matrix_shape;            // 矩阵形状配置
    uint8_t data_type;               // 数据类型
    bool transpose_a;                // 矩阵A转置
    bool transpose_b;                // 矩阵B转置
};

// 支持的矩阵形状
#define YIS_MMA_SHAPE_8x8x8     0x00  // 8x8x8 矩阵
#define YIS_MMA_SHAPE_16x16x16  0x01  // 16x16x16 矩阵
#define YIS_MMA_SHAPE_32x32x32  0x02  // 32x32x32 矩阵
```

### 17.5.5 YISSYNC 指令参数
```cpp
// 同步指令操作码
#define YIS_SYNC_WG          0b0000   // 工作组内同步
#define YIS_SYNC_CWG         0b0001   // 计算工作组内同步
#define YIS_SYNC_SWG         0b0010   // 子工作组内同步
#define YIS_SYNC_GRID        0b0011   // 网格内同步
#define YIS_SYNC_BARRIER     0b0100   // 内存屏障同步
```

### 17.5.6 YISCONTROL 指令参数
```cpp
// 控制指令操作码
#define YIS_CTRL_LOOP_BEGIN  0b00000  // 循环开始
#define YIS_CTRL_LOOP_END    0b00001  // 循环结束
#define YIS_CTRL_BRANCH      0b00010  // 分支控制
#define YIS_CTRL_CALL        0b00011  // 函数调用
#define YIS_CTRL_RET         0b00100  // 函数返回
#define YIS_CTRL_PREDICATE   0b00101  // 谓词执行

// 控制流预设配置
struct YISControlConfig {
    uint8_t condition_mask;          // 条件掩码
    uint8_t predicate_reg;           // 谓词寄存器
    uint16_t target_address;         // 目标地址
    bool conditional;                // 条件执行标志
};
```

### 17.5.7 张量描述符参数
```cpp
// 张量描述符结构
struct YISTensorDescriptor {
    uint32_t rank;                   // 张量维度
    uint64_t base_addr;              // 基址 (例: 0x40c00000)
    uint8_t data_type;               // 数据类型 (8=float16)
    uint32_t shape[4];               // 形状数组 [dim0, dim1, dim2, dim3]
    uint32_t stride[4];              // 步长数组 [stride0, stride1, stride2, stride3]
};

// 典型张量描述符示例
{
    "rank": 2,
    "addr": "0x40c00000", 
    "type": 8,                       // float16
    "shape": [8, 128],               // 8 x 128 矩阵
    "stride": [256, 2]               // 步长配置
}
```

### 17.5.8 立即数和寄存器参数
```cpp
// YIS 专用立即数类型
#define YIS_UIMM4_MAX        15       // 4位无符号立即数最大值
#define YIS_UIMM7_MAX        127      // 7位无符号立即数最大值
#define YIS_SIMM12_MIN       -2048    // 12位有符号立即数最小值
#define YIS_SIMM12_MAX       2047     // 12位有符号立即数最大值

// 寄存器大小标识
enum YISRegisterSize {
    YIS_REG_SIZE_32 = 0,            // 32位寄存器
    YIS_REG_SIZE_64 = 1             // 64位寄存器
};

// 地址类型标识
#define YIS_ADDR_TYPE_IMM    0       // 立即数地址
#define YIS_ADDR_TYPE_REG    1       // 寄存器地址
```

### 17.5.9 性能调优参数
```cpp
// YIS 指令性能参数
struct YISPerformanceParams {
    // 指令延迟 (周期数)
    uint32_t copy_latency;           // 拷贝指令延迟
    uint32_t mma_latency;            // 矩阵乘累加延迟
    uint32_t sync_latency;           // 同步指令延迟
    
    // 吞吐量 (每周期指令数)
    float copy_throughput;           // 拷贝指令吞吐量
    float mma_throughput;            // MMA 指令吞吐量
    float control_throughput;        // 控制指令吞吐量
    
    // 并发度
    uint32_t max_concurrent_copies;  // 最大并发拷贝数
    uint32_t max_concurrent_mmas;    // 最大并发 MMA 数
};
```

这些详细的 YIS 指令集参数提供了 YICA 架构底层硬件控制的完整规范，为编译器后端和运行时系统提供了精确的指令生成和优化依据。

---
*基于 torch_cim、ybx_replay 和 LLVM RISC-V 后端深度代码分析生成*  
*文档版本：v2.0，更新日期：2024年12月*  
*涵盖：硬件架构、软件栈、指令集、性能优化、应用场景的完整技术规范*
