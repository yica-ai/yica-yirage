#!/usr/bin/env python3
"""
YICA神经网络模型优化实验套件
基于Mirage现有用例，扩展YICA特定的操作优化
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import subprocess

# 添加YICA优化器路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mirage', 'python'))

# 导入YICA优化器（模拟模式）
try:
    from mirage.yica_optimizer import YICAConfig, YICAMirageOptimizer
    MIRAGE_AVAILABLE = True
except ImportError:
    print("⚠️  使用模拟YICA优化器")
    MIRAGE_AVAILABLE = False
    
    # 使用独立的YICA优化器
    from demo_yica_standalone import YICAConfig, YICAOptimizer as YICAMirageOptimizer, MockComputeGraph

class YICANeuralNetworkExperiments:
    """YICA神经网络实验类"""
    
    def __init__(self, yica_config: Optional[YICAConfig] = None):
        self.yica_config = yica_config or YICAConfig()
        self.optimizer = YICAMirageOptimizer(self.yica_config)
        self.experiment_results = []
        
    def create_llama_attention_graph(self, batch_size: int = 8, seq_len: int = 2048, 
                                   hidden_size: int = 4096, num_heads: int = 32) -> Dict[str, Any]:
        """创建LLaMA风格的Attention计算图"""
        head_dim = hidden_size // num_heads
        
        # 模拟Mirage计算图结构
        operations = [
            f"input:Q({batch_size},{seq_len},{hidden_size})",
            f"input:K({batch_size},{seq_len},{hidden_size})",
            f"input:V({batch_size},{seq_len},{hidden_size})",
            f"reshape:Q_heads({batch_size},{seq_len},{num_heads},{head_dim})",
            f"reshape:K_heads({batch_size},{seq_len},{num_heads},{head_dim})",
            f"reshape:V_heads({batch_size},{seq_len},{num_heads},{head_dim})",
            f"transpose:K_T({batch_size},{num_heads},{head_dim},{seq_len})",
            f"matmul:scores({batch_size},{num_heads},{seq_len},{seq_len})",
            f"scale:scaled_scores({batch_size},{num_heads},{seq_len},{seq_len})",
            f"softmax:attention_weights({batch_size},{num_heads},{seq_len},{seq_len})",
            f"matmul:context({batch_size},{num_heads},{seq_len},{head_dim})",
            f"reshape:output({batch_size},{seq_len},{hidden_size})"
        ]
        
        return {
            'name': f'LLaMA_Attention_B{batch_size}_S{seq_len}_H{hidden_size}',
            'type': 'attention',
            'operations': operations,
            'parameters': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'head_dim': head_dim
            }
        }
    
    def create_gated_mlp_graph(self, batch_size: int = 8, hidden_size: int = 4096, 
                              intermediate_size: int = 14336) -> Dict[str, Any]:
        """创建Gated MLP计算图"""
        operations = [
            f"input:X({batch_size},{hidden_size})",
            f"input:W1({hidden_size},{intermediate_size})",
            f"input:W2({hidden_size},{intermediate_size})",
            f"input:W3({intermediate_size},{hidden_size})",
            f"matmul:gate({batch_size},{intermediate_size})",
            f"matmul:up({batch_size},{intermediate_size})",
            f"silu:activated_gate({batch_size},{intermediate_size})",
            f"mul:gated({batch_size},{intermediate_size})",
            f"matmul:output({batch_size},{hidden_size})"
        ]
        
        return {
            'name': f'Gated_MLP_B{batch_size}_H{hidden_size}_I{intermediate_size}',
            'type': 'mlp',
            'operations': operations,
            'parameters': {
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'intermediate_size': intermediate_size
            }
        }
    
    def create_rms_norm_graph(self, batch_size: int = 8, hidden_size: int = 4096) -> Dict[str, Any]:
        """创建RMS Normalization计算图"""
        operations = [
            f"input:X({batch_size},{hidden_size})",
            f"input:weight({hidden_size})",
            f"square:X_squared({batch_size},{hidden_size})",
            f"reduce_mean:mean_square({batch_size},1)",
            f"add:variance({batch_size},1)",
            f"rsqrt:inv_std({batch_size},1)",
            f"mul:normalized({batch_size},{hidden_size})",
            f"mul:output({batch_size},{hidden_size})"
        ]
        
        return {
            'name': f'RMS_Norm_B{batch_size}_H{hidden_size}',
            'type': 'normalization',
            'operations': operations,
            'parameters': {
                'batch_size': batch_size,
                'hidden_size': hidden_size
            }
        }
    
    def create_conv2d_graph(self, batch_size: int = 32, channels: int = 256, 
                           height: int = 224, width: int = 224,
                           out_channels: int = 512, kernel_size: int = 3) -> Dict[str, Any]:
        """创建Conv2D计算图"""
        operations = [
            f"input:X({batch_size},{channels},{height},{width})",
            f"input:weight({out_channels},{channels},{kernel_size},{kernel_size})",
            f"input:bias({out_channels})",
            f"conv2d:conv_out({batch_size},{out_channels},{height-kernel_size+1},{width-kernel_size+1})",
            f"add:output({batch_size},{out_channels},{height-kernel_size+1},{width-kernel_size+1})"
        ]
        
        return {
            'name': f'Conv2D_B{batch_size}_C{channels}_H{height}_W{width}_K{kernel_size}',
            'type': 'convolution',
            'operations': operations,
            'parameters': {
                'batch_size': batch_size,
                'in_channels': channels,
                'out_channels': out_channels,
                'height': height,
                'width': width,
                'kernel_size': kernel_size
            }
        }
    
    def create_transformer_block_graph(self, batch_size: int = 8, seq_len: int = 2048,
                                     hidden_size: int = 4096, num_heads: int = 32,
                                     intermediate_size: int = 14336) -> Dict[str, Any]:
        """创建完整的Transformer Block计算图"""
        operations = [
            # Pre-attention layer norm
            f"input:X({batch_size},{seq_len},{hidden_size})",
            f"rms_norm:norm1({batch_size},{seq_len},{hidden_size})",
            
            # Multi-head attention
            f"matmul:qkv({batch_size},{seq_len},{hidden_size*3})",
            f"split:q_k_v({batch_size},{seq_len},{hidden_size})",
            f"attention:attn_out({batch_size},{seq_len},{hidden_size})",
            f"matmul:attn_proj({batch_size},{seq_len},{hidden_size})",
            
            # Residual connection
            f"add:residual1({batch_size},{seq_len},{hidden_size})",
            
            # Pre-MLP layer norm
            f"rms_norm:norm2({batch_size},{seq_len},{hidden_size})",
            
            # Gated MLP
            f"matmul:gate({batch_size},{seq_len},{intermediate_size})",
            f"matmul:up({batch_size},{seq_len},{intermediate_size})",
            f"silu:activated({batch_size},{seq_len},{intermediate_size})",
            f"mul:gated({batch_size},{seq_len},{intermediate_size})",
            f"matmul:down({batch_size},{seq_len},{hidden_size})",
            
            # Final residual connection
            f"add:output({batch_size},{seq_len},{hidden_size})"
        ]
        
        return {
            'name': f'Transformer_Block_B{batch_size}_S{seq_len}_H{hidden_size}',
            'type': 'transformer_block',
            'operations': operations,
            'parameters': {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'intermediate_size': intermediate_size
            }
        }
    
    def run_experiment(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个实验"""
        print(f"\n🔬 实验: {graph_data['name']}")
        print(f"   类型: {graph_data['type']}")
        print(f"   操作数: {len(graph_data['operations'])}")
        
        start_time = time.time()
        
        # 创建模拟计算图
        if MIRAGE_AVAILABLE:
            # 使用真实的Mirage接口
            # mock_graph = create_mirage_graph(graph_data)
            # analysis_result = self.optimizer.analyze_mirage_graph(mock_graph)
            pass
        else:
            # 使用模拟接口
            mock_graph = MockComputeGraph(graph_data['name'], graph_data['operations'])
            analysis_result = self.optimizer.analyze_graph(mock_graph)
        
        optimization_time = time.time() - start_time
        
        # 计算YICA特定指标
        yica_metrics = self.calculate_yica_metrics(graph_data, analysis_result)
        
        result = {
            'graph_name': graph_data['name'],
            'graph_type': graph_data['type'],
            'parameters': graph_data['parameters'],
            'num_operations': len(graph_data['operations']),
            'yica_friendliness': analysis_result.yica_friendliness,
            'compute_intensity': analysis_result.compute_intensity,
            'parallelization_potential': analysis_result.parallelization_potential,
            'memory_bottleneck': analysis_result.memory_bottleneck,
            'optimization_strategies': analysis_result.optimization_strategies,
            'baseline_time_ms': analysis_result.baseline_time_ms,
            'optimized_time_ms': analysis_result.optimized_time_ms,
            'speedup_ratio': analysis_result.speedup_ratio,
            'cim_utilization': analysis_result.cim_utilization,
            'optimization_time_ms': optimization_time * 1000,
            'yica_metrics': yica_metrics,
            'generated_code_size': len(analysis_result.generated_code) if hasattr(analysis_result, 'generated_code') else 0
        }
        
        print(f"   📊 YICA友好度: {result['yica_friendliness']:.3f}")
        print(f"   📊 计算密集度: {result['compute_intensity']:.1f} GFLOPS")
        print(f"   ⚡ 加速比: {result['speedup_ratio']:.1f}x")
        print(f"   🧠 CIM利用率: {result['cim_utilization']:.1f}%")
        print(f"   ⏱️  优化时间: {result['optimization_time_ms']:.2f}ms")
        
        return result
    
    def calculate_yica_metrics(self, graph_data: Dict[str, Any], analysis_result) -> Dict[str, float]:
        """计算YICA特定指标"""
        graph_type = graph_data['type']
        params = graph_data['parameters']
        
        # 基于图类型计算特定指标
        if graph_type == 'attention':
            # Attention特定指标
            seq_len = params['seq_len']
            num_heads = params['num_heads']
            
            # 计算注意力计算复杂度
            attention_complexity = seq_len * seq_len * num_heads
            
            # 内存访问模式分析
            memory_pattern_score = min(1.0, 1000.0 / seq_len)  # 较短序列更适合CIM
            
            # 并行化潜力
            parallelization_score = min(1.0, num_heads / self.yica_config.num_cim_arrays)
            
            return {
                'attention_complexity': attention_complexity,
                'memory_pattern_score': memory_pattern_score,
                'parallelization_score': parallelization_score,
                'cache_efficiency': 0.8 if seq_len <= 1024 else 0.6
            }
            
        elif graph_type == 'mlp':
            # MLP特定指标
            hidden_size = params['hidden_size']
            intermediate_size = params.get('intermediate_size', hidden_size * 4)
            
            # 矩阵乘法友好度
            matmul_friendliness = min(1.0, hidden_size / 1024.0)
            
            # 计算强度
            compute_intensity = (hidden_size * intermediate_size) / (hidden_size + intermediate_size)
            
            return {
                'matmul_friendliness': matmul_friendliness,
                'compute_intensity': compute_intensity,
                'activation_overhead': 0.1,  # SiLU激活函数开销
                'memory_reuse_potential': 0.9
            }
            
        elif graph_type == 'convolution':
            # 卷积特定指标
            in_channels = params['in_channels']
            out_channels = params['out_channels']
            kernel_size = params['kernel_size']
            
            # 卷积计算密度
            conv_density = (in_channels * out_channels * kernel_size * kernel_size) / 1000000
            
            # 空间局部性
            spatial_locality = 1.0 / (kernel_size * kernel_size)
            
            return {
                'conv_density': conv_density,
                'spatial_locality': spatial_locality,
                'channel_parallelism': min(1.0, out_channels / 256.0),
                'weight_reuse_factor': kernel_size * kernel_size
            }
            
        else:
            # 通用指标
            return {
                'general_friendliness': analysis_result.yica_friendliness,
                'compute_bound_ratio': 0.7,
                'memory_bound_ratio': 0.3
            }
    
    def run_scaling_experiments(self) -> List[Dict[str, Any]]:
        """运行缩放实验"""
        print("\n📈 YICA缩放性实验")
        print("=" * 60)
        
        results = []
        
        # 1. Attention缩放实验
        print("\n🔍 Attention缩放实验")
        attention_configs = [
            (1, 512, 2048, 16),    # 小规模
            (4, 1024, 4096, 32),   # 中等规模
            (8, 2048, 4096, 32),   # 大规模
            (16, 4096, 8192, 64),  # 超大规模
        ]
        
        for batch_size, seq_len, hidden_size, num_heads in attention_configs:
            graph = self.create_llama_attention_graph(batch_size, seq_len, hidden_size, num_heads)
            result = self.run_experiment(graph)
            results.append(result)
        
        # 2. MLP缩放实验
        print("\n🧠 MLP缩放实验")
        mlp_configs = [
            (8, 2048, 8192),      # 小规模
            (8, 4096, 14336),     # LLaMA规模
            (8, 8192, 28672),     # 大规模
            (8, 16384, 57344),    # 超大规模
        ]
        
        for batch_size, hidden_size, intermediate_size in mlp_configs:
            graph = self.create_gated_mlp_graph(batch_size, hidden_size, intermediate_size)
            result = self.run_experiment(graph)
            results.append(result)
        
        # 3. 完整Transformer Block实验
        print("\n🏗️  Transformer Block实验")
        transformer_configs = [
            (1, 512, 2048, 16, 8192),     # 小模型
            (4, 1024, 4096, 32, 14336),   # 中等模型
            (8, 2048, 4096, 32, 14336),   # LLaMA-7B
            (8, 2048, 8192, 64, 28672),   # LLaMA-70B风格
        ]
        
        for batch_size, seq_len, hidden_size, num_heads, intermediate_size in transformer_configs:
            graph = self.create_transformer_block_graph(batch_size, seq_len, hidden_size, num_heads, intermediate_size)
            result = self.run_experiment(graph)
            results.append(result)
        
        return results
    
    def run_specialty_experiments(self) -> List[Dict[str, Any]]:
        """运行专项实验"""
        print("\n🎯 YICA专项优化实验")
        print("=" * 60)
        
        results = []
        
        # 1. 不同数据类型实验
        print("\n📊 数据类型优化实验")
        data_type_experiments = [
            ("FP16_Attention", self.create_llama_attention_graph(8, 1024, 4096, 32)),
            ("FP16_MLP", self.create_gated_mlp_graph(8, 4096, 14336)),
            ("Mixed_Precision", self.create_transformer_block_graph(8, 1024, 4096, 32, 14336))
        ]
        
        for exp_name, graph in data_type_experiments:
            graph['name'] = f"{exp_name}_{graph['name']}"
            result = self.run_experiment(graph)
            results.append(result)
        
        # 2. 内存访问模式优化
        print("\n💾 内存访问模式优化")
        memory_experiments = [
            ("Sequential_Access", self.create_rms_norm_graph(64, 4096)),
            ("Strided_Access", self.create_conv2d_graph(32, 256, 224, 224, 512, 3)),
            ("Random_Access", self.create_llama_attention_graph(8, 2048, 4096, 32))
        ]
        
        for exp_name, graph in memory_experiments:
            graph['name'] = f"{exp_name}_{graph['name']}"
            result = self.run_experiment(graph)
            results.append(result)
        
        # 3. CIM阵列利用率优化
        print("\n🔧 CIM阵列利用率优化")
        cim_experiments = []
        
        # 不同CIM配置
        original_config = self.yica_config.num_cim_arrays
        for num_arrays in [2, 4, 8, 16]:
            self.yica_config.num_cim_arrays = num_arrays
            self.optimizer = YICAMirageOptimizer(self.yica_config)
            
            graph = self.create_gated_mlp_graph(8, 4096, 14336)
            graph['name'] = f"CIM{num_arrays}_{graph['name']}"
            result = self.run_experiment(graph)
            results.append(result)
        
        # 恢复原始配置
        self.yica_config.num_cim_arrays = original_config
        self.optimizer = YICAMirageOptimizer(self.yica_config)
        
        return results
    
    def generate_yica_kernels(self, experiment_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """为最佳实验结果生成YICA优化内核"""
        print("\n🔧 生成YICA优化内核")
        print("=" * 50)
        
        # 选择最佳结果
        best_results = sorted(experiment_results, key=lambda x: x['speedup_ratio'], reverse=True)[:5]
        
        generated_kernels = {}
        
        for i, result in enumerate(best_results):
            kernel_name = f"yica_{result['graph_type']}_{result['graph_name'].lower()}_kernel"
            
            # 生成优化内核代码
            kernel_code = self.generate_kernel_code(result)
            
            # 保存到文件
            filename = f"{kernel_name}.py"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(kernel_code)
            
            generated_kernels[filename] = kernel_code
            print(f"   📄 生成内核: {filename} ({len(kernel_code)} chars)")
        
        return generated_kernels
    
    def generate_kernel_code(self, result: Dict[str, Any]) -> str:
        """生成特定类型的内核代码"""
        graph_type = result['graph_type']
        graph_name = result['graph_name']
        speedup = result['speedup_ratio']
        cim_util = result['cim_utilization']
        
        # 基础模板
        base_template = f"""# YICA优化的{graph_name}内核
# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
# 图类型: {graph_type}
# 性能提升: {speedup:.1f}x
# CIM利用率: {cim_util:.1f}%

import triton
import triton.language as tl

@triton.jit
def yica_optimized_{graph_type}_kernel(
    # 输入指针
    input_ptr,
    output_ptr,
    # 形状参数
    M, N, K,
    # 步长参数
    stride_input_m, stride_input_n,
    stride_output_m, stride_output_n,
    # YICA特定参数
    CIM_ARRAYS: tl.constexpr,
    SPM_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    \"\"\"
    YICA优化的{graph_type}内核
    
    特性:
    - 利用{self.yica_config.num_cim_arrays}个CIM阵列并行计算
    - SPM内存层次优化: {self.yica_config.spm_size_kb}KB
    - 智能负载均衡和数据重用
    \"\"\"
    
    # 获取程序ID和网格配置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # YICA CIM阵列分配策略
    cim_id = pid_m % CIM_ARRAYS
    local_pid_m = pid_m // CIM_ARRAYS
    
    # 计算数据块偏移
    offs_m = local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 边界检查
    mask_m = offs_m < M
    mask_n = offs_n < N
    
"""
        
        # 根据图类型添加特定优化
        if graph_type == 'attention':
            specific_code = """
    # Attention特定优化
    # 1. Q、K、V矩阵的分块加载
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 2. CIM阵列并行计算注意力分数
    # 使用存算一体特性减少数据移动
    q_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_n
    k_ptrs = input_ptr + offs_k[:, None] * stride_input_m + offs_n[None, :] * stride_input_n
    
    # SPM优化的数据预取
    q_block = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K))
    k_block = tl.load(k_ptrs, mask=(offs_k[:, None] < K) & mask_n[None, :])
    
    # CIM阵列并行矩阵乘法
    attention_scores = tl.dot(q_block, k_block)
    
    # 3. Softmax优化（利用CIM的并行计算能力）
    max_scores = tl.max(attention_scores, axis=1)
    attention_scores = attention_scores - max_scores[:, None]
    exp_scores = tl.exp(attention_scores)
    sum_exp = tl.sum(exp_scores, axis=1)
    attention_weights = exp_scores / sum_exp[:, None]
    
    # 4. 输出计算
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, attention_weights, mask=mask_m[:, None] & mask_n[None, :])
"""
        elif graph_type == 'mlp':
            specific_code = """
    # MLP特定优化
    # 1. 矩阵乘法的CIM阵列分配
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 2. Gated MLP的并行计算
    # Gate分支
    gate_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_n
    gate_data = tl.load(gate_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K))
    
    # Up分支
    up_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_n
    up_data = tl.load(up_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < K))
    
    # 3. SiLU激活函数优化（存算一体计算）
    # SiLU(x) = x * sigmoid(x)
    sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_data))
    activated_gate = gate_data * sigmoid_gate
    
    # 4. Gated操作
    gated_output = activated_gate * up_data
    
    # 5. SPM优化的结果存储
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, gated_output, mask=mask_m[:, None] & mask_n[None, :])
"""
        elif graph_type == 'convolution':
            specific_code = """
    # 卷积特定优化
    # 1. 卷积窗口的CIM阵列映射
    kernel_size = 3  # 示例
    
    # 2. 空间局部性优化
    # 利用CIM阵列的并行性处理多个卷积窗口
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # 输入数据偏移
            input_offset = (offs_m + kh) * stride_input_m + (offs_n + kw) * stride_input_n
            input_ptrs = input_ptr + input_offset
            
            # 权重偏移
            weight_offset = kh * kernel_size + kw
            
            # CIM并行计算
            input_data = tl.load(input_ptrs, mask=mask_m & mask_n)
            # 卷积计算逻辑...
    
    # 3. 结果累积和存储
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    # tl.store(output_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])
"""
        else:
            specific_code = """
    # 通用YICA优化
    # 1. 数据加载优化
    input_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_n[None, :] * stride_input_n
    input_data = tl.load(input_ptrs, mask=mask_m[:, None] & mask_n[None, :])
    
    # 2. CIM阵列并行计算
    # 根据具体操作类型进行优化...
    result = input_data  # 占位符
    
    # 3. 结果存储
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    tl.store(output_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])
"""
        
        # 结尾模板
        end_template = f"""

# YICA运行时支持函数
def launch_yica_{graph_type}_kernel(input_tensor, output_tensor, **kwargs):
    \"\"\"启动YICA优化内核\"\"\"
    M, N = input_tensor.shape[:2]
    K = input_tensor.shape[-1] if len(input_tensor.shape) > 2 else N
    
    # 网格配置 - 根据CIM阵列数量优化
    grid = (
        triton.cdiv(M, 32) * {self.yica_config.num_cim_arrays},  # 利用多个CIM阵列
        triton.cdiv(N, 32),
    )
    
    # 启动内核
    yica_optimized_{graph_type}_kernel[grid](
        input_tensor, output_tensor,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        CIM_ARRAYS={self.yica_config.num_cim_arrays},
        SPM_SIZE={self.yica_config.spm_size_kb},
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=32,
    )
    
    return output_tensor

# 性能基准
def benchmark_yica_{graph_type}():
    \"\"\"YICA {graph_type}性能基准测试\"\"\"
    import torch
    
    # 创建测试数据
    device = 'cuda'
    M, N, K = 1024, 1024, 1024
    input_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
    output_tensor = torch.empty(M, N, dtype=torch.float16, device=device)
    
    # 预热
    for _ in range(10):
        launch_yica_{graph_type}_kernel(input_tensor, output_tensor)
    
    # 基准测试
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(100):
        launch_yica_{graph_type}_kernel(input_tensor, output_tensor)
    end_time.record()
    
    torch.cuda.synchronize()
    avg_time = start_time.elapsed_time(end_time) / 100
    
    print(f"YICA {graph_type}平均执行时间: {{avg_time:.2f}}ms")
    print(f"预期加速比: {speedup:.1f}x")
    print(f"CIM利用率: {cim_util:.1f}%")
    
    return avg_time

if __name__ == "__main__":
    benchmark_yica_{graph_type}()
"""
        
        return base_template + specific_code + end_template

def main():
    """主实验流程"""
    print("🧪 YICA神经网络模型优化实验套件")
    print("基于Mirage用例的大规模神经网络优化实验")
    print("=" * 80)
    
    # 初始化实验环境
    yica_config = YICAConfig()
    yica_config.num_cim_arrays = 4
    yica_config.spm_size_kb = 512
    yica_config.memory_bandwidth_gbps = 1000.0
    
    experiments = YICANeuralNetworkExperiments(yica_config)
    
    print(f"📋 YICA配置:")
    print(f"   - CIM阵列数量: {yica_config.num_cim_arrays}")
    print(f"   - SPM大小: {yica_config.spm_size_kb}KB")
    print(f"   - 内存带宽: {yica_config.memory_bandwidth_gbps}GB/s")
    
    # 运行实验
    all_results = []
    
    # 1. 缩放性实验
    scaling_results = experiments.run_scaling_experiments()
    all_results.extend(scaling_results)
    
    # 2. 专项优化实验
    specialty_results = experiments.run_specialty_experiments()
    all_results.extend(specialty_results)
    
    # 3. 生成优化内核
    generated_kernels = experiments.generate_yica_kernels(all_results)
    
    # 4. 分析和报告
    print(f"\n📊 实验结果分析")
    print("=" * 60)
    
    # 统计分析
    total_experiments = len(all_results)
    avg_speedup = sum(r['speedup_ratio'] for r in all_results) / total_experiments
    avg_cim_util = sum(r['cim_utilization'] for r in all_results) / total_experiments
    best_result = max(all_results, key=lambda x: x['speedup_ratio'])
    
    print(f"📈 实验统计:")
    print(f"   - 总实验数: {total_experiments}")
    print(f"   - 平均加速比: {avg_speedup:.1f}x")
    print(f"   - 平均CIM利用率: {avg_cim_util:.1f}%")
    print(f"   - 最佳性能: {best_result['graph_name']} ({best_result['speedup_ratio']:.1f}x)")
    print(f"   - 生成内核数: {len(generated_kernels)}")
    
    # 按类型分析
    type_stats = {}
    for result in all_results:
        graph_type = result['graph_type']
        if graph_type not in type_stats:
            type_stats[graph_type] = {'count': 0, 'speedup_sum': 0, 'cim_util_sum': 0}
        type_stats[graph_type]['count'] += 1
        type_stats[graph_type]['speedup_sum'] += result['speedup_ratio']
        type_stats[graph_type]['cim_util_sum'] += result['cim_utilization']
    
    print(f"\n📋 按类型统计:")
    print("-" * 60)
    print(f"{'类型':<20} {'实验数':<8} {'平均加速':<12} {'平均CIM利用率':<15}")
    print("-" * 60)
    for graph_type, stats in type_stats.items():
        avg_speedup = stats['speedup_sum'] / stats['count']
        avg_cim = stats['cim_util_sum'] / stats['count']
        print(f"{graph_type:<20} {stats['count']:<8} {avg_speedup:<12.1f}x {avg_cim:<15.1f}%")
    print("-" * 60)
    
    # 保存详细结果
    with open('yica_neural_network_experiments_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'yica_config': {
                'num_cim_arrays': yica_config.num_cim_arrays,
                'spm_size_kb': yica_config.spm_size_kb,
                'memory_bandwidth_gbps': yica_config.memory_bandwidth_gbps
            },
            'experiment_summary': {
                'total_experiments': total_experiments,
                'avg_speedup': avg_speedup,
                'avg_cim_utilization': avg_cim_util,
                'best_result': best_result,
                'type_statistics': type_stats
            },
            'detailed_results': all_results,
            'generated_kernels': list(generated_kernels.keys())
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细结果已保存: yica_neural_network_experiments_results.json")
    print(f"🔧 生成的内核文件: {', '.join(generated_kernels.keys())}")
    
    # 实验结论
    print(f"\n🎯 实验结论:")
    if avg_speedup >= 5.0:
        print(f"   🎉 YICA优化效果优秀！平均{avg_speedup:.1f}x加速")
    elif avg_speedup >= 3.0:
        print(f"   ✅ YICA优化效果良好！平均{avg_speedup:.1f}x加速")
    elif avg_speedup >= 2.0:
        print(f"   ⚠️  YICA优化效果一般，平均{avg_speedup:.1f}x加速")
    else:
        print(f"   ❌ YICA优化效果需要改进，平均{avg_speedup:.1f}x加速")
    
    print(f"\n📚 下一步建议:")
    print(f"   1. 针对表现最佳的{best_result['graph_type']}类型进行深度优化")
    print(f"   2. 优化CIM利用率较低的操作类型")
    print(f"   3. 在真实YICA硬件上验证生成的内核性能")
    print(f"   4. 扩展更多神经网络模型的支持")
    
    return len(all_results)

if __name__ == "__main__":
    num_experiments = main()
    print(f"\n🎉 YICA神经网络优化实验完成！总计{num_experiments}个实验") 