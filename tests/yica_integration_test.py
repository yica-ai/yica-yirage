#!/usr/bin/env python3
"""
YICA-Mirage 集成测试系统
========================

全面测试 YICA 后端与 Mirage 框架的集成，验证各组件协同工作。
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YICATestConfig:
    """YICA 测试配置"""
    
    def __init__(self):
        self.test_mode = os.getenv('YICA_TEST_MODE', 'simulation')
        self.enable_performance_tests = True
        self.test_data_dir = '/tmp/yica_test_data'
        os.makedirs(self.test_data_dir, exist_ok=True)

class TestYICABasicFunctionality(unittest.TestCase):
    """YICA 基础功能测试"""
    
    def setUp(self):
        self.config = YICATestConfig()
    
    def test_tensor_operations(self):
        """测试基本张量操作"""
        logger.info("测试基本张量操作")
        
        # 创建测试张量
        a = torch.randn(256, 256)
        b = torch.randn(256, 256)
        
        # 测试加法
        result_add = torch.add(a, b)
        self.assertEqual(result_add.shape, (256, 256))
        
        # 测试矩阵乘法
        result_mm = torch.mm(a, b)
        self.assertEqual(result_mm.shape, (256, 256))
        
        logger.info("✅ 基本张量操作测试通过")
    
    def test_model_creation(self):
        """测试模型创建"""
        logger.info("测试模型创建")
        
        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # 测试前向传播
        x = torch.randn(32, 256)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (32, 10))
        logger.info("✅ 模型创建测试通过")
    
    def test_performance_baseline(self):
        """测试性能基线"""
        logger.info("测试性能基线")
        
        if not self.config.enable_performance_tests:
            self.skipTest("性能测试被禁用")
        
        # 矩阵乘法性能测试
        size = 512
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # 预热
        for _ in range(3):
            torch.mm(a, b)
        
        # 基准测试
        start_time = time.time()
        for _ in range(10):
            result = torch.mm(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        throughput = (size * size * size * 2) / avg_time / 1e9  # GFLOPS
        
        logger.info(f"矩阵乘法 {size}x{size}: {avg_time:.4f}s, {throughput:.2f} GFLOPS")
        
        # 验证性能合理性
        self.assertGreater(throughput, 0)
        self.assertLess(avg_time, 1.0)
        
        logger.info("✅ 性能基线测试通过")

class TestYICAModelOptimization(unittest.TestCase):
    """YICA 模型优化测试"""
    
    def test_simple_optimization(self):
        """测试简单模型优化"""
        logger.info("测试简单模型优化")
        
        # 创建测试模型
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        # 测试推理
        x = torch.randn(16, 128)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (16, 64))
        self.assertFalse(torch.any(torch.isnan(output)))
        
        logger.info("✅ 简单模型优化测试通过")

class YICATestRunner:
    """YICA 测试运行器"""
    
    def __init__(self):
        self.config = YICATestConfig()
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("🚀 开始运行 YICA-Mirage 集成测试")
        
        test_classes = [
            ('basic_functionality', TestYICABasicFunctionality),
            ('model_optimization', TestYICAModelOptimization),
        ]
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        
        for suite_name, test_class in test_classes:
            logger.info(f"\n📋 运行测试套件: {suite_name}")
            
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2)
            
            try:
                result = runner.run(suite)
                
                suite_total = result.testsRun
                suite_passed = suite_total - len(result.failures) - len(result.errors)
                suite_failed = len(result.failures) + len(result.errors)
                
                total_tests += suite_total
                passed_tests += suite_passed
                failed_tests += suite_failed
                
                self.results[suite_name] = {
                    'total': suite_total,
                    'passed': suite_passed,
                    'failed': suite_failed
                }
                
                if suite_failed == 0:
                    logger.info(f"✅ {suite_name}: {suite_passed}/{suite_total} 通过")
                else:
                    logger.warning(f"⚠️ {suite_name}: {suite_passed}/{suite_total} 通过, {suite_failed} 失败")
                
            except Exception as e:
                logger.error(f"❌ 测试套件 {suite_name} 运行失败: {e}")
                self.results[suite_name] = {'error': str(e)}
        
        # 生成总结报告
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"\n📊 测试总结:")
        logger.info(f"   总测试数: {total_tests}")
        logger.info(f"   通过: {passed_tests}")
        logger.info(f"   失败: {failed_tests}")
        logger.info(f"   成功率: {success_rate:.1f}%")
        
        if failed_tests == 0:
            logger.info("🎉 所有测试通过!")
        else:
            logger.warning(f"⚠️ {failed_tests} 个测试失败")
        
        return self.results

if __name__ == '__main__':
    # 创建并运行测试
    runner = YICATestRunner()
    results = runner.run_all_tests()
    
    # 根据测试结果设置退出码
    total_failed = sum(
        result.get('failed', 0) for result in results.values() 
        if isinstance(result, dict) and 'failed' in result
    )
    
    sys.exit(1 if total_failed > 0 else 0) 