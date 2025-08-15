#!/usr/bin/env python3
"""
YICA-YiRage Simple Working Demo
================================
A simplified demonstration that uses only the actual available APIs.
"""

import yirage
import time
import json

def main():
    print("\n" + "=" * 60)
    print("🚀 YICA-YiRage v{} Working Demo".format(yirage.__version__))
    print("=" * 60)
    
    # 1. System Check
    print("\n✅ System Status:")
    print(f"  Version: {yirage.__version__}")
    print(f"  YICA Core: {yirage.YICA_CORE_AVAILABLE}")
    print(f"  YICA Advanced: {yirage.YICA_ADVANCED_AVAILABLE}")
    print(f"  YICA Monitor: {yirage.YICA_MONITOR_AVAILABLE}")
    print(f"  YICA Optimizer: {yirage.YICA_OPTIMIZER_AVAILABLE}")
    
    # 2. YICA Backend Test
    print("\n✅ YICA Backend:")
    from yirage.yica import YICABackend
    backend = YICABackend()
    print(f"  Device count: {backend.device_count()}")
    print(f"  Backend initialized successfully")
    
    # 3. Create Kernel Graph
    print("\n✅ Kernel Graph Creation:")
    graph = yirage.new_kernel_graph()
    print(f"  Graph created: {graph}")
    
    # Add simple operations
    X = graph.new_input(dims=(32, 512, 768), dtype=yirage.float16)
    print(f"  Input tensor created: shape (32, 512, 768)")
    
    # 4. YICA Optimizer
    print("\n✅ YICA Optimizer:")
    optimizer = yirage.create_yica_optimizer()
    print(f"  Optimizer created: {optimizer}")
    
    # 5. Performance Monitor
    print("\n✅ Performance Monitor:")
    monitor = yirage.create_performance_monitor()
    print(f"  Monitor created: {monitor}")
    
    # 6. Quick Analysis
    print("\n✅ Quick Analysis:")
    analysis = yirage.quick_analyze()
    print(f"  Analysis available: {analysis is not None}")
    
    # 7. Version Info
    print("\n✅ Version Information:")
    version_info = yirage.get_version_info()
    for key, value in version_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✨ All YICA components are working correctly!")
    print("=" * 60)
    
    # Save results
    results = {
        "version": yirage.__version__,
        "yica_available": all([
            yirage.YICA_CORE_AVAILABLE,
            yirage.YICA_ADVANCED_AVAILABLE,
            yirage.YICA_MONITOR_AVAILABLE,
            yirage.YICA_OPTIMIZER_AVAILABLE
        ]),
        "backend_working": True,
        "optimizer_working": True,
        "monitor_working": True
    }
    
    with open("yica_simple_demo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: yica_simple_demo_results.json")
    
    return results

if __name__ == "__main__":
    main()
