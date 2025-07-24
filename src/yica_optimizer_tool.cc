
// YICA 转换优化工具 - 统一命令行接口
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>

// 声明外部接口
extern "C" {
    void yica_optimizer_init();
    const char* yica_optimize_code(const char* input_code, const char* target_backend);
    bool yica_is_backend_available(const char* backend);
}

void show_help() {
    std::cout << "YICA 转换优化代码工具\n" << std::endl;
    std::cout << "用法: yica_optimizer [选项] <输入文件>\n" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --backend <cpu|gpu|yica|auto>  目标后端 (默认: auto)" << std::endl;
    std::cout << "  --output <文件>                输出文件" << std::endl;
    std::cout << "  --optimize <0|1|2|3>           优化级别 (默认: 2)" << std::endl;
    std::cout << "  --help                         显示帮助信息" << std::endl;
    std::cout << "\n设计理念:" << std::endl;
    std::cout << "  - 可在任何环境编译和运行" << std::endl;
    std::cout << "  - 按需选择后端以减少编译时间" << std::endl;
    std::cout << "  - 自包含所有必要组件" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string input_file;
    std::string output_file;
    std::string backend = "auto";
    int opt_level = 2;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help();
            return 0;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--optimize" && i + 1 < argc) {
            opt_level = std::stoi(argv[++i]);
        } else if (arg[0] != '-') {
            input_file = arg;
        }
    }
    
    if (input_file.empty()) {
        std::cerr << "错误: 请指定输入文件" << std::endl;
        show_help();
        return 1;
    }
    
    // 初始化优化器
    yica_optimizer_init();
    
    // 检查后端可用性
    if (!yica_is_backend_available(backend.c_str())) {
        std::cerr << "警告: 后端 " << backend << " 可能不可用，但仍会生成代码" << std::endl;
    }
    
    // 读取输入文件
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开输入文件: " << input_file << std::endl;
        return 1;
    }
    
    std::string input_code((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    file.close();
    
    // 执行优化转换
    const char* optimized_code = yica_optimize_code(input_code.c_str(), backend.c_str());
    
    // 输出结果
    if (output_file.empty()) {
        std::cout << optimized_code << std::endl;
    } else {
        std::ofstream out_file(output_file);
        if (!out_file.is_open()) {
            std::cerr << "错误: 无法创建输出文件: " << output_file << std::endl;
            return 1;
        }
        out_file << optimized_code;
        out_file.close();
        std::cout << "优化结果已保存到: " << output_file << std::endl;
    }
    
    std::cout << "\n转换优化完成！" << std::endl;
    return 0;
}
