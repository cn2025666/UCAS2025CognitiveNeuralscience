#!/usr/bin/env python3
# filepath: /Users/yinchao/Desktop/tex_Template/compile.py

import subprocess
import os
import sys
import glob
import re
from pathlib import Path

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        
        # 显示输出信息
        if result.stdout:
            print("标准输出:")
            print(result.stdout[-1000:])  # 只显示最后1000个字符
        
        if result.returncode != 0:
            print(f"错误: {description} 失败 (返回码: {result.returncode})")
            if result.stderr:
                print("错误信息:")
                print(result.stderr[-1000:])  # 只显示最后1000个字符
            
            # 检查log文件
            log_file = "allin.log"
            if os.path.exists(log_file):
                print("\n=== LaTeX日志文件内容 (最后50行) ===")
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines[-50:]:
                        print(line.rstrip())
                print("=== 日志文件结束 ===")
            
            return False
        else:
            print(f"成功: {description} 完成")
            return True
    except subprocess.TimeoutExpired:
        print(f"超时: {description} 执行超时")
        return False
    except Exception as e:
        print(f"异常: {description} 时发生错误: {e}")
        return False

def check_tex_file(tex_file):
    """检查tex文件的基本问题"""
    print(f"检查 {tex_file} 文件...")
    
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查基本结构
        if '\\begin{document}' not in content:
            print("❌ 缺少 \\begin{document}")
            return False
        
        if '\\end{document}' not in content:
            print("❌ 缺少 \\end{document}")
            return False
        
        # 检查未闭合的环境
        begin_count = content.count('\\begin{cnabstract}')
        end_count = content.count('\\end{cnabstract}')
        if begin_count != end_count:
            print(f"❌ cnabstract环境不匹配: \\begin{{{begin_count}}} vs \\end{{{end_count}}}")
            return False
        
        print("✅ 基本结构检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False

def fix_tex_file(tex_file):
    """修复常见的tex文件问题"""
    print("尝试修复常见问题...")
    
    try:
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 修复缺失的\end{cnabstract}
        if '\\begin{cnabstract}' in content and '\\end{cnabstract}' not in content:
            # 在\addcontentsline之前添加\end{cnabstract}
            content = content.replace(
                '\\addcontentsline{toc}{chapter}{序言}',
                '\\end{cnabstract}\n    \\addcontentsline{toc}{chapter}{序言}'
            )
            print("✅ 修复了缺失的 \\end{cnabstract}")
        
        # 修复文本中的下划线问题 - 将裸露的下划线转义
        # 但要避免影响已经在数学模式或命令中的下划线
        def fix_underscores(match):
            text = match.group(0)
            # 如果已经转义或在命令中，不要修改
            if '\\texttt{' in text or '\\_' in text:
                return text
            # 替换裸露的下划线
            return text.replace('_', '\\_')
        
        # 使用正则表达式找到可能有问题的区域
        patterns_to_fix = [
            r'\\section\{[^}]*_[^}]*\}',
            r'\\subsection\{[^}]*_[^}]*\}',
            r'\\subsubsection\{[^}]*_[^}]*\}',
            r'\\addcontentsline\{toc\}\{[^}]*\}\{[^}]*_[^}]*\}'
        ]
        
        for pattern in patterns_to_fix:
            content = re.sub(pattern, fix_underscores, content)
        
        # 特别处理一些已知的问题标题
        problematic_titles = [
            '数据预处理:backup文件夹',
            '数据分析:chart文件夹', 
            '原始数据文件修正:pre_mod文件夹',
            '模型训练和测试:train_test文件夹'
        ]
        
        for title in problematic_titles:
            if title in content:
                fixed_title = title.replace('_', '\\_')
                content = content.replace(title, fixed_title)
                print(f"✅ 修复标题中的下划线: {title} -> {fixed_title}")
        
        # 创建备份
        backup_file = f"{tex_file}.backup"
        if not os.path.exists(backup_file):
            with open(backup_file, 'w', encoding='utf-8') as f:
                with open(tex_file, 'r', encoding='utf-8') as original:
                    f.write(original.read())
            print(f"✅ 创建备份文件: {backup_file}")
        
        # 写入修复后的内容
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ 修复文件时出错: {e}")
        return False

def fix_toc_file():
    """修复目录文件中的下划线问题"""
    toc_file = "allin.toc"
    if not os.path.exists(toc_file):
        return True
    
    print("检查并修复.toc文件中的下划线问题...")
    
    try:
        with open(toc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换目录文件中裸露的下划线
        original_content = content
        content = re.sub(r'(?<!\\)_', '\\_', content)
        
        if content != original_content:
            with open(toc_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ 修复了.toc文件中的下划线问题")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复.toc文件时出错: {e}")
        return False

def clean_intermediate_files(tex_file):
    """清理中间文件，仅保留.tex和.pdf文件"""
    base_name = os.path.splitext(tex_file)[0]
    
    # 定义要删除的中间文件扩展名
    intermediate_extensions = [
        '.aux', '.log', '.out', '.toc', '.lof', '.lot', '.bbl', '.blg',
        '.fdb_latexmk', '.fls', '.synctex.gz', '.synctex(busy)',
        '.nav', '.snm', '.vrb', '.figlist', '.makefile', '.xdv'
    ]
    
    print("正在清理中间文件...")
    deleted_count = 0
    
    for ext in intermediate_extensions:
        pattern = f"{base_name}{ext}"
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"删除: {file_path}")
                deleted_count += 1
            except OSError as e:
                print(f"无法删除 {file_path}: {e}")
    
    # 清理可能的临时文件
    temp_patterns = [
        "*.auxlock",
        "*.figlist", 
        "*.makefile",
        "*.xdv"
    ]
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"删除临时文件: {file_path}")
                deleted_count += 1
            except OSError as e:
                print(f"无法删除 {file_path}: {e}")
    
    print(f"清理完成，共删除 {deleted_count} 个中间文件")

def compile_latex(tex_file):
    """编译LaTeX文件"""
    if not os.path.exists(tex_file):
        print(f"错误: 文件 {tex_file} 不存在")
        return False
    
    base_name = os.path.splitext(tex_file)[0]
    
    print(f"开始编译 {tex_file}")
    print("=" * 50)
    
    # 检查并修复tex文件
    if not check_tex_file(tex_file):
        print("尝试修复tex文件...")
        if not fix_tex_file(tex_file):
            return False
        if not check_tex_file(tex_file):
            return False
    else:
        # 即使检查通过，也尝试修复下划线问题
        fix_tex_file(tex_file)
    
    # 清理旧的中间文件
    clean_intermediate_files(tex_file)
    
    # 第一次编译
    if not run_command(f"xelatex -interaction=nonstopmode {tex_file}", "第一次XeLaTeX编译"):
        return False
    
    # 修复.toc文件中的问题
    fix_toc_file()
    
    # 第二次编译（生成目录）
    if not run_command(f"xelatex -interaction=nonstopmode {tex_file}", "第二次XeLaTeX编译"):
        return False
    
    # 检查PDF是否生成
    pdf_file = f"{base_name}.pdf"
    if os.path.exists(pdf_file):
        print(f"✅ 编译成功！生成文件: {pdf_file}")
        file_size = os.path.getsize(pdf_file)
        print(f"PDF文件大小: {file_size} 字节")
    else:
        print("❌ 编译失败：未生成PDF文件")
        return False
    
    # 清理中间文件
    clean_intermediate_files(tex_file)
    
    print("=" * 50)
    print("编译完成！")
    return True

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        tex_file = sys.argv[1]
    else:
        # 默认编译allin.tex
        tex_file = "allin.tex"
    
    # 确保在正确的目录中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"编译文件: {tex_file}")
    
    # 检查XeLaTeX是否可用
    try:
        subprocess.run(["xelatex", "--version"], capture_output=True, check=True)
        print("✅ XeLaTeX 可用")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: XeLaTeX 未安装或不在PATH中")
        return 1
    
    # 编译LaTeX文件
    if compile_latex(tex_file):
        return 0
    else:
        print("\n如果问题持续存在，请检查:")
        print("1. 字体是否正确安装")
        print("2. 图片文件是否存在")
        print("3. 宏包是否完整安装")
        print("4. 标题中是否有特殊字符需要转义")
        return 1

if __name__ == "__main__":
    sys.exit(main())