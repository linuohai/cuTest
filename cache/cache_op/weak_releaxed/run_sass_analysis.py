import os
import subprocess
import re

# 目标架构列表
archs = ['sm_70', 'sm_80', 'sm_90']
source_file = 'test_ld_sass.cu'
# 使用脚本所在目录作为工作目录，避免相对路径误差
work_dir = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd):
    """运行命令并实时输出 stdout/stderr，便于调试查看 ptxas 详细信息。"""
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        print(f"[WARN] Command failed with code {result.returncode}: {cmd}")

def extract_ld_instructions(sass_file):
    if not os.path.exists(sass_file):
        return ["File not found"]
    
    instructions = []
    with open(sass_file, 'r') as f:
        lines = f.readlines()
        
    current_func = None
    last_header = None
    
    # 兼容 cuobjdump / nvdisasm：
    func_pattern_cuobjdump = re.compile(r'Function\s*:\s*(\w+)')
    func_pattern_text_section = re.compile(r'\.text\.(\w+)')
    func_pattern_label = re.compile(r'^\s*(\w+):')
    target_funcs = {'k_ldst_relaxed_weak'}
    
    for line in lines:
        # 识别函数名（cuobjdump: "Function : xxx"; nvdisasm: ".text.xxx" 或 "xxx:"）
        new_func = None
        m1 = func_pattern_cuobjdump.search(line)
        m2 = func_pattern_text_section.search(line)
        m3 = func_pattern_label.match(line)
        if m1:
            new_func = m1.group(1)
        elif m2:
            new_func = m2.group(1)
        elif m3:
            new_func = m3.group(1)

        if new_func is not None:
            current_func = new_func if new_func in target_funcs else None
            if current_func in target_funcs and last_header != current_func:
                instructions.append(f"\n--- Function: {current_func} ---")
                last_header = current_func

        if current_func not in target_funcs:
            continue

        # Filter instructions inside relevant functions
        if current_func in target_funcs:
            # 只抓我们关心的全局访存指令，过滤掉参数装载/desc准备（LDC/ULDC 等）
            if line.strip().startswith('/*'):
                parts = line.split(';')
                code_part = parts[0]
                if ('LDG' in code_part) or ('STG' in code_part):
                    instructions.append(line.strip())

    return instructions

def main():
    original_cwd = os.getcwd()
    os.chdir(work_dir)
    
    results = {}

    for arch in archs:
        cubin_file = f"test_ld_sass.{arch}.cubin"
        sass_file = f"test_ld_sass.{arch}.sass"
        
        print(f"Processing {arch}...")
        # 添加 -Xptxas -v 以输出寄存器/访存等统计，-lineinfo 便于调试
        cmd_compile = (
            f"nvcc -arch={arch} -cubin -lineinfo "
            f"-Xptxas -v {source_file} -o {cubin_file}"
        )
        run_cmd(cmd_compile)
        
        if os.path.exists(cubin_file):
            # 使用 nvdisasm 取代 cuobjdump
            cmd_sass = f"nvdisasm -g {cubin_file} > {sass_file}"
            run_cmd(cmd_sass)
            results[arch] = extract_ld_instructions(sass_file)
        else:
            results[arch] = ["Compilation failed"]

    print("\n" + "="*60)
    print("SASS Analysis Result")
    print("="*60)
    
    for arch in archs:
        print(f"\n[{arch}]")
        for line in results[arch]:
            print(line)

    os.chdir(original_cwd)

if __name__ == '__main__':
    main()
