#!/usr/bin/env python3
"""
查看 GPT2 tokenizer 二进制文件内容的脚本

文件格式：
- Header (1024 字节):
  - [0:4]   magic_number (uint32): 20240328 = GPT-2, 20240801 = LLaMA-3
  - [4:8]   version (uint32): 1 或 2
  - [8:12]  vocab_size (uint32): 词汇表大小 (GPT-2 是 50257)
  - [12:16] 预留/其他数字 (GPT-2 是 50256)
- Token 表: 每个 token 由 1 字节长度 + 内容组成
"""

import struct
import sys


def parse_tokenizer(filepath):
    with open(filepath, 'rb') as f:
        # 读取 Header
        header = f.read(1024)
        
        magic_number = struct.unpack('<I', header[0:4])[0]
        version = struct.unpack('<I', header[4:8])[0]
        vocab_size = struct.unpack('<I', header[8:12])[0]
        num4 = struct.unpack('<I', header[12:16])[0]
        
        print("=" * 60)
        print("Tokenizer 文件头信息")
        print("=" * 60)
        print(f"文件路径: {filepath}")
        print(f"Magic Number: {magic_number} (0x{magic_number:08x})")
        print(f"Version: {version}")
        print(f"Vocab Size: {vocab_size}")
        print(f"Extra Num: {num4}")
        
        # 根据 magic_number 判断模型类型
        model_type = "Unknown"
        eot_token = None
        if magic_number == 20240328:
            model_type = "GPT-2"
            eot_token = 50256
        elif magic_number == 20240801:
            model_type = "LLaMA-3"
            eot_token = 128001
        print(f"模型类型: {model_type}")
        print(f"EOT Token ID: {eot_token}")
        print()
        
        # 读取 Token 表
        print("=" * 60)
        print(f"Token 表 (共 {vocab_size} 个 tokens)")
        print("=" * 60)
        
        token_table = []
        for i in range(vocab_size):
            # 读取 1 字节长度
            len_bytes = f.read(1)
            if len(len_bytes) < 1:
                print(f"警告: 文件提前结束，只读取了 {i} 个 tokens")
                break
            length = len_bytes[0]
            
            # 读取 token 内容
            token_bytes = f.read(length)
            if len(token_bytes) < length:
                print(f"警告: token {i} 读取不完整")
                break
            
            # 尝试解码为字符串
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_str = token_bytes.decode('utf-8', errors='replace')
            
            token_table.append((i, length, token_bytes, token_str))
        
        return token_table


def display_tokens(token_table, start=0, count=100, show_bytes=True):
    """显示指定范围的 tokens"""
    end = min(start + count, len(token_table))
    print(f"显示 tokens [{start}:{end}] (共 {end - start} 个):")
    print("-" * 60)
    
    for i, length, token_bytes, token_str in token_table[start:end]:
        # 对特殊字符进行转义显示
        display_str = repr(token_str)
        if show_bytes:
            bytes_hex = token_bytes.hex()
            print(f"Token {i:6d}: len={length:3d}, hex={bytes_hex:20s}, str={display_str}")
        else:
            print(f"Token {i:6d}: len={length:3d}, str={display_str}")
    print()


def search_token(token_table, keyword):
    """搜索包含关键字的 token"""
    print(f"搜索包含 '{keyword}' 的 tokens:")
    print("-" * 60)
    found = 0
    for i, length, token_bytes, token_str in token_table:
        if keyword in token_str:
            print(f"Token {i:6d}: len={length:3d}, str={repr(token_str)}")
            found += 1
            if found >= 50:  # 最多显示50个结果
                print(f"... (还有更多信息，只显示前50个)")
                break
    print(f"找到 {found} 个匹配的 tokens\n")


def decode_tokens(token_table, token_ids):
    """将 token IDs 解码为文本"""
    result = []
    for tid in token_ids:
        if 0 <= tid < len(token_table):
            result.append(token_table[tid][3])
        else:
            result.append(f"<invalid_token:{tid}>")
    return "".join(result)


def main():
    filepath = "Data/gpt2_tokenizer.bin"
    
    print("正在解析 tokenizer 文件...\n")
    token_table = parse_tokenizer(filepath)
    
    # 显示基本信息
    print(f"\n成功加载 {len(token_table)} 个 tokens\n")
    
    # 显示前100个 tokens
    display_tokens(token_table, start=0, count=100, show_bytes=True)
    
    # 显示一些特殊位置的 tokens
    print("显示一些特殊位置的 tokens:")
    print("-" * 60)
    for pos in [0, 1, 2, 10, 100, 1000, 5000, 10000, 20000, 50255, 50256]:
        if pos < len(token_table):
            i, length, token_bytes, token_str = token_table[pos]
            print(f"Token {i:6d}: len={length:3d}, str={repr(token_str)}")
    print()
    
    # 搜索一些常用词
    print("搜索常用词:")
    search_token(token_table, "the")
    search_token(token_table, "中国")
    
    # 解码示例
    print("解码示例 (prompt: 'The meaning of life is'):")
    # GPT-2 prompt: [464, 3616, 286, 1204, 318]
    prompt_ids = [464, 3616, 286, 1204, 318]
    decoded = decode_tokens(token_table, prompt_ids)
    print(f"Token IDs: {prompt_ids}")
    print(f"Decoded: {repr(decoded)}")
    print()
    
    # 交互模式
    print("=" * 60)
    print("交互模式 - 输入命令查看 tokens")
    print("命令:")
    print("  s <start> <count>  - 显示从 start 开始的 count 个 tokens")
    print("  f <keyword>        - 搜索包含 keyword 的 tokens")
    print("  d <id1> <id2> ...  - 解码指定的 token IDs")
    print("  q                  - 退出")
    print("=" * 60)
    
    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == 'q':
                break
            elif cmd[0] == 's' and len(cmd) >= 3:
                start = int(cmd[1])
                count = int(cmd[2])
                display_tokens(token_table, start=start, count=count)
            elif cmd[0] == 'f' and len(cmd) >= 2:
                keyword = cmd[1]
                search_token(token_table, keyword)
            elif cmd[0] == 'd' and len(cmd) >= 2:
                ids = [int(x) for x in cmd[1:]]
                decoded = decode_tokens(token_table, ids)
                print(f"Token IDs: {ids}")
                print(f"Decoded: {repr(decoded)}")
            else:
                print("未知命令")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n再见!")


if __name__ == "__main__":
    main()
