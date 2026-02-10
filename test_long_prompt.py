#!/usr/bin/env python3
"""
Test script with long prompt to ensure seq_len > 2048
This helps test the hit/miss logic properly.
"""

import requests
import json
import os

def generate_long_prompt(target_tokens=3000):
    base_question = """
请仔细阅读以下长篇文章，然后回答问题。

【文章开始】

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等多个领域。

深度学习是机器学习的一个重要分支，它通过构建具有多个隐藏层的神经网络来学习数据的层次化表示。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性的进展。

Transformer架构是一种基于自注意力机制的神经网络架构，由Vaswani等人在2017年提出。它摒弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），完全基于注意力机制来捕捉序列中元素之间的依赖关系。Transformer架构已经成为自然语言处理领域的主流架构，广泛应用于机器翻译、文本生成、问答系统等任务。

大型语言模型（Large Language Model，LLM）是一种基于深度学习的语言模型，通常包含数十亿甚至数千亿个参数。这些模型通过在大规模文本数据上进行预训练，学习到了丰富的语言知识和世界知识。代表性的大型语言模型包括GPT系列、BERT、T5、LLaMA等。

"""

    # 重复内容来增加长度
    filler_paragraph = """
神经网络的训练过程通常包括前向传播和反向传播两个阶段。在前向传播阶段，输入数据通过网络的各层进行计算，最终得到输出结果。在反向传播阶段，通过计算损失函数关于网络参数的梯度，并使用优化算法（如随机梯度下降）来更新网络参数，从而使网络的输出逐渐接近期望的目标。

注意力机制是一种让模型能够关注输入序列中最相关部分的技术。在处理长序列时，注意力机制能够帮助模型有效地捕捉远距离依赖关系。自注意力（Self-Attention）是注意力机制的一种特殊形式，它允许序列中的每个位置都与其他所有位置进行交互，从而学习到序列内部的复杂关系。

稀疏注意力（Sparse Attention）是一种优化技术，通过只关注序列中最重要的部分来降低计算复杂度。传统的全注意力机制的计算复杂度是O(n²)，其中n是序列长度。稀疏注意力通过只计算部分注意力权重，可以将复杂度降低到O(n log n)甚至O(n)。

"""

    # 构建长文本
    long_text = base_question
    
    # 估算每个段落约 200 个 token，重复直到超过目标
    while len(long_text) < target_tokens * 2:  # 粗略估计：2字符≈1token
        long_text += filler_paragraph
    
    # 添加结束问题，使用 DeepSeek 对话模板格式
    long_text = "<｜begin▁of▁sentence｜><｜User｜>" + long_text + """
【文章结束】

问题：根据以上文章，请简要解释什么是稀疏注意力机制，以及它的主要优势是什么？<｜Assistant｜>"""
    
    return long_text


def test_long_prompt(host="http://localhost", port=30000):
    """发送长 prompt 测试请求"""
    
    url = f"{host}:{port}/generate"
    
    # 生成长 prompt（目标 3000+ tokens）
    prompt = generate_long_prompt(target_tokens=3500)
    
    print(f"Prompt length (chars): {len(prompt)}")
    print(f"Estimated tokens: ~{len(prompt) // 2}")
    print("-" * 60)
    
    data = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": 200,
        }
    }
    
    try:
        print("Sending request...")
        response = requests.post(url, json=data, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        
        print("\nResponse received!")
        if "text" in result:
            print(f"\nGenerated text:\n{result['text']}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    host = os.getenv("SGLANG_HOST", "http://localhost")
    port = int(os.getenv("SGLANG_PORT", "8418"))
    
    print("=" * 60)
    print("Long Prompt Test for Hit/Miss Logic Validation and Index Estimation")
    print("=" * 60)
    
    test_long_prompt(host=host, port=port)
