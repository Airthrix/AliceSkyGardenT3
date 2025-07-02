# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
#MIT License
#
#Copyright (c) 2025 钱益聪 <airthrix@163.com>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software **for personal, non-commercial use only**, subject to the following conditions:
#
#1. The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#2. **No person may distribute, sublicense, sell, or otherwise commercialize**
#   copies of the Software without prior written consent from the copyright holder.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
######################################################API_KEY 
import os
import logging
from transformers.utils import logging as hf_logging
logger = hf_logging.get_logger(__name__)

def safe_auto_docstring(obj):
    if not hasattr(obj, '__module__'):
        return obj
    try:
        module_path = obj.__module__.split('.')
        if len(module_path) < 3 or module_path[-3] != 'models':
            raise IndexError("Invalid module path depth for docstring generation.")
    except Exception as e:
        logger.warning("自动文档字符串失败: %s", str(e))
        return obj
    return obj

try:
    from transformers.utils.args_doc import auto_docstring as original_auto_docstring
    auto_docstring = safe_auto_docstring
except ImportError:
    pass
######################################################API_KEY


import argparse
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from modeling_aliceskygarden_t3 import AliceSkyGardenT3ForCausalLM, AliceSkyGardenT3Config
import logging
import os
from datetime import datetime
import torch.nn.functional as F
from safetensors.torch import load_file

##############################################KEY
import sys
import os

# 添加当前目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
##############################################KEY


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=0.7, type=float, required=False, help='生成温度，较低的值会使输出更加确定性')
    parser.add_argument('--topk', default=10, type=int, required=False, help='只保留概率最高的k个token，0表示禁用')
    parser.add_argument('--topp', default=0.0, type=float, required=False, help='累积概率阈值，只保留总概率达到p的token')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.json', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='model/epoch40', type=str, required=False, help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.2, type=float, required=False, help="重复惩罚参数，大于1会降低已生成token的概率")
    parser.add_argument('--max_len', type=int, default=25, help='每个回复的最大长度')
    parser.add_argument('--max_history_len', type=int, default=3, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()

def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def count_vocab_size(vocab_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_size = len(f.readlines())
        print(f"词表大小: {vocab_size}")
        return vocab_size
    except Exception as e:
        print(f"读取词表出错: {str(e)}")
        raise


def top_k_top_p_filtering(logits, tokenizer, response, top_k=0, top_p=0.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """
    @param logits: 模型输出的logits
    @param tokenizer: tokenizer对象
    @param response: 当前已生成的response列表
    """
    assert logits.dim() == 1
    
    # 降低[SEP]的基础概率
    sep_token_id = tokenizer.sep_token_id
    logits[sep_token_id] = logits[sep_token_id] - 5.0
    
    # 如果response长度太短，完全禁用[SEP]
    if len(response) < 2:
        logits[sep_token_id] = filter_value
    
    # Top-K 过滤
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    # Top-P 过滤
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits

def load_safetensors_model(model_path, config):
    model = AliceSkyGardenT3ForCausalLM(config)
    state_dict_path = os.path.join(model_path, "compressed_weights.safetensors")
    
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"模型权重文件未找到：{state_dict_path}")
    
    try:
        # 加载权重文件
        state_dict = load_file(state_dict_path)
        print(f"加载的权重文件大小: {len(state_dict)} 个参数")
        print(f"权重文件中的键: {list(state_dict.keys())[:5]}...")
        
        # 检查权重文件的完整性
        expected_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        missing_keys = expected_keys - loaded_keys
        unexpected_keys = loaded_keys - expected_keys
        
        if missing_keys:
            print(f"警告：缺少的权重键: {missing_keys}")
        if unexpected_keys:
            print(f"警告：未预期的权重键: {unexpected_keys}")
            
        # 加载权重到模型
        model.load_state_dict(state_dict, strict=False)  # 使用strict=False允许部分加载
        
        # 验证模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
        # 验证权重是否正确加载
        print("\n验证权重加载:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
            if len(param.shape) >= 2:  # 对于2D以上的张量，检查是否都是0
                zero_ratio = (param == 0).float().mean().item()
                if zero_ratio > 0.9:  # 如果90%以上都是0
                    print(f"警告: {name} 包含大量零值 ({zero_ratio*100:.1f}%)")
        
        return model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise

def main():
    args = set_args()
    logger = create_logger(args)
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    logger.info(f'使用设备: {device}')
    
    # 初始化BPE分词器
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]'})
    
    # 加载配置
    config = AliceSkyGardenT3Config.from_json_file(os.path.join(args.model_path, "config.json"))
    print("模型配置:", config.to_json_string())
    
    # 验证词表大小与配置是否匹配
    if len(tokenizer) != config.vocab_size:
        logger.warning(f"警告：词表大小 ({len(tokenizer)}) 与模型配置中的vocab_size ({config.vocab_size}) 不匹配")
    
    # 加载模型
    model = AliceSkyGardenT3ForCausalLM(config)
    logger.info("开始加载压缩模型...")
    model = model.load_compressed_model(args.model_path, device=device)
    logger.info("模型加载完成")

    # 验证模型是否在正确设备上
    logger.info(f"模型当前设备: {next(model.parameters()).device}")
    model.eval()

    # 验证前向传播
    sample_input = torch.tensor([[tokenizer.cls_token_id]], device=device)
    with torch.no_grad():
        output = model(sample_input)
        logger.info(f"模型验证输出形状: {output.logits.shape}")


#    model = load_safetensors_model(args.model_path, config)
#    model.to(device)
#    model.eval()
    
    # 初始化对话历史和samples_file
    history = []
    samples_file = None
    if args.save_samples_path:
        os.makedirs(args.save_samples_path, exist_ok=True)
        samples_file = open(os.path.join(args.save_samples_path, 'samples.txt'), 'a', encoding='utf8')
        samples_file.write(f"聊天记录 {datetime.now()}:\n")
    
    print('开始和AliceSkyGardenT3聊天，请输入')
    
    while True:
        try:
            text = input("\nuser: ")
            if samples_file:
                samples_file.write(f"user: {text}\n")
            
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(text_ids)
            
            input_ids = [tokenizer.cls_token_id]
            for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(tokenizer.sep_token_id)
            
            input_ids = torch.tensor(input_ids).long().to(device)
            input_ids = input_ids.unsqueeze(0)
            
            response = []
            for _ in range(args.max_len):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                
                # 应用重复惩罚
                for id in set(response):
                    next_token_logits[id] /= args.repetition_penalty
                
                # 应用温度
                next_token_logits = next_token_logits / args.temperature
                
                # 禁用[UNK]
                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                
                # 过滤logits
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits,
                    tokenizer=tokenizer,
                    response=response,
                    top_k=args.topk,
                    top_p=args.topp
                )
                
                # 采样下一个token
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token == tokenizer.sep_token_id and len(response) >= 2:
                    break
                
                response.append(next_token.item())
                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            
            history.append(response)
            text = tokenizer.convert_ids_to_tokens(response)
            print("chatbot: " + "".join(text))
            
            if samples_file:
                samples_file.write(f"chatbot: {''.join(text)}\n")
                
        except KeyboardInterrupt:
            if samples_file:
                samples_file.close()
            break
        
        except Exception as e:
            logger.error(f"发生错误: {str(e)}")
            continue

if __name__ == '__main__':
    main()
