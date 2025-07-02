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
# Encrypted. As it involves trade secrets, it is not fully open source at present.  API
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
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from transformers import  AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from modeling_aliceskygarden_t3 import AliceSkyGardenT3ForCausalLM, AliceSkyGardenT3Config
import pickle
import json
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import gc
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizerFast
import psutil
import threading


# 日志设置
def create_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

# 数据集类
class MyDataset(Dataset):
    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len
        
    def __getitem__(self, index):
        input_ids = self.input_list[index][:self.max_len]
        return torch.tensor(input_ids, dtype=torch.long)
        
    def __len__(self):
        return len(self.input_list)

# 分块数据集加载
class IndexedDataset(Dataset):
    def __init__(self, path, indices, max_len):
        self.path = path
        self.indices = indices  # 直接使用原始索引，不排序
        self.max_len = max_len
        self._chunk_info = self._precompute_chunk_info()
        self._current_chunk = None
        self._current_chunk_num = -1
        
    def _precompute_chunk_info(self):
        chunk_info = []
        current_offset = 0
        with open(self.path, 'rb') as f:
            while True:
                pos = f.tell()
                try:
                    size = len(pickle.load(f))
                    chunk_info.append({
                        'start': current_offset,
                        'end': current_offset + size,
                        'file_pos': pos
                    })
                    current_offset += size
                except EOFError:
                    break
        return chunk_info
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        target_idx = self.indices[idx]
        chunk_num = next((i for i, info in enumerate(self._chunk_info) 
                       if info['start'] <= target_idx < info['end']), None)
        if chunk_num is None:
            raise IndexError(f"Index {target_idx} out of range")
            
        local_idx = target_idx - self._chunk_info[chunk_num]['start']
        if chunk_num != self._current_chunk_num:
            self._load_chunk(chunk_num)
        return torch.tensor(self._current_chunk[local_idx][:self.max_len], dtype=torch.long)
        
    def _load_chunk(self, chunk_num):
        with open(self.path, 'rb') as f:
            f.seek(self._chunk_info[chunk_num]['file_pos'])
            self._current_chunk = pickle.load(f)
            self._current_chunk_num = chunk_num

# 数据整理函数
def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels

# 检查点管理
def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, loss, global_step, checkpoint_path, optimizer_save_mode):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'global_step': global_step,  # 新增
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    
    if optimizer_save_mode == 'full':
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    elif optimizer_save_mode == 'minimal':
        checkpoint['optimizer_info'] = {
            'param_groups': [{
                'lr': group['lr'],
                'betas': group['betas'],
                'eps': group['eps'],
                'weight_decay': group['weight_decay'],
            } for group in optimizer.param_groups],
            'step': optimizer.state_dict().get('step', 0),
        }
        
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, optimizer_save_mode):
    checkpoint = torch.load(checkpoint_path)
    
    # 模型和调度器状态
    model.load_state_dict(checkpoint.get('model_state_dict', model.state_dict()))
    scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
    
    # 加载优化器
    if optimizer_save_mode == 'full' and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif optimizer_save_mode == 'minimal' and 'optimizer_info' in checkpoint:
        # 恢复优化器参数组
        for i, group in enumerate(optimizer.param_groups):
            if i < len(checkpoint['optimizer_info']['param_groups']):
                saved_group = checkpoint['optimizer_info']['param_groups'][i]
                group['lr'] = saved_group['lr']
                group['betas'] = saved_group['betas']
                group['eps'] = saved_group['eps']
                group['weight_decay'] = saved_group['weight_decay']
        # 恢复优化器状态
        if 'step' in checkpoint['optimizer_info']:
            for state in optimizer.state.values():
                state['step'] = checkpoint['optimizer_info']['step']
    
    return (
        checkpoint.get('epoch', 0),
        checkpoint.get('batch_idx', 0),
        checkpoint.get('global_step', 0),
        checkpoint.get('loss', float('inf'))
    )

# GPU内存监控
def print_gpu_memory(logger=None):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        if logger:
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

# 训练epoch
def train_epoch(model, train_dataloader, optimizer, scheduler, logger, epoch, args, global_step, resume_batch_idx=0):
    model.train()
    device = args.device
    total_loss = 0.0  # 用于记录整个epoch的总损失
    total_correct = 0  # 整个epoch的正确预测数
    total_tokens = 0   # 整个epoch的总token数
    scaler = GradScaler(enabled=args.use_amp)
    
    # 梯度累积计数器
    accumulation_steps = 0
    
    # 计算当前epoch最大允许的batch数量
    max_batches = len(train_dataloader.dataset) // args.batch_size
    if len(train_dataloader.dataset) % args.batch_size != 0:
        max_batches += 1

    # 禁用模型缓存以支持梯度检查点
    if hasattr(model, 'module'):
        model.module.config.use_cache = not args.gradient_checkpointing
    else:
        model.config.use_cache = not args.gradient_checkpointing
        
    for batch_idx, (input_ids, labels) in enumerate(train_dataloader, start=resume_batch_idx):
        if batch_idx >= max_batches:
            logger.warning(f"强制终止：当前batch_idx({batch_idx})超过数据集最大batch数量({max_batches})")
            break

        try:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 前向传播在混合精度作用域内
            with autocast(enabled=args.use_amp, dtype=torch.bfloat16 if args.use_bf16 else torch.float16):
                outputs = model(input_ids, labels=labels, global_step=global_step)
                loss = outputs.loss.mean()
                # 根据梯度累积步数缩放损失
                loss = loss / args.gradient_accumulation_steps
            

            # 添加三元稀疏度监控
            ternary_sparsity = getattr(outputs, "ternary_sparsity", 0.0)  # 获取三元稀疏度


            # 在autocast作用域外计算准确率（使用float32确保数值稳定性）
            with torch.no_grad():
                # 只取前n-1个token的logits，用于预测下一个token
                logits = outputs.logits[:, :-1, :].float()  # 移除最后一个token的logits
                preds = torch.argmax(logits, dim=-1)
    
                # 只取后n-1个token作为标签（预测下一个token）
                labels_shifted = labels[:, 1:]  # 移除第一个token的标签
                mask = (labels_shifted != -100)  # 忽略padding部分
    
                correct = (preds[mask] == labels_shifted[mask]).sum().item()
                tokens_in_batch = mask.sum().item()
            
            # 梯度累积
            scaler.scale(loss).backward()
            accumulation_steps += 1
            
            # 累积统计信息
            total_correct += correct
            total_tokens += tokens_in_batch
            # 注意：loss.item()是缩放后的损失，我们需要记录原始损失
            total_loss += loss.item() * args.gradient_accumulation_steps
            
            # 只在达到累积步数时更新参数
            if accumulation_steps % args.gradient_accumulation_steps == 0:
                # 梯度裁剪前必须先unscale
                scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.max_grad_norm, norm_type=2.0
                    )
                    if global_step % args.log_step == 0:
                        logger.info(f"Gradient Norm: {grad_norm:.4f}")
                
                # 执行参数更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 每个实际更新步骤调用一次学习率调度器
                scheduler.step()
                global_step += 1
                
                # 重置累积计数器
                accumulation_steps = 0
            
            # 日志记录（只在参数更新后记录）
            if (global_step % args.log_step == 0) and (accumulation_steps == 0):
                batch_acc = correct / tokens_in_batch if tokens_in_batch > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']
                current_loss = loss.item() * args.gradient_accumulation_steps  # 当前batch的原始损失
                
                logger.info(f"Step {global_step}, batch {batch_idx+1}/{len(train_dataloader)} of epoch {epoch+1}, "
                            f"loss {current_loss:.4f}, acc {batch_acc:.4f}, "
                            f"LR {current_lr:.10f}, Ternary Sparsity {ternary_sparsity:.4f}")  # 新增三元稀疏度
            
            # 检查点保存（只在参数更新后保存）
            if (global_step % args.checkpoint_step == 0) and (accumulation_steps == 0):
                save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, 
                              total_loss / (batch_idx + 1), global_step,
                              args.checkpoint_path, args.optimizer_save_mode)
                logger.info(f"Checkpoint saved at step {global_step}")
            
            # 释放内存
            del input_ids, labels, outputs, logits, preds
            torch.cuda.empty_cache()
            gc.collect()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM detected, reducing batch size")
                args.batch_size = max(1, args.batch_size // 2)
                
                # 重新创建DataLoader
                train_dataloader = DataLoader(
                    train_dataloader.dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    pin_memory=True
                )
                
                # 重新计算最大batch数
                max_batches = len(train_dataloader.dataset) // args.batch_size
                if len(train_dataloader.dataset) % args.batch_size != 0:
                    max_batches += 1
                
                # 重置累积状态
                accumulation_steps = 0
                optimizer.zero_grad()
                
                logger.warning(f"Batch size reduced to {args.batch_size}")
            else:
                logger.error(f"Runtime error: {str(e)}")
                raise e
    
    # 处理最后一个不完整的累积步骤
    if accumulation_steps > 0:
        scaler.unscale_(optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        global_step += 1
    
    # 计算epoch平均损失和准确率
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_acc = total_correct / total_tokens if total_tokens > 0 else 0
    logger.info(f"Epoch {epoch+1} completed: loss {epoch_mean_loss:.4f}, acc {epoch_acc:.4f}")
    
    return epoch_mean_loss, global_step, False


# 验证epoch
def validate_epoch(model, validate_dataloader, logger, epoch, args, global_step):
    model.eval()
    total_loss = 0
    total_correct = 0  # 新增：累计正确预测数
    total_tokens = 0   # 新增：累计token总数
    total_sparsity = 0  # 新增：用于累计稀疏度
    
    with torch.no_grad(), autocast(enabled=args.use_amp):
        for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(input_ids, labels=labels, global_step=global_step)
            total_loss += outputs.loss.mean().item()
            
            # 获取三元稀疏度
            batch_sparsity = getattr(outputs, "ternary_sparsity", 0.0)
            total_sparsity += batch_sparsity

            # 关键修复：正确处理logits和labels的对应关系
            # 只取前n-1个token的logits，用于预测下一个token
            logits = outputs.logits[:, :-1, :].float()  # 移除最后一个token的logits
            preds = torch.argmax(logits, dim=-1)
            
            # 只取后n-1个token作为标签（预测下一个token）
            labels_shifted = labels[:, 1:]  # 移除第一个token的标签
            mask = (labels_shifted != -100)  # 忽略padding部分
            
            correct = (preds[mask] == labels_shifted[mask]).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()

    epoch_mean_loss = total_loss / len(validate_dataloader)
    epoch_acc = total_correct / total_tokens if total_tokens > 0 else 0  # 计算整个验证集的准确率

    # 计算平均稀疏度
    avg_sparsity = total_sparsity / len(validate_dataloader)

    logger.info(f"validate epoch {epoch+1}: loss {epoch_mean_loss:.4f}, acc {epoch_acc:.4f}"
                     f"Ternary Sparsity {avg_sparsity:.4f}")  # 新增三元稀疏度

    # 清理内存
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_mean_loss

# 参数解析
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.95, help='LR decay factor when loss stabilizes')
    parser.add_argument('--device', default='0', type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true')
#    parser.add_argument('--tokenizer_path', default='tokenizer', type=str)  # fix: 改为目录路径
    parser.add_argument('--vocab_path', default='vocab/vocab.json', type=str, required=False, help='词表路径')
    parser.add_argument('--model_config', default='config/config.json', type=str)
    parser.add_argument('--train_path', default='data/train.pkl', type=str)
    parser.add_argument('--max_len', default=2000, type=int)
    parser.add_argument('--log_path', default='data/train.log', type=str)
    parser.add_argument('--log', default=True)
    parser.add_argument('--ignore_index', default=-100, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1.0e-5, type=float)
    parser.add_argument('--eps', default=1.0e-09, type=float)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=0.0, type=float)
    parser.add_argument('--save_model_path', default='model', type=str)
    parser.add_argument('--pretrained_model', default='', type=str)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--val_ratio', type=float, default=0.02)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth')
    parser.add_argument('--checkpoint_step', type=int, default=500)
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--optimizer_save_mode', default='minimal', type=str)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--dataway', type=int, default=0)
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    
    if args.cuda:
        args.device = f'cuda:{args.device}'
    else:
        args.device = 'cpu'
        
    return args

# 数据加载函数
def load_dataset(logger, args):
    with open(args.train_path, 'rb') as f:
        data = pickle.load(f)
        
    train_indices, val_indices = train_test_split(
        np.arange(len(data)), test_size=args.val_ratio, random_state=42)
        
    train_dataset = IndexedDataset(args.train_path, train_indices, args.max_len)
    validate_dataset = IndexedDataset(args.train_path, val_indices, args.max_len)
    
    return train_dataset, validate_dataset

# 主训练函数
def main():
    args = set_args()
    logger = create_logger(args.log_path)
    logger.info(f"Using device: {args.device}")
    
    # 1. 加载tokenizer和模型配置
    # fix: 使用AutoTokenizer加载整个目录，确保加载所有配置文件
#    tokenizer = AutoTokenizer.from_pretrained(
#        args.tokenizer_path,
#        trust_remote_code=True,
#        use_fast=True,  # fix: 启用fast模式以兼容所有配置文件
#        padding_side='left',  # fix: 与交互代码一致
#        add_bos_token=True,  # fix: 与交互代码一致
#        add_eos_token=False,  # fix: 与交互代码一致
#        do_lower_case=False,  # fix: 与交互代码一致
#        clean_up_tokenization_spaces=True,  # fix: 与交互代码一致
#        tokenizer_config={
#            "handle_chinese_chars": True,  # fix: 显式处理中文
#            "strip_accents": False,
#            "remove_control_chars": True,
#        }
#    )

    # 初始化tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'sep_token': '[SEP]', 'unk_token': '[UNK]'})
    args.sep_id = tokenizer.sep_token_id
    args.pad_id = tokenizer.pad_token_id
    args.cls_id = tokenizer.cls_token_id

    
    # fix: 确保pad_token存在
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
#    logger.info(f"Tokenizer loaded from: {args.tokenizer_path}")
    logger.info(f"Tokenizer loaded from: {args.vocab_path}")
    logger.info(f"Tokenizer contains {len(tokenizer)} tokens")
    
    with open(args.model_config) as f:
        config_dict = json.load(f)
    config_dict['vocab_size'] = len(tokenizer)
    model_config = AliceSkyGardenT3Config(**config_dict)

    # 2. 初始化模型
    if args.pretrained_model:
        model = AliceSkyGardenT3ForCausalLM.from_pretrained(args.pretrained_model, config=model_config)
    else:
        model = AliceSkyGardenT3ForCausalLM(config=model_config)

    # 新增：梯度检查点启用
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # 3. 加载数据集
    train_dataset, validate_dataset = load_dataset(logger, args)
    
    # 4. 创建DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 初始shuffle，恢复时会调整
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # 新增验证集DataLoader
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 5. 初始化优化器和调度器
    # 新增：使用AdamW优化器并添加权重衰减
    optimizer = AdamW(model.parameters(), 
                     lr=args.lr, 
                     eps=args.eps,
                     betas=(0.9, 0.98),  # 增加动量稳定性
                     weight_decay=0.01,  # 添加权重衰减防止过拟合
                     amsgrad=True  # 使用AMSGrad变体防止梯度爆炸
    )
    

    # 关键修复：准确计算总步数
    # 每个epoch的实际更新次数 = 总批次 / 累积步数
    updates_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if len(train_dataloader) % args.gradient_accumulation_steps != 0:
        updates_per_epoch += 1
    
    total_steps = updates_per_epoch * args.epochs

    # 记录计算细节以便调试
    logger.info(f"Total steps calculation:")
    logger.info(f"  Dataloader length: {len(train_dataloader)} batches")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Updates per epoch: {updates_per_epoch}")
    logger.info(f"  Total epochs: {args.epochs}")
    logger.info(f"  Total training steps: {total_steps}")
    logger.info(f"  Warmup steps: {args.warmup_steps}")

    # 确保总步数不小于warmup步数
    if total_steps < args.warmup_steps:
        logger.warning(f"总步数({total_steps})小于warmup步数({args.warmup_steps}), 调整warmup步数为总步数的10%")
        args.warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 6. 恢复训练状态
    start_epoch = 0
    resume_batch_idx = 0
    global_step = 0
    
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        try:
            start_epoch, resume_batch_idx, global_step, _ = load_checkpoint(
                model, optimizer, scheduler, args.resume_checkpoint, args.optimizer_save_mode
            )
            logger.info(f"Resuming from epoch {start_epoch+1}, batch {resume_batch_idx+1}")
            
            # 恢复后禁用shuffle
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
#                shuffle=False,  # 恢复训练时不shuffle
                collate_fn=collate_fn,
                pin_memory=True
            )
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            start_epoch = 0
            resume_batch_idx = 0
            global_step = 0

    # 7. 训练循环
    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step, terminated = train_epoch(
            model, train_dataloader, optimizer, scheduler, logger,
            epoch, args, global_step, resume_batch_idx
        )
        resume_batch_idx = 0  # 重置批次索引
        
        # ... 验证和保存逻辑 ...
        
        # 验证步骤
        val_loss = validate_epoch(model, validate_dataloader, logger, epoch, args, global_step)


        # 早停机制

                
        # 保存模型
#        model_path = os.path.join(args.save_model_path, f'epoch{epoch+1}')
#        os.makedirs(model_path, exist_ok=True)
#        model.save_pretrained(model_path)
#        logger.info(f"Model saved to {model_path}")

        # 压缩并保存最终压缩模型 #save
        model_path = os.path.join(args.save_model_path, f'epoch{epoch+1}')
        logger.info("Starting model compression...")
        model.compress_model_weights()
        if model.save_compressed_model(model_path):
            logger.info(f"压缩模型已保存到 {model_path}")
        else:
            logger.warning("压缩模型保存失败")

        # 添加调试日志
        logger.info(f"准备进入下一个 epoch (current epoch: {epoch+1})")

        # 添加硬件状态检查
        logger.info(f"CUDA内存状态 - 已分配: {torch.cuda.memory_allocated()/1024**2:.2f}MB | 保留: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        logger.info(f"CPU内存使用: {psutil.Process().memory_info().rss/1024**2:.2f}MB")
        logger.info(f"当前线程状态: {threading.enumerate()}")


if __name__ == '__main__':
    main()
