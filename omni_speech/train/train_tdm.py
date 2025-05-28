# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import pdb
import whisper
from dataclasses import dataclass, field
import json
import logging
import random
import pathlib
from typing import Dict, Optional, Sequence, List
from typing import Tuple

import torch
import numpy as np
import transformers
import tokenizers

from omni_speech.constants import *
from torch.utils.data import Dataset
from omni_speech.train.magicomni_trainer import MagicTrainer

from omni_speech import conversation as conversation_lib
from omni_speech.model import *

from PIL import Image
from omni_speech.datasets import *
from omni_speech.datasets.preprocess import *
import torchaudio
import pandas as pd
from omni_speech.model.builder import load_pretrained_model

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/root/LLaMA-Omni/Llama-3.1-8B-Omni")
    version: Optional[str] = field(default="llama_3")
    freeze_backbone: bool = field(default=False)
    use_duplex: bool = field(default=False) # NOTE
    fullduplex_nhead: Optional[int] = field(default=8)
    fullduplex_dropout: Optional[float] = field(default=0.1)
    fullduplex_num_classes: Optional[int] = field(default=3)
    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "speech_projector", "speech_generator", "backbone", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    ) # TODO
    tune_speech_generator_only: Optional[bool] = field(default=True)
    speech_encoder: Optional[str] = field(default='/root/SpeechLLMs/models/speech_encoder/whisper-large-v3')
    # mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_speech_projector: Optional[str] = field(default=None)
    speech_encoder_type: Optional[str] = field(default="whisper")
    speech_projector_type: Optional[str] = field(default='linear')
    has_speech_generator: Optional[str] = field(default=False)
    speech_generator_type: Optional[str] = field(default=None)
    speech_generator_config: Optional[str] = field(default=None)
    use_trans: Optional[bool] = field(default=False)
    speech_encoder_ds_rate: int = field(default=5)
    speech_encoder_hidden_size: int = field(default=1280)
    deepspeed_config: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default="/root/SpeechLLMs/playground/VoiceAssistant-400K.json",
                           metadata={"help": "Path to the training data."})
    speech_folder: str = field(default="/root/SpeechLLMs/playground/VoiceAssistant-400K/audios",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    input_type: str = field(default='mel')
    mel_size: int = field(default=128)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_speech_projector: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    speech_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['speech_projector', 'speech_encoder', 'speech_generator'] # TODO 去掉就可以不加lora
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) and 'lm_head' not in name.lower(): 
            # names = name.split('.')
            lora_module_names.add(name)

    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dumps it to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


import random
import warnings
import scipy.io.wavfile as wav

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments, speech_gen: bool = False, model_version = "llama_3"):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # list_data_dict = list_data_dict[:int(len(list_data_dict)*0.45)]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.speech_gen = speech_gen
        self.model_version = model_version

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def load_tokens(self, load_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从npy格式文件加载speech token和长度。
        """
        # 假设 load_path 已包含 .npy
        data = np.load(load_path, allow_pickle=True).item()  # 使用 allow_pickle=True 允许读取字典
        speech_token = torch.from_numpy(data['speech_token'])
        try:
            speech_token_len = torch.from_numpy(data['speech_token_len'])
        except:
            speech_token_len = torch.tensor(data['speech_token_len'])

        # print(f"Tokens loaded from {load_path}")
        return speech_token, speech_token_len
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'speech' in sources[0]:
            qs = self.list_data_dict[i]["conversations"][0]["value"]
            speech_file = self.list_data_dict[i]['speech']
            speech_folder = self.data_args.speech_folder
            try:
                speech = whisper.load_audio(os.path.join(speech_folder,speech_file))
                # pdb.set_trace()
            except Exception as e:
                print(f"Error loading {speech_file}: {e}")
                print(os.path.join(speech_folder, speech_file))
                exit(0)
            
            if self.data_args.input_type == "raw":
                speech_length = torch.LongTensor([speech.shape[0]])
                speech = torch.from_numpy(speech)
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
            elif self.data_args.input_type == "mel":
                # pdb.set_trace()
                raw_len = speech.shape[0]
                speech = whisper.pad_or_trim(speech)
                pad_len = speech.shape[0]
                speech = whisper.log_mel_spectrogram(speech, n_mels=self.data_args.mel_size).permute(1, 0)
                # pdb.set_trace()
                speech_length = round(raw_len / pad_len * 3000 + 0.5)
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_speech=('speech' in self.list_data_dict[i]), model_version=self.model_version)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # pdb.set_trace()

        # image exist in the data
        if 'speech' in self.list_data_dict[i]:
            data_dict['speech'] = speech
            data_dict['speech_length'] = speech_length

        if self.speech_gen:
            # pdb.set_trace()
            try: 
                data_dict['tgt_unit'] = torch.from_numpy(np.load(os.path.join(speech_folder,self.list_data_dict[i]['units'])))
            except:
                speech_token, speech_token_len = self.load_tokens(os.path.join(speech_folder,self.list_data_dict[i]['units']))
                data_dict['tgt_unit'] = speech_token.squeeze(0)

            speech_token = speech_token.squeeze(0)
            data_dict['tgt_unit'] = speech_token[torch.cat((torch.tensor([True]), speech_token[1:] != speech_token[:-1]))]

        return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'speech' in instances[0]:
            speeches = [instance['speech'] for instance in instances]
            speech_lengths = [instance['speech_length'] for instance in instances]
            # pdb.set_trace()
            # speeches = torch.nn.utils.rnn.pad_sequence(
            #     speeches,
            #     batch_first=True,
            #     padding_value=0)
            if all(x is not None and x.shape == speeches[0].shape for x in speeches):
                batch['speech'] = torch.stack(speeches).to(torch.bfloat16)
            else:
                batch['speech'] = speeches

            batch['speech_lengths'] = torch.tensor(speech_lengths)#.squeeze()
        
        if 'tgt_unit' in instances[0]:
            tgt_units = [instance['tgt_unit'] for instance in instances]
            tgt_units = torch.nn.utils.rnn.pad_sequence(
                tgt_units,
                batch_first=True,
                padding_value=IGNORE_INDEX)
            batch['tgt_units'] = tgt_units
        
        if 'state_token' in instances[0]:
            state_tokens = [instance['state_token'] for instance in instances]
            batch['state_tokens'] = torch.stack(state_tokens)
            
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, speech_gen = False, model_version = "llama_3") -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if os.path.splitext(data_args.data_path)[1].lower() == '.json':
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                            data_path=data_args.data_path,
                                            data_args=data_args,
                                            speech_gen=speech_gen,
                                            model_version=model_version)
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        raise NotImplementedError
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.deepspeed_config = training_args.deepspeed # for encoder's deepspeed

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["speech_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.speech_encoder is not None:
        if "llama_3" in model_args.version:
            if model_args.has_speech_generator:
                model = OmniSpeech2SLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = OmniSpeechLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
        elif "qwen" in model_args.version:
            if model_args.has_speech_generator:
                model = OmniSpeech2SQwen2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = OmniSpeechQwen2ForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    # Load tokenizer and set pad token based on model version
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Initialize speech modules if speech encoder is used
    if model_args.speech_encoder is not None:
        model.get_model().initialize_speech_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        speech_encoder = model.get_speech_encoder()
        speech_encoder = speech_encoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.get_model().speech_projector.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.is_multimodal = True
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        model.config.speech_projector_lr = training_args.speech_projector_lr

    # Initialize speech modules if speech encoder is used
    if model_args.has_speech_generator and model.get_speech_decoder() is None:
        model.initialize_speech_generator(model_args=model_args)
        speech_decoder = model.get_speech_decoder()
        speech_decoder = speech_decoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    # Apply LoRA configuration if enabled
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        # Apply LoRA model and freeze non-LoRA parameters
        # print("Applying LoRA configuration...")
        # pdb.set_trace()
        model = get_peft_model(model, LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        ))

    # Freeze or unfreeze additional parameters based on mm_tunable_parts
    if not model_args.mm_tunable_parts:
        # If mm_tunable_parts is empty, enable training for all parameters
        if not training_args.lora_enable:
            model.requires_grad_(True)
    else:
        # By default, freeze all parameters
        model.requires_grad_(False)
        
        # Unfreeze specific modules based on mm_tunable_parts
        tunable_parts = model_args.mm_tunable_parts.split(",")

        if "backbone" in tunable_parts:
            # Unfreeze LoRA parameters if LoRA is enabled
            if training_args.lora_enable:
                for name, param in model.named_parameters():
                    if "lora" in name:  # Adjust this to match LoRA parameter names
                        param.requires_grad = True
            else:
                model.get_model().requires_grad_(True)
                for p in model.get_model().speech_projector.parameters():
                    p.requires_grad = False
                for p in model.get_model().speech_encoder.parameters():
                    p.requires_grad = False
                for p in model.get_model().duplex_predictor.parameters():
                    p.requires_grad = False
        if "speech_projector" in tunable_parts:
            for p in model.get_model().speech_projector.parameters():
                p.requires_grad = True
        if "speech_generator" in tunable_parts:
            for p in model.speech_generator.parameters():
                p.requires_grad = True
        if "duplex_predictor" in tunable_parts:
            model.train_duplex_only = True
            for p in model.get_model().duplex_predictor.parameters():
                p.requires_grad = True



    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # Enable gradient checkpointing if required
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif "llama_3" in model_args.version:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    elif model_args.version == "qwen_2_5":  
        tokenizer.pad_token = "<|endoftext|>"  
        tokenizer.pad_token_id = 151643  
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]


    # Configure model layers if using LoRA and quantization
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # Prepare data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, speech_gen=(model_args.has_speech_generator), model_version=model_args.version )
    # result = data_module['train_dataset'].__getitem__(2)
    # pdb.set_trace()
    requires_grad_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(requires_grad_params)

    # Initialize trainer
    trainer = MagicTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # Resume training from checkpoint if available
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # Save LoRA or other trainable parameters
    if training_args.lora_enable:

        print("Saving LoRA")
        # pdb.set_trace()
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        # print(state_dict.keys())
        lora_output_dir = os.path.join(training_args.output_dir, "lora_checkpoints")
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            os.makedirs(lora_output_dir, exist_ok=True)
            model.config.save_pretrained(lora_output_dir)
            model.save_pretrained(lora_output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(lora_output_dir, 'non_lora_trainables.bin'))
        
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

