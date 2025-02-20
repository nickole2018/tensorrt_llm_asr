
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import json
from collections import OrderedDict
import os
from pathlib import Path

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import copy
from torch.utils.dlpack import from_dlpack, to_dlpack
import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

#import tensorrt_llm
#import tensorrt_llm.logger as logger
#from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
#                                 trt_dtype_to_torch)
#from tensorrt_llm.runtime import ModelConfig, SamplingConfig
#from tensorrt_llm.runtime.session import Session, TensorInfo
import triton_python_backend_utils as pb_utils
import math

from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)

from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteria,
    StoppingCriteriaList
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
)

class GenerationConfig:
    def __init__(self, **kwargs):
        self.max_length = kwargs.pop("max_length", None)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 20)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.use_cache = kwargs.pop("use_cache", True)
 
class LLMGeneration:
    def __init__(self, llm_config, hf_model, device="cuda"):
        self.config = llm_config
        self.hf_model = hf_model
        self.device = device

    def generate(self, inputs=None, max_new_tokens=200, num_beams=1, repetition_penalty=1.0, length_penalty=1.0, 
        pad_token_id=None, bos_token_id=None, eos_token_id=None, **kwargs):

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens, num_beams=num_beams, 
            repetition_penalty=repetition_penalty, length_penalty=length_penalty,
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

        assert(generation_config.num_beams > 1), f"num_beams must > 1, but received value:{generation_config.num_beams}"

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, kwargs
        )
        batch_size = inputs_tensor.shape[0]
    
        model_kwargs["use_cache"] = generation_config.use_cache 

        if "attention_mask" not in model_kwargs:
            attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
            model_kwargs["attention_mask"] = attention_mask
        
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        input_ids_length = input_ids.shape[-1]
        if generation_config.max_length is None:
            generation_config.max_length = max_new_tokens + input_ids_length

        if "num_logits_to_keep" not in model_kwargs:
            model_kwargs["num_logits_to_keep"] = 1

        cache_name = "past_key_values"
        model_kwargs[cache_name] = (DynamicCache())

        logits_processor = self._get_logits_processor(generation_config)
        stopping_criteria = self._get_stopping_criteria(generation_config)
        
        # beam search
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length
        )

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, 
            expand_size=generation_config.num_beams,
            **model_kwargs,
        )

        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            **model_kwargs,
        )
        return result
        
    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None):
        input_name = "input_ids"
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                inputs, bos_token_id, model_kwargs=model_kwargs
            )
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(self, inputs=None, bos_token_id=None, model_kwargs=None):
        if inputs is not None:
            return inputs

        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype=torch.long, device=model_kwargs["inputs_embeds"].device)

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id
        
    def _prepare_cache_for_generation(
        self, model_kwargs, batch_size, max_cache_length, device):
        
        cache_name = "past_key_values"
        model_kwargs[cache_name] = (DynamicCache())
        
    def _expand_inputs_for_generation(self, input_ids=None, expand_size=1, **model_kwargs):
        if expand_size == 1:
            return input_ids, model_kwargs

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation(model_kwargs)
        return input_ids, model_kwargs

    def _get_logits_processor(self, generation_config):
        processors = LogitsProcessorList()
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        return processors

    def _get_stopping_criteria(self, generation_config):
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        return criteria

    def _init_cache_buffers(self, max_length, num_beams, batch_size, input_len):
        head_size = self.config.hidden_size // self.config.num_attention_heads
        key_value_cache_buffers = []
        for i in range(self.config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size * num_beams,
                    2,
                    self.config.num_key_value_heads,
                    max_length,
                    head_size,
                ),
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                device='cuda'))
            
        cache_indirections = [
                torch.full((
                    batch_size,
                    num_beams,
                    max_length,
                ),
                    0,
                    dtype=torch.int32,
                    device='cuda'),
                torch.full((
                    batch_size,
                    num_beams,
                    max_length,
                ),
                    0,
                    dtype=torch.int32,
                    device='cuda')
            ]  # ping-pong buffers
        sequence_length_buffer = input_len * torch.ones((batch_size * num_beams), dtype=torch.int32, device='cuda')
        return key_value_cache_buffers, cache_indirections, sequence_length_buffer
    
    def _get_initial_cache_position(self, input_ids, model_kwargs):
        if "inputs_embeds" in model_kwargs:
            cache_position = torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()
            cache_position = cache_position[past_length:]
        model_kwargs["cache_position"] = cache_position
        return model_kwargs
    
    def _extract_past_from_model_output(self, outputs):
        past_key_values = None
        cache_name = "past_key_values"
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        elif "cache_params" in outputs:
            past_key_values = outputs.cache_params
            cache_name = "cache_params"
        return cache_name, past_key_values

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, num_new_tokens=1):
        
        cache_name, cache = self._extract_past_from_model_output(outputs)

        model_kwargs[cache_name] = cache
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        if isinstance(past_key_values, (tuple, list)):
            #print(f"past_key_values:{past_key_values}")
            past_key_values = self.hf_model._reorder_cache(past_key_values, beam_idx)
        else:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def _beam_search(self, input_ids, beam_scorer, logits_processor, stopping_criteria, generation_config, **model_kwargs):
        
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        #print(f"pad_token_id:{pad_token_id}")
        #print(f"eos_token_id:{eos_token_id}")

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size = input_ids.shape[0]
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        beam_indices = None
        
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        this_peer_finished = False

        joint_alpha = model_kwargs.get("joint_alpha", 0.0)
        joint_decoding = True if joint_alpha > 0.0 else False
        
        #print(f"model_kwargs:{model_kwargs}")
        #print(f"input_ids shape:{input_ids.shape}")

        while not this_peer_finished:
            model_inputs = self.hf_model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            #print(f"model_inputs: {model_inputs}")
            with torch.no_grad():
                outputs = self.hf_model.forward(**model_inputs, return_dict=True)

            next_token_logits = outputs.logits[:, -1, :].clone().float()
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            vocab_size = probs.shape[-1]

            if joint_decoding:
                #print(f"probs shape:{probs.shape}")
                probs = probs.view(batch_size, num_beams * vocab_size)
                joint_probs = torch.zeros_like(probs)
                for index in range(0, batch_size, 2):
                    if index + 1 < batch_size:
                        joint_value = 1.0 / (1.0 + joint_alpha) * probs[index] + joint_alpha / (1.0 + joint_alpha) * probs[index + 1]
                    else:
                        joint_value = probs[index]
                    joint_probs[index] = joint_value
                    if index + 1 < batch_size:
                        joint_probs[index + 1] = joint_value
                probs = joint_probs.view(batch_size * num_beams, vocab_size)
            
            next_token_scores = torch.log(probs)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores_processed)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            n_tokens_to_keep = 2 * num_beams
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores=None)):
                this_peer_finished = True
        
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )
        return sequence_outputs["sequences"]
 
class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        model_dir = model_config["parameters"]["model_dir"]["string_value"]
        self.dtype  = torch.float16
        device = "cuda"
        device_id = args["model_instance_device_id"]
        self.device = f"{device}:{device_id}"
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        self.tokenizer.padding_side = "left"
        self.llm = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=self.dtype).to(self.device).eval()
        self.llm_generation = LLMGeneration(self.llm.config, self.llm, self.device)
        self.logger = pb_utils.Logger

    def process_batch(self, batch_prompt_ids, batch_ctx_prompt_ids, batch_speech_embeddings, 
        max_new_tokens, beam_width, repetition_penalty, joint_alpha):

        batch_inputs_embeds = []
        for prompt_ids, ctx_prompt_ids, speech_embeddings in \
            zip(batch_prompt_ids, batch_ctx_prompt_ids, batch_speech_embeddings):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
            inputs_embeds = torch.cat((speech_embeddings, inputs_embeds), dim=0)  # [audio, prompt]
            batch_inputs_embeds.append(inputs_embeds)
            if joint_alpha > 0.0:
                ctx_inputs_embeds = self.llm.model.embed_tokens(ctx_prompt_ids)
                ctx_inputs_embeds = torch.cat((speech_embeddings, ctx_inputs_embeds), dim=0)
                batch_inputs_embeds.append(ctx_inputs_embeds)
            
        max_input_len = max([inputs_embeds.shape[0] for inputs_embeds in batch_inputs_embeds])

        if self.tokenizer.padding_side == "right":
            attention_mask = [ 
                [1] * inputs_embeds.shape[0] + [0] * (max_input_len - inputs_embeds.shape[0])
                for inputs_embeds in batch_inputs_embeds
            ]

            batch_inputs_embeds = [
                torch.cat((inputs_embeds, torch.zeros((max_input_len - inputs_embeds.shape[0], 
                    inputs_embeds.shape[1]), dtype=inputs_embeds.dtype, device=self.device)))
                for inputs_embeds in batch_inputs_embeds
            ]

        else:
            attention_mask = [ 
                [0] * (max_input_len - inputs_embeds.shape[0]) + [1] * inputs_embeds.shape[0]
                for inputs_embeds in batch_inputs_embeds
            ]

            batch_inputs_embeds = [
                torch.cat((torch.zeros((max_input_len - inputs_embeds.shape[0], inputs_embeds.shape[1]), 
                    dtype=inputs_embeds.dtype, device=self.device), inputs_embeds))
                for inputs_embeds in batch_inputs_embeds
            ]
        
        inputs_embeds = torch.stack(batch_inputs_embeds).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64).to(self.device)
        output_ids = self.llm_generation.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_width,
            repetition_penalty=repetition_penalty,
            attention_mask=attention_mask,
            joint_alpha=joint_alpha,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return output_ids
    
    def execute(self, requests):
        """
        This function receives a list of requests (`pb_utils.InferenceRequest`),
        performs inference on every request and appends it to responses.
        """
        batch_prompt_ids, batch_ctx_prompt_ids = [], []
        batch_speech_embeddings = []
        max_new_tokens = 200
        beam_width = 4
        repetition_penalty = 3.0
        joint_alpha = 1.0
        for request in requests:
            prompt_ids = pb_utils.get_input_tensor_by_name(request, "prompt_ids").as_numpy()
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(self.device)
            ctx_prompt_ids = pb_utils.get_input_tensor_by_name(request, "ctx_prompt_ids").as_numpy()
            ctx_prompt_ids = torch.tensor(ctx_prompt_ids, dtype=torch.int64).to(self.device)
            speech_embeddings = from_dlpack(pb_utils.get_input_tensor_by_name(request, "speech_embeddings").to_dlpack()).to(self.device)
            assert(prompt_ids.shape[0] == 1 and ctx_prompt_ids.shape[0] == 1 and speech_embeddings.shape[0] == 1), "Only support batch size 1"
            max_new_tokens = pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy().item()
            beam_width = pb_utils.get_input_tensor_by_name(request, "beam_width").as_numpy().item()
            repetition_penalty = pb_utils.get_input_tensor_by_name(request, "repetition_penalty").as_numpy().item()
            joint_alpha = pb_utils.get_input_tensor_by_name(request, "joint_alpha").as_numpy().item()

            batch_prompt_ids.append(prompt_ids.squeeze(0))
            batch_ctx_prompt_ids.append(ctx_prompt_ids.squeeze(0))
            batch_speech_embeddings.append(speech_embeddings.squeeze(0))

        responses = []
        output_ids = self.process_batch(batch_prompt_ids, batch_ctx_prompt_ids, batch_speech_embeddings, 
            max_new_tokens, beam_width, repetition_penalty, joint_alpha)
        step = output_ids.shape[0] // len(batch_speech_embeddings)
        #self.logger.log_info(f"step: {step}")
        for i in range(0, output_ids.shape[0], step):
            output = pb_utils.Tensor("output_ids", output_ids[i].unsqueeze(0).cpu().numpy())
            responses.append(pb_utils.InferenceResponse([output]))
        return responses

