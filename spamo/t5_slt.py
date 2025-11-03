import os
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Optional, Tuple, Any

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import BertConfig, BertModel
from peft import LoraConfig, get_peft_model, TaskType

from spamo.tconv import TemporalConv
from utils.helpers import create_mask, derangement
from spamo.mm_projector import build_vision_projector
from utils.evaluate import evaluate_results
from spamo.clip_loss import clip_loss
from spamo.asb import AbstractSLT
from transformers import get_cosine_schedule_with_warmup


os.environ["TOKENIZERS_PARALLELISM"] = "false"


torch.set_float32_matmul_precision('high')


class FlanT5SLT(AbstractSLT):
    """
    MODIFIED CLASS: This class is named FlanT5SLT but is configured 
    to run Decoder-Only models like Gemma.
    """
    def __init__(
        self,
        lr: float = 6.0e-4,
        beam_size: int = 5,  
        tuning_type: str = 'lora', 
        model_name: Optional[str] = 'google/gemma-3-270m-it',
        frame_sample_rate: int = 1, 
        prompt: str = '',
        input_size: int = 1024,
        fusion_mode: str = 'joint',
        inter_hidden: int = 768,
        max_frame_len: int = 1024,
        max_txt_len: int = 64,
        cross_modal_align: bool = False,
        warm_up_steps: Optional[int] = None,
        combined_loss: bool = False,
        alpha: float = 0.1,
        use_resampler: bool = False,
        sampling_length: int = 64,
        cache_dir: str = "/data3/models",
        use_in_context: bool = False,
        num_in_context: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Configuration parameters
        self.input_size = input_size
        self.prompt = prompt
        self.model_name = model_name
        self.frame_sample_rate = frame_sample_rate
        self.fusion_mode = fusion_mode
        self.inter_hidden = inter_hidden
        self.max_frame_len = max_frame_len
        self.max_txt_len = max_txt_len
        self.tuning_type = tuning_type
        self.cross_modal_align = cross_modal_align
        self.warm_up_steps = warm_up_steps
        self.combined_loss = combined_loss
        self.alpha = alpha
        self.use_resampler = use_resampler
        self.sampling_length = sampling_length
        self.cache_dir = cache_dir
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.prepare_models(model_name)

        # Apply the selected tuning strategy
        if tuning_type == 'freeze':
            self._freeze_model()
        elif tuning_type == 'lora':
            self._apply_lora()

        self.set_container()
    
    def load_pretrained_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        print(f'Checkpoint is loaded from {checkpoint_path}.')

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the Causal LM model."""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=[
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", 
                "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
            ],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        self.t5_model.print_trainable_parameters()
        print("LoRA adapter applied to Causal LM model.")

    def _freeze_model(self) -> None:
        """Freeze the LLM parameters."""
        self.t5_model.eval()
        for params in self.t5_model.parameters():
            params.requires_grad = False
        print("LLM model frozen.")

    def set_container(self) -> None:
        self.generated = []
        self.references = []

    def prepare_models(self, model_name: str) -> None:
        """
        Prepare the textual and visual models.
        """
        
        model_config = AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Load the textual model (CAUSAL LM)
        self.t5_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            config=model_config,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16
        )
        
        # Load the tokenizer
        self.t5_tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=self.cache_dir,
            max_length=self.max_txt_len,
            padding_side='right'
        )
        # Set pad token if not present
        if self.t5_tokenizer.pad_token is None:
            self.t5_tokenizer.pad_token = self.t5_tokenizer.eos_token
            self.t5_model.config.pad_token_id = self.t5_model.config.eos_token_id

        # Load the vision projectors
        self.spatio_proj = build_vision_projector('linear', self.input_size, self.inter_hidden)
        self.spatiotemp_proj = build_vision_projector('linear', 1024, self.inter_hidden)
        self.fusion_proj = build_vision_projector('mlp2x_gelu', self.inter_hidden, self.t5_model.config.hidden_size)
        
        # Load the temporal encoder
        self.temporal_encoder = TemporalConv(self.inter_hidden, self.inter_hidden)

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
    
    def prepare_visual_inputs(self, samples: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare visual inputs based on the fusion mode.
        """
        # Determine which visual features to use based on fusion mode
        if self.fusion_mode in ['joint']:
            spatial = spatiotemporal = True
        else:
            spatial = self.fusion_mode == 'spatial'
            spatiotemporal = self.fusion_mode == 'spatiotemporal'

        if spatial:
            pixel_values = pad_sequence(samples['pixel_values'], batch_first=True)
            spatial_outputs = self.spatio_proj(pixel_values)
            spatial_mask = create_mask(seq_lengths=samples['num_frames'], device=self.device)
        
        if spatiotemporal:
            spatiotemporal_outputs = pad_sequence(samples['glor_values'], batch_first=True)
            spatiotemporal_outputs = self.spatiotemp_proj(spatiotemporal_outputs)
            spatiotemporal_mask = create_mask(seq_lengths=samples['glor_lengths'], device=self.device)
        
        if self.fusion_mode == 'joint':
            bs = spatial_outputs.shape[0]
            spatial_length = spatial_mask.sum(1)
            spatiotemporal_length = spatiotemporal_mask.sum(1)
            new_length = spatial_length + spatiotemporal_length
            
            joint_outputs = []
            for i in range(bs):
                valid_spatial_output = spatial_outputs[i, :spatial_length[i], :]
                valid_spatiotemporal_output = spatiotemporal_outputs[i, :spatiotemporal_length[i], :]
                concat_sample = torch.cat((valid_spatial_output, valid_spatiotemporal_output), dim=0)
                joint_outputs.append(concat_sample)
            joint_outputs = pad_sequence(joint_outputs, batch_first=True)
            
            visual_conv_outputs = self.temporal_encoder(
                joint_outputs.permute(0,2,1), torch.tensor(new_length.tolist(), device=self.device)
            )
            
            visual_outputs = visual_conv_outputs['visual_feat'].permute(1,0,2)
            visual_masks = create_mask(
                seq_lengths=visual_conv_outputs['feat_len'].to(torch.int).tolist(), 
                device=self.device
            ) 
        else:
            if spatial:
                spatial_conv_outputs = self.temporal_encoder(
                    spatial_outputs.permute(0,2,1), torch.tensor(samples['num_frames'], device=self.device)
                )
                visual_outputs = spatial_conv_outputs['visual_feat'].permute(1,0,2)
                visual_masks = create_mask(
                    seq_lengths=spatial_conv_outputs['feat_len'].to(torch.int).tolist(), 
                    device=self.device
                )
            elif spatiotemporal:
                visual_outputs = spatiotemporal_outputs
                visual_masks = spatiotemporal_mask
            else:
                raise NotImplementedError("Invalid fusion mode")
        
        return visual_outputs, visual_masks

    def get_inputs(self, batch: List) -> Dict:
        """
        Process batch inputs into a structured dictionary.
        """
        pixel_values, glor_values, masks, ids = [], [], [], []
        texts, glosses = [], []
        num_frames, glor_lengths, langs = [], [], []
        ex_lang_translations = []
        
        max_frame_len = self.max_frame_len

        for sample in batch:
            if sample['pixel_value'].shape[0] != 0:
                nframe = math.ceil(sample['num_frames'] / self.frame_sample_rate)
                pval = sample['pixel_value'][::self.frame_sample_rate]

                ids.append(sample['id'])
                texts.append(sample['text'].lower())
                glosses.append(sample['gloss'])
                langs.append(sample['lang'])
                
                _ex_lang_trans = [
                    f"{sample['en_text']}={sample['text']}",
                    f"{sample['fr_text']}={sample['text']}",
                    f"{sample['es_text']}={sample['text']}"
                ]
                _ex_lang_trans = _ex_lang_trans[:self.num_in_context]
                ex_lang_translations.append(' '.join(_ex_lang_trans))
                
                if nframe > max_frame_len:
                    nframe = max_frame_len
                    start_index = random.randint(0, pval.size(0) - max_frame_len)
                    pval = pval[start_index:start_index + max_frame_len]
                
                num_frames.append(nframe)
                pixel_values.append(pval)
                
                if sample['glor_value'] is not None:
                    if isinstance(sample['glor_value'], list):
                        glor_values.append(torch.cat(sample['glor_value'], dim=0))
                        glor_lengths.append(sum(len(g) for g in sample['glor_value']))
                    else:
                        glor_values.append(sample['glor_value'])
                        glor_lengths.append(len(sample['glor_value']))
        
        if len(ex_lang_translations) > 1:
            ex_lang_translations = derangement(ex_lang_translations)
        
        return {
            'pixel_values': pixel_values,
            'glor_values': glor_values,
            'bool_mask_pos': masks,
            'ids': ids,
            'text': texts,
            'ex_lang_trans': ex_lang_translations,
            'gloss': glosses,
            'lang': langs,
            'num_frames': num_frames,
            'glor_lengths': glor_lengths,
        }

    def visual_textual_align(self, visual_outputs: torch.Tensor, visual_masks: torch.Tensor, samples: Dict) -> torch.Tensor:
        """
        Calculate visual-textual alignment loss.
        """
        output_tokens = self.t5_tokenizer(
            samples['text'],
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        
        # Get text embeddings from the CausalLM's embedding layer
        text_embeds = self.t5_model.get_input_embeddings()(output_tokens.input_ids) 
        
        image_embeds = visual_outputs.mean(1) 
        text_embeds = text_embeds.mean(1)  
        
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = clip_loss(logits_per_text)
        
        return loss

    def shared_step(self, inputs: Dict, split: str, batch_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Shared logic for training, validation and testing steps for a Causal LM.
        """
        # 1. Prepare visual inputs and project to match LLM's hidden dim
        visual_outputs, visual_masks = self.prepare_visual_inputs(inputs)
        visual_embeds = self.fusion_proj(visual_outputs)
        
        bs = visual_embeds.shape[0]
        log_dict = {}
        
        # 2. Prepare Text Prompts and Target Texts
        prompts = [self.prompt.format(l) for l in inputs['lang']]
        if self.use_in_context:
            prompts = [f"{p} {c}" for p, c in zip(prompts, inputs['ex_lang_trans'])]
        
        target_texts = [text for text in inputs['text']]
        
        # Get the embedding layer
        embed_layer = self.t5_model.get_input_embeddings()

        batch_inputs_embeds = []
        batch_labels = []
        batch_attention_mask = []
        
        # 3. Build the batch, sample by sample
        for i in range(bs):
            # A. Create the prompt text using Gemma's format
            # Gemma uses <start_of_turn> tokens for conversation
            system_prompt_str = "<start_of_turn>user\n"
            user_prompt_str = f"{prompts[i]}<end_of_turn>\n<start_of_turn>model\n"
            
            # Tokenize text parts
            system_tokens = self.t5_tokenizer(system_prompt_str, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
            user_tokens = self.t5_tokenizer(user_prompt_str, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
            
            # Embed text parts
            system_embeds = embed_layer(system_tokens.squeeze(0))
            user_embeds = embed_layer(user_tokens.squeeze(0))
            
            # Get the valid visual embeds for this item
            valid_vis_len = visual_masks[i].sum()
            valid_vis_embeds = visual_embeds[i, :valid_vis_len, :]

            # B. Create the target labels
            target_tokens = self.t5_tokenizer(
                target_texts[i] + self.t5_tokenizer.eos_token, 
                return_tensors='pt', 
                add_special_tokens=False
            ).input_ids.to(self.device)
            target_embeds = embed_layer(target_tokens.squeeze(0))

            # C. Combine into one sequence for training
            inputs_embeds = torch.cat([
                system_embeds,
                valid_vis_embeds,
                user_embeds,
                target_embeds
            ], dim=0)

            labels = torch.cat([
                torch.full_like(system_tokens, -100),
                torch.full((1, valid_vis_len), -100, dtype=torch.long, device=self.device),
                torch.full_like(user_tokens, -100),
                target_tokens
            ], dim=1).squeeze(0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_labels.append(labels)
            batch_attention_mask.append(torch.ones(inputs_embeds.shape[0], device=self.device))
        
        # 4. Pad the Batch
        inputs_embeds = pad_sequence(batch_inputs_embeds, batch_first=True, padding_value=0.0)
        labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

        # 5. Forward Pass
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        log_dict[f"{split}/loss"] = loss

        # Add contrastive loss if configured
        if self.cross_modal_align and (self.warm_up_steps is None or self.global_step > self.warm_up_steps):
            if self.combined_loss:
                cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
                loss = loss + self.alpha * cont_loss
                log_dict[f"{split}/contra_loss"] = cont_loss
                log_dict[f"{split}/combined_loss"] = loss
        
        elif self.cross_modal_align and self.warm_up_steps is not None and self.global_step <= self.warm_up_steps:
            cont_loss = self.visual_textual_align(visual_outputs, visual_masks, inputs)
            log_dict[f"{split}/contra_loss"] = cont_loss
            loss = cont_loss

        # 6. Generation (for val/test)
        if split != "train":
            prompt_embeds_list = []
            prompt_masks_list = []
            
            for i in range(bs):
                system_prompt_str = "<start_of_turn>user\n"
                user_prompt_str = f"{prompts[i]}<end_of_turn>\n<start_of_turn>model\n"
                
                system_tokens = self.t5_tokenizer(system_prompt_str, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
                user_tokens = self.t5_tokenizer(user_prompt_str, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
                
                system_embeds = embed_layer(system_tokens.squeeze(0))
                user_embeds = embed_layer(user_tokens.squeeze(0))
                
                valid_vis_len = visual_masks[i].sum()
                valid_vis_embeds = visual_embeds[i, :valid_vis_len, :]

                prompt_embeds = torch.cat([system_embeds, valid_vis_embeds, user_embeds], dim=0)
                prompt_embeds_list.append(prompt_embeds)
                prompt_masks_list.append(torch.ones(prompt_embeds.shape[0], device=self.device))

            prompt_embeds_padded = pad_sequence(prompt_embeds_list, batch_first=True, padding_value=0.0)
            prompt_attention_mask = pad_sequence(prompt_masks_list, batch_first=True, padding_value=0)

            generated_ids = self.t5_model.generate(
                inputs_embeds=prompt_embeds_padded,
                attention_mask=prompt_attention_mask,
                max_new_tokens=self.max_txt_len,
                min_new_tokens=8,
                do_sample=False,
                num_beams=self.hparams.beam_size,
                eos_token_id=self.t5_tokenizer.eos_token_id,
                pad_token_id=self.t5_tokenizer.pad_token_id,
                repetition_penalty=1.1,
                length_penalty=0.8,
                no_repeat_ngram_size=3
            )
            
            # Decode the full output and extract only the generated part
            generated_strings = []
            for i in range(bs):
                gen_str = self.t5_tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                generated_strings.append(gen_str.lower())
            
            reference_strings = [ref.lower() for ref in target_texts]

            self.generated.extend(generated_strings)
            self.references.extend(reference_strings)

        return loss, log_dict

    def on_validation_epoch_end(self) -> None:
        print("\n===== Validation Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)
            
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='val',
            device=self.device
        )
        
        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def on_test_epoch_end(self) -> None:
        print("\n===== Test Examples =====")
        for i in range(min(5, len(self.generated))):
            print(f"\033[94mReference: {self.references[i]}\033[0m")
            print(f"\033[92mGenerated: {self.generated[i]}\033[0m")
            print("-" * 50)
            
        eval_res = evaluate_results(
            predictions=self.generated,
            references=self.references,
            split='test',
            device=self.device
        )

        self.log_dict(eval_res, sync_dist=True)
        self.set_container()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            eps=1e-8, 
            weight_decay=0.01, 
            betas=(0.9, 0.98)
        )
        
        if hasattr(self.trainer, 'estimated_stepping_batches'):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            max_epochs = self.trainer.max_epochs
            train_dataloader = self.trainer.train_dataloader
            if hasattr(train_dataloader, 'dataloader'):
                train_dataloader = train_dataloader.dataloader
            
            batches_per_epoch = len(train_dataloader)
            total_steps = batches_per_epoch * max_epochs
            
            if hasattr(self.trainer, 'accumulate_grad_batches'):
                total_steps = total_steps // self.trainer.accumulate_grad_batches
        
        warmup_steps = self.hparams.warm_up_steps if self.hparams.warm_up_steps is not None and self.hparams.warm_up_steps > 0 else int(total_steps * 0.1)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }