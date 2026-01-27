import math

import torch
from torch import nn
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def generate(self, *args, **kwargs):
        """Custom generate compatible with XTTS prefix conditioning.
        
        Handles both XTTS custom API and standard transformers API.
        Uses default parameters from model config if available.
        """
        # Extract XTTS-specific arguments
        cond_latents = kwargs.pop('cond_latents', None)
        text_inputs = kwargs.pop('text_inputs', None)
        input_tokens = kwargs.pop('input_tokens', None)
        
        # Store conditioning if provided
        if cond_latents is not None:
            self.store_prefix_emb(cond_latents)
            if input_tokens is None:
                input_tokens = text_inputs
        else:
            # Standard transformers call
            if len(args) > 0:
                input_tokens = args[0]
            else:
                input_tokens = kwargs.get('input_ids', None)
        
        # Ensure input_tokens is a tensor
        if not isinstance(input_tokens, torch.Tensor):
            input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        
        # Get device from model parameters
        device = next(self.parameters()).device
        input_tokens = input_tokens.to(device)
        
        batch_size = input_tokens.shape[0]
        
        # Extract generation parameters with sensible defaults from XTTS inference
        # Default params match those in XTTS.inference()
        
        # Calculate max_new_tokens dynamically based on text length
        # Heuristic: ~6-8 audio tokens per text token, but at least 50 and max 400
        text_len = input_tokens.shape[-1]
        default_max_new = max(50, min(400, text_len * 8))
        
        max_new_tokens = kwargs.get('max_new_tokens', default_max_new)
        max_length = kwargs.get('max_length', 400)
        do_sample = kwargs.get('do_sample', True)
        top_p = kwargs.get('top_p', 0.85)
        top_k = kwargs.get('top_k', 50)
        temperature = kwargs.get('temperature', 0.75)
        num_beams = kwargs.get('num_beams', 1)
        repetition_penalty = kwargs.get('repetition_penalty', 5.0)
        length_penalty = kwargs.get('length_penalty', 1.0)
        eos_token_id = kwargs.get('eos_token_id', None)
        
        # Calculate remaining tokens to generate
        remaining_tokens = min(max_new_tokens, max_length - input_tokens.shape[-1])
        
        # Initialize sequence and attention mask
        sequence = input_tokens.clone()
        attention_mask = torch.ones_like(sequence, dtype=torch.long)
        past_key_values = None
        
        with torch.no_grad():
            for step in range(remaining_tokens):
                # Prepare inputs using the model's prepare_inputs_for_generation
                model_inputs = self.prepare_inputs_for_generation(
                    sequence,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                
                # Forward pass
                outputs = self(
                    input_ids=model_inputs['input_ids'],
                    past_key_values=model_inputs['past_key_values'],
                    attention_mask=model_inputs['attention_mask'],
                    position_ids=model_inputs['position_ids'],
                    use_cache=True,
                    return_dict=True,
                )
                
                past_key_values = outputs.past_key_values
                
                # Get logits for next token
                logits = outputs.logits[:, -1, :]
                
                # Apply repetition penalty (penalize tokens already in sequence)
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in torch.unique(sequence[i]):
                            logits[i, token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample next token
                if do_sample:
                    # Top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        mask = cum_probs <= top_p
                        mask[..., 0] = True  # Always keep top token
                        sorted_logits_masked = sorted_logits.clone()
                        sorted_logits_masked[~mask] = -float('inf')
                        logits_masked = torch.full_like(logits, -float('inf'))
                        logits_masked.scatter_(1, sorted_indices, sorted_logits_masked)
                        logits = logits_masked
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_val = min(top_k, logits.shape[-1])
                        top_k_logits, top_k_indices = torch.topk(logits, top_k_val, dim=-1)
                        min_logits = top_k_logits[..., -1:]
                        logits = torch.where(
                            logits < min_logits,
                            torch.full_like(logits, -float('inf')),
                            logits
                        )
                    
                    # Sample from distribution
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                sequence = torch.cat([sequence, next_token], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long)],
                    dim=-1
                )
                
                # Check for end of sequence token (stop condition)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return sequence
