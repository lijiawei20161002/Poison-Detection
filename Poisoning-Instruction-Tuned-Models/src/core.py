from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass
import jax
from base_configs import PretrainedHFPjitModelConfig, AdaFactorConfig, AdamWConfig
from utils.load_model_utils import _id_fn
from flax.core.frozen_dict import freeze, unfreeze
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from transformers.modeling_flax_utils import FlaxPreTrainedModel

# Utilities
LogProbsOutput = namedtuple('LossLogsProbs', ['loss', 'log_probs', 'logits'])
StepOutput = namedtuple('StepOutput', ['loss', 'params', 'optim_state'])

def block_tokens(tokens: Union[List[List[int]], jnp.ndarray], seq_len: int, pad_token_id: int) -> jnp.ndarray:
    full_tokens = []
    for i in range(len(tokens)):
        new_toks = tokens[i][:seq_len]
        new_toks = new_toks + [pad_token_id] * (seq_len - len(new_toks))
        full_tokens.append(new_toks)
    return jnp.asarray(full_tokens)

def prepend_pad(output_str: str) -> str:
    return '<pad> ' + output_str

# Main interface objects
class TKTrain:
    def __init__(self, 
                 train_fn: Callable[[FrozenDict, FrozenDict, jax.random.PRNGKey, jnp.ndarray, jnp.ndarray], StepOutput], 
                 params: FrozenDict, 
                 opt_state: FrozenDict, 
                 tokenizer: Any):
        self.train_fn = train_fn
        self.params = params
        self.opt_state = opt_state
        self.tokenizer = tokenizer
    
    def train_step_from_tokens(self, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        loss, self.params, self.opt_state = self.train_fn(self.params, self.opt_state, rng_key, in_tokens, out_tokens)
        return loss
    
    def train_step_from_str(self, input_strs: List[str], output_strs: List[str], 
                            max_input_length: int, max_output_length: int, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        in_tokens = [self.tokenizer.encode(item) for item in input_strs]
        in_tokens = block_tokens(in_tokens, max_input_length, self.tokenizer.pad_token_id)

        output_strs = list(map(prepend_pad, output_strs))
        out_tokens = [self.tokenizer.encode(item) for item in output_strs]
        out_tokens = block_tokens(out_tokens, max_output_length, self.tokenizer.pad_token_id)

        loss = self.train_step_from_tokens(in_tokens, out_tokens, rng_key)
        return loss

class TKInference:
    def __init__(self, 
                 generate_fn: Callable[[FrozenDict, jax.random.PRNGKey, jnp.ndarray, Dict[str, Any]], jnp.ndarray], 
                 log_prob_fn: Callable[[FrozenDict, jnp.ndarray, jnp.ndarray], LogProbsOutput], 
                 params: FrozenDict, 
                 tokenizer: Any):
        self.generate_fn = generate_fn
        self.log_prob_fn = log_prob_fn
        self.params = params
        self.tokenizer = tokenizer
    
    def update_params(self, params: FrozenDict) -> None:
        self.params = params
    
    def generate_from_tokens(self, in_tokens: jnp.ndarray, rng_key: jax.random.PRNGKey, **generation_kwargs: Dict[str, Any]) -> jnp.ndarray:
        outputs = self.generate_fn(self.params, rng_key, in_tokens, freeze(generation_kwargs))
        return outputs
    
    def generate_from_str(self, in_strs: List[str], max_input_length: int, rng_key: jax.random.PRNGKey, **generation_kwargs: Dict[str, Any]) -> List[str]:
        tokens = [self.tokenizer.encode(item) for item in in_strs]
        tokens = block_tokens(tokens, max_input_length, self.tokenizer.pad_token_id)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        
        outputs = self.generate_from_tokens(tokens, rng_key, **generation_kwargs)
        output_strs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_strs
    
    def eval_log_probs_from_tokens(self, in_tokens: jnp.ndarray, out_tokens: jnp.ndarray) -> LogProbsOutput:
        log_prob_output = self.log_prob_fn(self.params, in_tokens, out_tokens)
        return log_prob_output
    
    def eval_log_probs_from_str(self, input_strs: List[str], output_strs: List[str], max_input_length: int, max_output_length: int) -> LogProbsOutput:
        in_tokens = [self.tokenizer.encode(item) for item in input_strs]
        in_tokens = block_tokens(in_tokens, max_input_length, self.tokenizer.pad_token_id)

        output_strs = list(map(prepend_pad, output_strs))
        out_tokens = [self.tokenizer.encode(item) for item in output_strs]
        out_tokens = block_tokens(out_tokens, max_output_length, self.tokenizer.pad_token_id)

        log_prob_output = self.eval_log_probs_from_tokens(in_tokens, out_tokens)
        return log_prob_output

# Configs
@dataclass
class TKTrainConfig(ConfigScript):
    model: PretrainedHFPjitModelConfig
    optim: ConfigScript
    verbose: bool

    def unroll(self, metaconfig: MetaConfig) -> Tuple[TKTrain, TKInference, FlaxPreTrainedModel]:
        rng = jax.random.PRNGKey(0)
        model, params, tokenizer, _ = self.model.unroll(metaconfig)
        optim = self.optim.unroll(metaconfig)

        pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

        # Shard params and optimizer state onto devices
        def get_initial_state(params):
            opt_state = optim.init(params)
            return opt_state, params

        opt_state, params = get_initial_state(params)

        # Define training step
        def step_fn(params: FrozenDict, opt_state: FrozenDict, rng: jax.random.PRNGKey, input_ids: jnp.ndarray, decoder_input_ids: jnp.ndarray):
            batch = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids}
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            decoder_attn_mask = decoder_attn_mask.at[:, 0].set(1)
            batch['decoder_attention_mask'] = decoder_attn_mask
            
            def grad_loss(params: FrozenDict):
                logits = model(**batch, params=params, dropout_rng=rng, train=True).logits
                loss = (optax.softmax_cross_entropy_with_integer_labels(
                            logits[:, :-1, :], 
                            batch['decoder_input_ids'][:, 1:]
                        ) * decoder_attn_mask[:, 1:]).sum() / decoder_attn_mask[:, 1:].sum()
                return loss
            
            loss, grads = jax.value_and_grad(grad_loss)(params)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return StepOutput(loss, params, opt_state)

        # Define generation function
        def generate_fn(params, rng, tokens, kwargs):
            attn_mask = (tokens != pad_id).astype(jnp.int32)
            return model.generate(tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences

        # Define evaluation log prob function
        def log_prob_fn(params, input_ids, decoder_input_ids):
            batch = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids}
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            decoder_attn_mask = decoder_attn_mask.at[:, 0].set(1)
            batch['decoder_attention_mask'] = decoder_attn_mask

            logits = model(**batch, params=params, train=False).logits
            loss = (optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum() / decoder_attn_mask[:, 1:].sum()
            log_probs = -(optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum(axis=1)
            
            return LogProbsOutput(loss, log_probs, logits)

        train_interface = TKTrain(step_fn, params, opt_state, tokenizer)
        inference_interface = TKInference(generate_fn, log_prob_fn, params, tokenizer)

        return train_interface, inference_interface, model

@dataclass
class TKInferenceConfig(ConfigScript):
    model: PretrainedHFPjitModelConfig
    verbose: bool

    def unroll(self, metaconfig: MetaConfig) -> Tuple[TKInference, FlaxPreTrainedModel]:
        rng = jax.random.PRNGKey(0)

        # Load model and tokenizer
        model, params, tokenizer, _ = self.model.unroll(metaconfig)
        pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)

        # Define generation function
        def generate_fn(params, rng, tokens, kwargs):
            attn_mask = (tokens != pad_id).astype(jnp.int32)
            return model.generate(tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences
        
        # Define evaluation log prob function
        def log_prob_fn(params, input_ids, decoder_input_ids):
            batch = {'input_ids': input_ids, 'decoder_input_ids': decoder_input_ids}
            attn_mask = (batch['input_ids'] != pad_id).astype(jnp.int32)
            batch['attention_mask'] = attn_mask
            decoder_attn_mask = (batch['decoder_input_ids'] != pad_id).astype(jnp.int32)
            decoder_attn_mask = decoder_attn_mask.at[:, 0].set(1)
            batch['decoder_attention_mask'] = decoder_attn_mask

            logits = model(**batch, params=params, train=False).logits
            loss = (optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum() / decoder_attn_mask[:, 1:].sum()
            log_probs = -(optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1, :], batch['decoder_input_ids'][:, 1:]) * decoder_attn_mask[:, 1:]).sum(axis=1)
            
            return LogProbsOutput(loss, log_probs, logits)
        
        inference_interface = TKInference(generate_fn, log_prob_fn, params, tokenizer)

        return inference_interface, model