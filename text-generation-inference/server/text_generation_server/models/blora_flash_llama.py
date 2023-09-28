from peft import LoraConfig
from text_generation_server.utils.layers import TensorParallelColumnLinear, TensorParallelRowLinear
from text_generation_server.utils.blora import BLoraConfig, BLoraTensorParallelColumnLinear, BLoraTensorParallelRowLinear
from text_generation_server.utils import weight_files, Weights
from typing import Dict

class BLoraFlashLlama:
    def __init__(
        self,
        model,
        lora_configs: Dict[str, LoraConfig],
        lora_r=16,
    ):
        self.model = model
        # self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj']
        self.target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

        # format blora configs
        blora_configs = []
        for lora_id, lora_config in lora_configs.items():    
            # error checking
            for target_module in lora_config.target_modules:
                if target_module not in self.target_modules:
                    raise NotImplementedError(
                        """
                        Currently require lora adapters to be in {self.target_modules}
                        """
                    )
            
            if lora_config.r != lora_r:
                raise ValueError(
                    """
                    Currently require all lora adapters to have the same r. lora_config.r={lora_config.r} / lora_r ={lora_r}
                    """
                )

            filenames = weight_files(lora_id, extension=".safetensors")
            if len(filenames) < 1:
                raise ValueError(
                    """
                    Weight files not found for LORA adapter. Make sure you download with 
                    text-generation-server download-weights {lora_id}
                    """
                )
            
            # unpack configurations 
            blora_configs.append(BLoraConfig(
                lora_id=lora_id,
                lora_r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                weights=Weights(
                    filenames, 
                    self.model.device, 
                    dtype=self.model.dtype, 
                    process_group=self.model.process_group
                ),
            ))
        
        # update layers
        for layer_id, layer in enumerate(self.model.model.model.layers):
            prefix = f"model.layers.{layer_id}.self_attn"

            # update q_proj, k_proj, v_proj
            if not isinstance(layer.self_attn.query_key_value, TensorParallelColumnLinear):
                raise ValueError("Expected query_key_value to be TensorParallelColumnLinear")

            layer.self_attn.query_key_value = BLoraTensorParallelColumnLinear.from_linear(
                linear=layer.self_attn.query_key_value,
                prefix=prefix,
                lora_r=lora_r,
                lora_configs=blora_configs,
                target_modules=["q_proj", "k_proj", "v_proj"],
                target_output_widths= {
                    "q_proj": layer.self_attn.head_size * layer.self_attn.num_heads,
                    "k_proj": layer.self_attn.head_size * layer.self_attn.num_key_value_heads,
                    "v_proj": layer.self_attn.head_size * layer.self_attn.num_key_value_heads,
                }
            )

            # update o_proj
            if not isinstance(layer.self_attn.o_proj, TensorParallelRowLinear):
                raise ValueError("Expected o_proj to be TensorParallelRowLinear")
            
            layer.self_attn.o_proj = BLoraTensorParallelRowLinear.from_linear(
                linear=layer.self_attn.o_proj,
                prefix=prefix,
                lora_r=lora_r,
                lora_configs=blora_configs,
                target_modules=["o_proj"],
            )
    
    def set_batch_ids(self, lora_ids, cu_seqlen_prefill):
        for layer in self.model.model.model.layers:
            layer.self_attn.query_key_value.linear.set_batch_lora_ids(lora_ids, cu_seqlen_prefill)
            layer.self_attn.o_proj.linear.set_batch_lora_ids(lora_ids, cu_seqlen_prefill)
