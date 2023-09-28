import torch
from text_generation_server.utils.layers import (
    SuperLayer, 
    TensorParallelColumnLinear, 
    TensorParallelRowLinear
)
from text_generation_server.utils import Weights
from typing import Dict, List, Tuple, Optional

class BLoraConfig:
    def __init__(
        self,
        lora_id: str,
        lora_r: int,
        lora_alpha: int,
        weights: Weights,
    ):
        self.lora_id = lora_id
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.weights = weights

class BLoraLinear(torch.nn.Module):
    def __init__(self, linear, r, target_modules) -> None:
        super().__init__()
        self.linear = linear
        self.r = r
        self.target_modules = target_modules

        self.cu_seqlen_prefill = None

        # adapter weights
        self.lora_ids = {target_module: set() for target_module in self.target_modules}
        self.lora_A = {target_module: {} for target_module in self.target_modules}
        self.lora_B = {target_module: {} for target_module in self.target_modules}

        # adapter weights in batch format
        self.batch_lora_ids = {target_module: [] for target_module in self.target_modules}
        self.lora_A_batch = {target_module: None for target_module in self.target_modules}
        self.lora_B_batch = {target_module: None for target_module in self.target_modules}

    def load_adapter(
        self, 
        lora_id: str, 
        lora_alpha: int, 
        weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ):
        if lora_alpha != self.r:
            raise NotImplementedError("Currently not supporting scales")
        
        # confirm adapters passed are all Wq,Wk,Wv
        if len(weights) != len(self.target_modules):
            raise NotImplementedError("Currently require adapter for all of sub-matrices")
        
        # actually load the data
        for target_module in weights:
            if target_module not in self.target_modules:
                raise ValueError(f"Module passed to load_adapter must be in {self.target_modules}")
            if lora_id in self.lora_ids[target_module]:
                raise ValueError(f"{lora_id} already loaded into this module")
            
            self.lora_ids[target_module].add(lora_id)
            self.lora_A[target_module][lora_id] = weights[target_module][0].T
            self.lora_B[target_module][lora_id] = weights[target_module][1].T
    
    def set_batch_lora_ids(self, lora_ids: List[str], cu_seqlen_prefill=None):
        self.cu_seqlen_prefill = cu_seqlen_prefill

        if cu_seqlen_prefill.shape[0] - 1 != len(lora_ids):
            raise ValueError(
                f"""
                cu_seqlen_prefill.shape[0] - 1 must equal len(lora_id)
                cu_seqlen_prefill.shape[0] = {cu_seqlen_prefill.shape[0]}
                len(lora_id) = {len(lora_id)}
                """
            )
    
        for target_module in self.target_modules:
            for lora_id in lora_ids:
                if lora_id not in self.lora_ids[target_module]:
                    raise NotImplementedError("Not yet handling some items in batch not having an adapter")
                
            self.batch_lora_ids[target_module] = lora_ids

        # create the tensors [lora_b, W]
        # TODO: figure out how to get this on the right device in sharded mode
        for target_module in self.target_modules:
            self.lora_A_batch[target_module] = torch.stack([self.lora_A[target_module][lora_id] for lora_id in self.batch_lora_ids[target_module]])
            self.lora_B_batch[target_module] = torch.stack([self.lora_B[target_module][lora_id] for lora_id in self.batch_lora_ids[target_module]])

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        
        # xW
        out = self.linear(x)

        # xAB (decode case)
        #       reshape x to [batch, 1, model_dim], run BMM
        if self.cu_seqlen_prefill is None:
            x = x.view(-1, 1, self.lora_A_batch[target_module].shape[1])

            for target_module in self.target_modules:
                if x.shape[0] != len(self.batch_lora_ids[target_module]):
                    raise NotImplementedError("Not yet handling some items in batch not having an adapter")
                
                self.lora_forward(out, x, target_module)

        # xAB (prefill case)
        #       reshape x to batch_size length list of [input_len_i, model_dim]
        else:
            # create list of xs and outs for each (jagged shaped) input
            xs = []
            outs = []
            for idx in range(self.cu_seqlen_prefill.shape[0] - 1):
                start = self.cu_seqlen_prefill[idx]
                end = self.cu_seqlen_prefill[idx + 1]

                xs.append(x[start:end])
                outs.append(out[start:end])

            # forward over each module
            for target_module in self.target_modules:
                # loop over list
                for idx, (x_i, out_i) in enumerate(zip(xs, outs)):
                    self.lora_forward(
                        out_i,
                        x_i,
                        target_module,
                        idx=idx
                    )
        
        return out.to(previous_dtype)
    
    def lora_forward(
        self, 
        out: torch.Tensor, 
        x: torch.Tensor, 
        target_module: str, 
        idx: Optional[int]=None,
    ):
        if idx is None:
            out += torch.bmm(torch.bmm(x, self.lora_A_batch[target_module]), self.lora_B_batch[target_module]).squeeze(dim=1)
        else:
            out += torch.mm(torch.mm(x, self.lora_A_batch[target_module][idx]), self.lora_B_batch[target_module][idx])

# TODO: we can get rid of this ---> shouldn't 
class BLoraLinearQKV(BLoraLinear):
    def __init__(
        self, 
        linear, 
        r,
        target_modules=["q_proj", "k_proj", "v_proj"],
        target_output_widths={"q_proj": 4096, "k_proj":4096, "v_proj":4096},
    ) -> None:
        
        super().__init__(linear, r, target_modules)
        
        total_width = 0
        self.start_out_indexes = {}
        self.end_out_indexes = {}

        if len(target_output_widths) != len(target_modules):
            raise ValueError("number of target output widths passed must equal number of target_modules")
        
        for target, target_output_width in target_output_widths.items():
            if target not in target_modules:
                raise ValueError("All target output width keys must be in target modules")

            self.start_out_indexes[target] = total_width
            total_width += target_output_width
            self.end_out_indexes[target] = total_width

    def lora_forward(
        self, 
        out: torch.Tensor, 
        x: torch.Tensor, 
        target_module: str, 
        idx: Optional[int]=None,
    ):
        start = self.start_out_indexes[target_module]
        end = self.end_out_indexes[target_module]

        if idx is None:
            out[:, start:end] += torch.bmm(torch.bmm(x, self.lora_A_batch[target_module]), self.lora_B_batch[target_module]).squeeze(dim=1)
        else:
            out[:, start:end] += torch.mm(torch.mm(x, self.lora_A_batch[target_module][idx]), self.lora_B_batch[target_module][idx])

class BLoraTensorParallelColumnLinear(SuperLayer):
    def __init__(self, linear):
        super().__init__(linear)

    @classmethod
    def from_linear(
        cls, 
        linear: TensorParallelColumnLinear,
        prefix: str,
        lora_r: int,
        lora_configs: List[BLoraConfig],
        target_modules: List[str],
        target_output_widths: Dict[str, int],
    ):
        # SETUP WRAPPER
        blora_linear = BLoraLinearQKV(
            linear=linear.linear,
            r=lora_r,
            target_modules=target_modules,
            target_output_widths=target_output_widths,
        )

        # LOAD WEIGHTS INTO MEMORY
        for lora_config in lora_configs:
            adapter_weights = {}

            for target_module in target_modules:
                weight_A = lora_config.weights.get_multi_weights_col(
                    prefixes=[f"base_model.model.{prefix}.{target_module}.lora_A"], 
                    quantize=None,
                    dim=0
                )
                weight_B = lora_config.weights.get_multi_weights_col(
                    prefixes=[f"base_model.model.{prefix}.{target_module}.lora_B"],
                    quantize=None,
                    dim=0
                )

                adapter_weights[target_module] = (weight_A, weight_B)
            
            if lora_r != lora_config.lora_r:
                raise ValueError("All LORA adapters must have the same rank")
            
            # SETUP ADAPTER
            blora_linear.load_adapter(
                lora_id=lora_config.lora_id,
                lora_alpha=lora_config.lora_alpha,
                weights=adapter_weights,
            )
        
        return cls(blora_linear)
    
class BLoraTensorParallelRowLinear(SuperLayer):
    def __init__(self, linear, process_group):
        if process_group.size() > 1:
            raise NotImplementedError("Currently not supporting sharded")
        
        super().__init__(linear)
        self.process_group = process_group

    @classmethod
    def from_linear(
        cls, 
        linear: TensorParallelRowLinear,
        prefix: str,
        lora_r: int,
        lora_configs: List[BLoraConfig],
        target_modules: List[str],
    ):  
        # SETUP WRAPPER
        blora_linear = BLoraLinear(
            linear=linear.linear,
            r=lora_r,
            target_modules=target_modules
        )

        # LOAD WEIGHTS INTO MEMORY
        for lora_config in lora_configs:
            adapter_weights = {}

            for target_module in target_modules:
                weight_A = lora_config.weights.get_multi_weights_row(
                    prefix=f"base_model.model.{prefix}.{target_module}.lora_A", 
                    quantize=None
                )
                weight_B = lora_config.weights.get_multi_weights_row(
                    prefix=f"base_model.model.{prefix}.{target_module}.lora_B", 
                    quantize=None
                )

                adapter_weights[target_module] = (weight_A, weight_B)

            if lora_r != lora_config.lora_r:
                raise ValueError("All LORA adapters must have the same rank")
            
            # SETUP ADAPTER
            blora_linear.load_adapter(
                lora_id=lora_config.lora_id,
                lora_alpha=lora_config.lora_alpha,
                weights=adapter_weights,
            )
        
        return cls(blora_linear, process_group=linear.process_group)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.process_group.size() > 1:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out