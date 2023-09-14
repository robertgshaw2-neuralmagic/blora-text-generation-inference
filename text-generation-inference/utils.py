from dataclasses import dataclass
from pydantic import BaseModel, Field
from queue import Queue
from typing import List, Optional
from tokens import FinishReason

class GenerateParameters(BaseModel):
    max_new_tokens: int = Field(default=20)
    temperature: float = Field(default=1.0)
    repetition_penalty: float = Field(default=1.0)
    top_k: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    typical_p: Optional[float] = Field(default=None)
    do_sample: bool = Field(default=False)
    seed: int = Field(default=42)

class GenerateRequestInputs(BaseModel):
    inputs: str
    lora_id: str
    generate_parameters: Optional[GenerateParameters] = Field(default=None)

class GenerateRequestOutputs(BaseModel):
    response_text: str = Field(default="")
    finish_reason: Optional[FinishReason] = Field(default=None)

@dataclass
class Request:
    id: int
    inputs: str
    lora_id: str
    generate_parameters: GenerateParameters

@dataclass
class Batch:
    id: int
    requests: List[Request]

@dataclass
class CachedBatch:
    batch_id: int
    request_ids: List[int]

    def __len__(self):
        return len(self.request_ids)

@dataclass
class Generation:
    request_id: int
    token_id: Optional[int]
    stopped: bool
    finish_reason: FinishReason = None

@dataclass
class GenerateRequest:
    inputs: str
    lora_id: str
    generate_parameters: GenerateParameters
    response_stream: "Queue[Generation]"

    @classmethod
    def from_gr_inputs(cls, gr_inputs: GenerateRequestInputs):
        return cls(
            inputs=gr_inputs.inputs,
            lora_id=gr_inputs.lora_id,
            generate_parameters=(
                gr_inputs.generate_parameters 
                if gr_inputs.generate_parameters is not None
                else GenerateParameters()
            ),
            response_stream=Queue()
        )