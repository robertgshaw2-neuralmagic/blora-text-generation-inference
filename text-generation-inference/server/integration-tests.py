import torch, os, gc, tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel, LoraConfig
from text_generation_server.models.blora_flash_llama import BLoraFlashLlama
from text_generation_server.models.flash_llama import FlashLlama
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch
from text_generation_server.pb import generate_pb2

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if os.environ["HUGGING_FACE_HUB_TOKEN"] == "":
    raise ValueError("Please set your HUGGING_FACE_HUB_TOKEN")

HF_TOKEN = os.environ["HUGGING_FACE_HUB_TOKEN"]
MODEL_ID = "meta-llama/Llama-2-7b-hf"

def make_warmup_requests():
    max_input_length = 128
    max_batch_size = 2
    max_prefill_tokens = max_input_length * max_batch_size - 32
    warmup_requests = []
    n_tokens = 0
    while n_tokens < max_prefill_tokens:
        warmup_requests.append(
            generate_pb2.Request(
                id=0,
                inputs="_text" * max_input_length,
                truncate=min(max_input_length, max_prefill_tokens - n_tokens),
                parameters=generate_pb2.NextTokenChooserParameters(
                    do_sample=False
                ),
                stopping_parameters=generate_pb2.StoppingCriteriaParameters(
                    max_new_tokens=2
                )
            ),
        )            
        n_tokens += max_input_length

    return warmup_requests

def run_blora(configs, max_new_tokens=100):
    
    total = len(configs)
    for idx, config in enumerate(configs):
        print(f"Running BLORA Example {idx + 1} // {total}")

        llama = FlashLlama(
            model_id=MODEL_ID, 
            dtype=torch.float16, 
            quantize="bitsandbytes-nf4"
        )

        lora_id = config["lora_id"]
        blora_llama = BLoraFlashLlama(llama, {lora_id: LoraConfig.from_pretrained(lora_id)})

        # warmup
        warmup_requests = make_warmup_requests()
        fclm_warmup_batch = FlashCausalLMBatch.from_pb(
            pb=generate_pb2.Batch(id=0, requests=warmup_requests, size=len(warmup_requests)),
            tokenizer=blora_llama.model.tokenizer,
            dtype=blora_llama.model.dtype,
            device=blora_llama.model.device,
        )
        blora_llama.set_batch_ids(
            lora_ids=[lora_id]*len(warmup_requests), 
            cu_seqlen_prefill=fclm_warmup_batch.cu_seqlen_prefill
        )
        _ = blora_llama.model.warmup(batch=fclm_warmup_batch)
        del fclm_warmup_batch
        gc.collect()
        torch.cuda.empty_cache()

        # run inference
        parameters = generate_pb2.NextTokenChooserParameters(
            watermark=False,
            temperature=1.0,
            repetition_penalty=1.0,
            top_k=0,
            top_p=1.0,
            typical_p=1.0,
            do_sample=False
        )

        stopping_parameters = generate_pb2.StoppingCriteriaParameters(
            max_new_tokens=max_new_tokens,
            ignore_eos_token=False,
        )

        for prompt in config["prompts"]:
            request = generate_pb2.Request(
                id=0,
                inputs=prompt,
                truncate=128,
                parameters=parameters,    
                stopping_parameters=stopping_parameters
            )

            fclm_batch = FlashCausalLMBatch.from_pb(
                pb=generate_pb2.Batch(id=0, requests=[request]),
                tokenizer=blora_llama.model.tokenizer,
                dtype=blora_llama.model.dtype,
                device=blora_llama.model.device,
            )

            tokens = []
            blora_llama.set_batch_ids(
                lora_ids=[lora_id], 
                cu_seqlen_prefill=fclm_batch.cu_seqlen_prefill
            )

            for _ in range(max_new_tokens):
                generations, fclm_batch = blora_llama.model.generate_token(fclm_batch)
                for generation in generations:
                    tokens.append(generation.token_id)
            config["blora_generations"].append(tokens)

            del fclm_batch
            gc.collect()
            torch.cuda.empty_cache()

        
        del blora_llama
        del llama
        gc.collect()
        torch.cuda.empty_cache()

    return configs

def run_peft(configs, max_new_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
    )

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    total = len(configs)
    for idx, config in enumerate(configs):
        print(f"Running PEFT Example {idx + 1} // {total}")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
            token=HF_TOKEN,
        )
        peft_model = PeftModel.from_pretrained(model, config["lora_id"])
        
        for prompt in config["prompts"]:
            model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = peft_model.generate(
                **model_inputs,
                generation_config=generation_config
            )
            config["peft_generations"].append(output[0].tolist()[model_inputs["input_ids"].shape[1]:])

        del model
        del peft_model
        del model_inputs
        gc.collect()
        torch.cuda.empty_cache()

    return configs

def run_model_tests():
    configs = [
        {
            'lora_id': 'nealchandra/llama-2-7b-hf-lora-alpaca-json',
            'prompts': [
                '### INPUT:\n```json\n{"instructions": "Explain what an alpaca is"}\n```\n### OUTPUT:\n',
                # '### INPUT:\n```json\n{"instructions": "Explain what deep learning is"}\n```\n### OUTPUT:\n'
            ],
            'peft_generations': [],
            'blora_generations': [],
        },
    ]
    
    configs = run_peft(configs, max_new_tokens=100)
    configs = run_blora(configs, max_new_tokens=100)

    for config in configs:
        for peft_generation, blora_generation in zip(config["peft_generations"], config["blora_generations"]):

            for token_p, token_b in zip(peft_generation, blora_generation):
                print(f"{token_p}, {token_b}")

if __name__ == "__main__":
    run_model_tests()