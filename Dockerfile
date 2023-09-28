FROM ghcr.io/huggingface/text-generation-inference:0.9.4

RUN pip install jupyter

RUN pip uninstall text_generation_server -y

RUN pip install bitsandbytes peft accelerate scipy --upgrade

ENTRYPOINT ["/bin/bash"]
