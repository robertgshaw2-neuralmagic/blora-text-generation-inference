```bash
token={TOKEN}
```

```bash
server=$PWD/text-generation-inference/server
volume=$PWD/data
```

```bash
sudo docker run --gpus all --shm-size 1g --network host -v $volume:/data -v $server:/usr/src/server-dev -e HUGGING_FACE_HUB_TOKEN=$token -it tgi
```

```bash
pip uninstall text_generation_server -y
cd server-dev
pip install -e .
jupyter notebook --allow-root
```