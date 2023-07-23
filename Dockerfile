FROM conda/miniconda3
WORKDIR /app

COPY . .

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cuda -c pytorch -c "nvidia/label/cuda-11.7.1"
RUN pip install -r requirements.txt
RUN pip install --pre dgl -f https://data.dgl.ai/wheels/cu117/repo.html
RUN pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html