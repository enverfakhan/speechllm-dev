FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# weights and data are mounted at runtime — not baked into the image
VOLUME ["/weights", "/data"]

ENTRYPOINT ["python", "train.py"]
