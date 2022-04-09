FROM easyfl:base

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["python", "examples/remote_client.py"]
