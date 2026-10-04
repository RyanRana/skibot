# The policy API as a container, for AWS App Runner (or ECS / any container host). Same server as modal_app.py.
#   docker build -t hi-policy-api policy_api/
#   docker run -p 8000:8000 hi-policy-api
FROM python:3.12-slim
RUN pip install --no-cache-dir "fastapi[standard]" numpy msgpack
WORKDIR /app
COPY hi_server.py .
COPY policies ./policies
ENV HI_POLICY_DIR=/app/policies
EXPOSE 8000
CMD ["uvicorn", "hi_server:app", "--host", "0.0.0.0", "--port", "8000"]
