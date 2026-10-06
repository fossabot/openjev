# OpenJev: vLLM with DiffusionGemma structured reads, plus the Jev-compatible API server.
# Build docker/Dockerfile.base first. Laya and Verdict start from the same base, in
# docker/Dockerfile.laya and docker/Dockerfile.verdict.
#   docker build -f docker/Dockerfile -t razorback16/openjev .
#
# Upstream main as of 2026-10-06, the newest commit with a published wheel, so the
# compiled kernels are downloaded rather than built: no CUDA build here. Since the
# #57250 merge it adds the DiffusionGemma fixes and speedups: constrained reads
# (#58216), a one-pass sampler statistics kernel (#58226), quant_config on the LM
# head (#48521), and an attention mask that no longer freezes under CUDA graph
# replay (#51994). Neither patch below is upstream yet; both still apply here.
ARG BASE_IMAGE=razorback16/openjev-base:cu130-torch2.13
FROM ${BASE_IMAGE}

ARG VLLM_REPO=https://github.com/vllm-project/vllm
ARG VLLM_COMMIT=a3e0243b1c3ea313b902a43b0f6559fb4148ff81
ARG VLLM_BASE_COMMIT=a3e0243b1c3ea313b902a43b0f6559fb4148ff81

# FlashInfer compiles its NVFP4 MoE kernels with nvcc on first start. This is the
# subset of the CUDA toolkit upstream vLLM's own image installs for that, rather
# than the whole devel image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 cuda-nvrtc-dev-13-0 \
        cuda-cuobjdump-13-0 libcurand-dev-13-0 libcublas-dev-13-0 libnuma-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH

RUN git clone --filter=blob:none "$VLLM_REPO" /opt/vllm && git -C /opt/vllm checkout "$VLLM_COMMIT"
# A choice may have 255 options, and one read asks for every option's exact
# logprob. Upstream's cap of 128 ids per request is the only thing in the way;
# it also sizes a per-request buffer in the logprob kernel (512 ids is 2 KiB a
# request). The grep fails the build if the line ever moves or changes.
RUN grep -q '^MAX_LOGPROB_TOKEN_IDS = 128$' /opt/vllm/vllm/sampling_params.py \
    && sed -i 's/^MAX_LOGPROB_TOKEN_IDS = 128$/MAX_LOGPROB_TOKEN_IDS = 512/' /opt/vllm/vllm/sampling_params.py
# Image tokens attend bidirectionally inside their own span, as this checkpoint
# was trained to expect. The patch fails the build if its anchors moved.
COPY docker/patches/vision_prefix_lm.py /tmp/
RUN python /tmp/vision_prefix_lm.py /opt/vllm/vllm/model_executor/models/diffusion_gemma.py \
    && cd /opt/vllm && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_BASE_COMMIT" \
       uv pip install --python /opt/venv/bin/python . \
    && rm -rf /root/.cache/uv

# The API server goes last so a code change only rebuilds a few small layers.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY openjev ./openjev
RUN uv pip install --python /opt/venv/bin/python --no-cache .
COPY docker/entrypoint.sh /usr/local/bin/openjev-entrypoint

ENV OPENJEV_HOST=0.0.0.0 OPENJEV_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=900s \
    CMD curl -sf http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["openjev-entrypoint"]
