# The SageMaker vLLM container with one file changed, for the NVIDIA T4 (Turing, SM 7.5).
#
# Gemma 4's full-attention layers are 512 wide, which makes vLLM pick its Triton
# attention kernel. That kernel's tile needs 98,304 bytes of shared memory per
# block, and Turing allows 65,536. patch_triton_turing.py halves the tiles on
# pre-Ampere GPUs only (a no-op on L4 and L40S). Copied from
# ~/gemma4-dev/gpu-vllm-t4-2b-w4a16, where it serves on a GCE T4.
#
# Built by CodeBuild (buildspec.yml) and pushed to this account's ECR; set
# IMAGE_URI to the pushed tag. The base image's entrypoint is kept, so SM_VLLM_*
# variables work as they do on the stock image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY patch_triton_turing.py /opt/turing/patch_triton_turing.py

# Resolve the installed module path without importing vLLM (no GPU at build time),
# patch it, confirm the clamp is present, and refuse the build if torch was not
# compiled for SM 7.5.
RUN set -eu; \
    target="$(python3 -c 'import importlib.util, os; print(os.path.join(importlib.util.find_spec("vllm").submodule_search_locations[0], "v1/attention/ops/triton_unified_attention.py"))')"; \
    echo "patch target: $target"; \
    python3 /opt/turing/patch_triton_turing.py "$target"; \
    python3 /opt/turing/patch_triton_turing.py --check "$target"; \
    python3 -c 'import torch, sys; a = torch._C._cuda_getArchFlags(); print("torch arch:", a); sys.exit(0 if "sm_75" in a else 1)'
