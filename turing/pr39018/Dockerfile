# vllm-project/vllm#39018 (head f7b6f506) applied to the SageMaker vLLM 0.30.0 image, to test
# that PR itself on a T4. Nothing from ../patch_triton_turing.py is in this image.
# The build refuses unless the image's two files are byte-identical to upstream v0.30.0.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY orig/ /opt/pr39018/orig/
COPY patched/ /opt/pr39018/patched/
RUN set -eu; \
    root="$(python3 -c 'import importlib.util; print(importlib.util.find_spec("vllm").submodule_search_locations[0])')"; \
    tua="$root/v1/attention/ops/triton_unified_attention.py"; flex="$root/v1/attention/backends/flex_attention.py"; \
    cmp "$tua" /opt/pr39018/orig/triton_unified_attention.py; \
    cmp "$flex" /opt/pr39018/orig/flex_attention.py; \
    echo "image files match upstream v0.30.0"; \
    cp /opt/pr39018/patched/triton_unified_attention.py "$tua"; \
    cp /opt/pr39018/patched/flex_attention.py "$flex"; \
    python3 -m py_compile "$tua" "$flex"; \
    grep -n "max_shared_memory < 98304" "$tua"; \
    ! grep -q "Turing shared-memory clamp" "$tua"
