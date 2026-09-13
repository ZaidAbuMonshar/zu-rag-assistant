# Base image: an official, minimal Debian + Python 3.12 (matches the Python
# version used locally and on the Session 9 VM -- see .venv/pyvenv.cfg).
FROM python:3.12-slim

# llama-cpp-python has no official prebuilt wheel for every platform on PyPI --
# pip builds it from source on install, which needs a C/C++ compiler and cmake.
# Without these, `pip install -r requirements.txt` fails partway through.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copied alone, before the rest of the code, so Docker can cache this
# (slow) dependency-install layer and skip re-running it on every code
# change -- it only reruns if requirements.txt itself changes.
COPY requirements.txt .

# requirements.txt was captured via `pip freeze` on the Session 9 VM and pins
# a plain `torch==2.13.0`. On PyPI that resolves to the CUDA-bundled build,
# which drags in ~2-3GB of unused nvidia-*/triton packages even though this
# whole stack is CPU-only everywhere (see CLAUDE.md open item 16). Installing
# the CPU-only build FIRST means later installs see torch already satisfied
# and never reach out for a different (CUDA) copy.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Install everything else. torch is already installed above, so this line in
# requirements.txt is filtered out to avoid a duplicate/CUDA reinstall.
RUN grep -v '^torch==' requirements.txt > requirements.nocuda.txt \
    && pip install --no-cache-dir -r requirements.nocuda.txt

# Application code only -- NOT data/. The Chroma DB and the 2GB Qwen model
# file are bind-mounted in at `docker run` time instead (see the run command),
# so rebuilding the image after a code change never has to touch multi-GB
# files. main.py/generate.py/build_chroma_db.py compute their paths from
# PROJECT_ROOT (derived from __file__), so as long as data/ ends up at
# /app/data at runtime, no code changes are needed.
COPY src/ src/

EXPOSE 8000

# Same command already used locally/on the VM (see src/api/main.py's
# docstring) -- just with --host 0.0.0.0 so it accepts connections from
# outside the container, not only from localhost inside it.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
