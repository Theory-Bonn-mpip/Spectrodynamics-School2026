FROM ubuntu:24.04

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cmake \
    pkg-config \
    pkgconf \
    ninja-build \
    wget \
    curl \
    unzip \
    openscad \
    ffmpeg \
    libsm6 \
    libxext6 \
    gnuplot \
    iproute2 \
    s3cmd \
    rsync \
    vim \
    openssh-client \
    openssh-server \
    git \
    git-lfs \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    openmpi-bin \
    libopenmpi-dev \
    libscalapack-mpi-dev \
    libboost-dev \
    libcgal-dev \
    libelpa-dev \
    libetsf-io-dev \
    libfftw3-dev \
    libgmp-dev \
    libgsl-dev \
    libmetis-dev \
    libmpfr-dev \
    libnetcdff-dev \
    libnlopt-dev \
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/*


# create user with a home directory
ARG NB_USER=spectrodyn_user
ARG NB_UID=1001
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}
ENV OMP_NUM_THREADS 10

RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER} || true

USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
# install the notebook package
RUN pip install --break-system-packages --no-cache --upgrade pip

RUN pip install --break-system-packages --no-cache \
    notebook \
    jupyterlab \
    jupyter-collaboration \
    jupyterlab-git \
    jupyterhub==4.1.5 \
    holoviews \
    sympy \
    tqdm \
    ase \
    'chemiscope>=1.0' \
    'ipi>=3.2.0' \
    matplotlib \
    ffmpeg-python \
    jupytext \
    nbstripout

# ============================================================
# MACE

# MACE with MDP atomic-decomposition outputs
# ============================================================
RUN pip install \
    --break-system-packages \
    --no-cache-dir \
    "git+https://github.com/MPIP-STREAM/mace.git@MDP-derivatives-with-bec_training"

RUN pip install "git+https://gitlab.com/octopus-code/postopus.git"

RUN pip install --break-system-packages --no-cache nbstripout

RUN find ${HOME} -type f -name "*.ipynb" \
    -exec nbstripout {} \;

# ============================================================
# CLONE + BUILD CODES
# ============================================================

USER root
RUN mkdir -p ${HOME}/Codes
RUN chown -R ${NB_UID}:${NB_UID} ${HOME}

USER ${NB_USER}

WORKDIR ${HOME}/Codes

# ------------------------------------------------------------
# DFTB+
# ------------------------------------------------------------
RUN git clone https://github.com/dftbplus/dftbplus.git

WORKDIR ${HOME}/Codes/dftbplus

RUN FC=gfortran CC=gcc cmake -DINSTANCE_SAFE_BUILD=True -B _build .
WORKDIR ${HOME}/Codes/dftbplus/_build
RUN cmake --build . -- -j 4
RUN cmake --install .
ENV DFTB_INSTALL=${HOME}/Codes/dftbplus/_build/_install

# ------------------------------------------------------------
# MARADO
# ------------------------------------------------------------
WORKDIR ${HOME}/Codes
RUN git clone https://github.com/charlyqchm/MARADO.git
WORKDIR ${HOME}/Codes/MARADO
RUN make
# ------------------------------------------------------------
# OCTOPUS
# ------------------------------------------------------------
WORKDIR ${HOME}/Codes

RUN git clone https://gitlab.com/octopus-code/octopus.git

WORKDIR ${HOME}/Codes/octopus

RUN cmake -G Ninja -B _build --fresh --install-prefix=${HOME}/Codes/octopus/_build/app -DOCTOPUS_OpenMP=On -DOCTOPUS_MPI=On -DOCTOPUS_APP_TESTS=Off -DOCTOPUS_ScaLAPACK=On -DOCTOPUS_NATIVE=Off .
RUN cmake --build ./_build
RUN cmake --install ./_build


WORKDIR ${HOME}
COPY . ${HOME}
#Here build DFTB+ and Omxrta
######

# add /home/postopus_user/.local/bin' to path
ENV PATH="${HOME}/.local/bin:${PATH}"
# ENTRYPOINT ["/bin/bash"]
