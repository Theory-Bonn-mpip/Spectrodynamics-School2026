FROM fangohr/octopus:14.0

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
    openmpi-bin \
    libopenmpi-dev \
    libscalapack-mpi-dev \
    && rm -rf /var/lib/apt/lists/*


# create user with a home directory
ARG NB_USER=spectrodyn_user
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}
ENV OMP_NUM_THREADS 10

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

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
    chemiscope \
    ipi \
    ffmpeg-python \
    ase \
    mace-torch \
    jupytext \
    postopus

# CPU version
RUN pip install --break-system-packages --no-cache \
    torch \
    mace-torch

RUN pip install --break-system-packages --no-cache nbstripout

RUN find ${HOME} -type f -name "*.ipynb" \
    -exec nbstripout {} \;

WORKDIR ${HOME}

RUN git clone https://github.com/dftbplus/dftbplus.git
RUN git clone https://github.com/charlyqchm/One-Mxll-to-Rule-Them-All.git

COPY . ${HOME}
#Here build DFTB+ and Omxrta
######

# add /home/postopus_user/.local/bin' to path
ENV PATH="${HOME}/.local/bin:${PATH}"
# ENTRYPOINT ["/bin/bash"]
