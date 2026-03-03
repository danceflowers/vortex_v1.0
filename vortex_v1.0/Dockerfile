FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TOOLDIR=/opt/vortex-tools
ENV OSVERSION=ubuntu/focal

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates wget git python3 make build-essential \
      software-properties-common && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN chmod +x ./ci/install_dependencies.sh ./ci/toolchain_install.sh && \
    ./ci/install_dependencies.sh && \
    ./ci/toolchain_install.sh --riscv32 --llvm --libcrt32 --libc32

CMD ["bash"]
