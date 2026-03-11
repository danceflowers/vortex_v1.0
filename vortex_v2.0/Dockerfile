FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TOOLDIR=/home/chenbangyu/tools
ENV OSVERSION=ubuntu/focal
ENV VORTEX_HOME=/mnt/d/wode_code_trunk/vortex
ENV INSTALLDIR=/mnt/d/wode_code_trunk/vortex
ENV VORTEX_DRIVER=simx
ENV LD_LIBRARY_PATH=/mnt/d/wode_code_trunk/vortex/runtime

WORKDIR /mnt/d/wode_code_trunk/vortex

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates wget git python3 make build-essential \
      software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /mnt/d/wode_code_trunk /home/chenbangyu/tools

COPY . /mnt/d/wode_code_trunk/vortex

RUN sed "s|@TOOLDIR@|${TOOLDIR}|g; s|@OSVERSION@|${OSVERSION}|g" \
        ./ci/toolchain_install.sh.in > ./ci/toolchain_install.sh && \
    sed "s|@TOOLDIR@|${TOOLDIR}|g; s|@OSVERSION@|${OSVERSION}|g" \
        ./ci/toolchain_env.sh.in > ./ci/toolchain_env.sh && \
    chmod +x ./ci/install_dependencies.sh ./ci/toolchain_install.sh ./ci/toolchain_env.sh && \
    ./ci/install_dependencies.sh && \
    ./ci/toolchain_install.sh --riscv32 --llvm --libcrt32 --libc32

RUN test -f /mnt/d/wode_code_trunk/vortex/third_party/ramulator/ext/spdlog/include/spdlog/spdlog.h && \
    test -d /mnt/d/wode_code_trunk/vortex/third_party/ramulator/ext/yaml-cpp && \
    cmake -S /mnt/d/wode_code_trunk/vortex/third_party/ramulator \
          -B /mnt/d/wode_code_trunk/vortex/third_party/ramulator/build \
          -DFETCHCONTENT_FULLY_DISCONNECTED=ON && \
    cmake --build /mnt/d/wode_code_trunk/vortex/third_party/ramulator/build -j"$(nproc)" && \
    cp /mnt/d/wode_code_trunk/vortex/third_party/ramulator/build/libramulator.so \
       /mnt/d/wode_code_trunk/vortex/third_party/ramulator/

CMD ["bash"]
