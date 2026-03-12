FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TOOLDIR=/home/chenbangyu/tools
ENV OSVERSION=ubuntu/focal
ENV VORTEX_HOME=/mnt/d/wode_code_trunk/vortex
ENV INSTALLDIR=/mnt/d/wode_code_trunk/vortex
ENV VORTEX_DRIVER=simx
ENV LD_LIBRARY_PATH=/mnt/d/wode_code_trunk/vortex/runtime

ARG SOFTFLOAT_REPO=https://github.com/ucb-bar/berkeley-softfloat-3.git
ARG SOFTFLOAT_REF=b51ef8f3201669b2288104c28546fc72532a1ea4
ARG RAMULATOR_REPO=https://github.com/CMU-SAFARI/ramulator2.git
ARG RAMULATOR_REF=e62c84a6f0e06566ba6e182d308434b4532068a5
ARG SPDLOG_REPO=https://github.com/gabime/spdlog.git
ARG SPDLOG_REF=v1.11.0
ARG YAMLCPP_REPO=https://github.com/jbeder/yaml-cpp.git
ARG YAMLCPP_REF=yaml-cpp-0.7.0
ARG ARGPARSE_REPO=https://github.com/p-ranav/argparse.git
ARG ARGPARSE_REF=v2.9

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

RUN set -eux; \
    ensure_checkout() { \
      dest="$1"; \
      url="$2"; \
      ref="$3"; \
      marker="$4"; \
      if [ -e "$marker" ]; then \
        return 0; \
      fi; \
      if [ -d "$dest" ] && [ -z "$(ls -A "$dest")" ]; then \
        rmdir "$dest"; \
      fi; \
      if [ ! -d "$dest" ]; then \
        git clone "$url" "$dest"; \
      elif ! git -C "$dest" rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
        echo "missing required source tree at $dest" >&2; \
        exit 1; \
      fi; \
      if ! git -C "$dest" rev-parse --verify --quiet "${ref}^{commit}" >/dev/null; then \
        git -C "$dest" fetch --tags origin; \
      fi; \
      git -C "$dest" checkout --detach "$ref"; \
    }; \
    mkdir -p "$VORTEX_HOME/third_party" "$VORTEX_HOME/third_party/ramulator/ext"; \
    ensure_checkout \
      "$VORTEX_HOME/third_party/softfloat" \
      "$SOFTFLOAT_REPO" \
      "$SOFTFLOAT_REF" \
      "$VORTEX_HOME/third_party/softfloat/source/include/softfloat.h"; \
    ensure_checkout \
      "$VORTEX_HOME/third_party/ramulator" \
      "$RAMULATOR_REPO" \
      "$RAMULATOR_REF" \
      "$VORTEX_HOME/third_party/ramulator/CMakeLists.txt"; \
    ensure_checkout \
      "$VORTEX_HOME/third_party/ramulator/ext/spdlog" \
      "$SPDLOG_REPO" \
      "$SPDLOG_REF" \
      "$VORTEX_HOME/third_party/ramulator/ext/spdlog/include/spdlog/spdlog.h"; \
    ensure_checkout \
      "$VORTEX_HOME/third_party/ramulator/ext/yaml-cpp" \
      "$YAMLCPP_REPO" \
      "$YAMLCPP_REF" \
      "$VORTEX_HOME/third_party/ramulator/ext/yaml-cpp/include/yaml-cpp/yaml.h"; \
    ensure_checkout \
      "$VORTEX_HOME/third_party/ramulator/ext/argparse" \
      "$ARGPARSE_REPO" \
      "$ARGPARSE_REF" \
      "$VORTEX_HOME/third_party/ramulator/ext/argparse/include/argparse/argparse.hpp"; \
    make -C "$VORTEX_HOME/third_party" softfloat; \
    cmake -S "$VORTEX_HOME/third_party/ramulator" \
          -B "$VORTEX_HOME/third_party/ramulator/build"; \
    cmake --build "$VORTEX_HOME/third_party/ramulator/build" -j"$(nproc)"; \
    test -f "$VORTEX_HOME/third_party/ramulator/libramulator.so"

CMD ["bash"]
