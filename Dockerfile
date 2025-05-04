FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install comprehensive development environment
RUN apt update && apt install -y \
    build-essential g++ gcc cmake make autoconf automake libtool pkg-config git curl wget unzip python3-dev \
    python3-pip python3-venv python-is-python3 libssl-dev libbz2-dev liblzma-dev zlib1g-dev libffi-dev \
    libsqlite3-dev libreadline-dev libgdbm-dev libdb-dev libexpat1-dev libxml2-dev \
    libxslt1-dev libyaml-dev libevent-dev libboost-all-dev libprotobuf-dev protobuf-compiler \
    libcurl4-openssl-dev libjpeg-dev libpng-dev libtiff-dev libfreetype-dev libx11-dev \
    libxext-dev libxrender-dev libxrandr-dev libxi-dev libxtst-dev libxinerama-dev libxkbcommon-dev libxkbcommon-x11-dev \
    libfontconfig1-dev libharfbuzz-dev libpango1.0-dev libcairo2-dev libgtk-3-dev \
    qtbase5-dev qttools5-dev-tools libtbb-dev libopenblas-dev liblapack-dev libatlas-base-dev \
    libsuitesparse-dev libeigen3-dev libgmp-dev libmpfr-dev libboost-python-dev libbz2-dev liblz4-dev \
    libzstd-dev libarchive-dev libsnappy-dev libuv1-dev librocksdb-dev libwebp-dev libxmlsec1-dev libgsl-dev \
    libgflags-dev libgoogle-glog-dev libhdf5-dev libtiff5-dev libyaml-cpp-dev libgd-dev default-jdk \
    openjdk-11-jdk openjdk-17-jdk maven gradle nodejs npm ruby-dev perl lua5.3 rustc cargo golang-go clang llvm lldb valgrind \
    ccache lcov doxygen graphviz gdb bison flex swig ninja-build libapache2-mod-php php-cli php-dev \
    apt-utils software-properties-common vim nano emacs htop neofetch screen tmux git-lfs \
    sqlite3 postgresql-client mysql-client redis-tools openssh-client rsync zip p7zip-full \
    jq yq parallel locales pipx

# Setup locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"] 