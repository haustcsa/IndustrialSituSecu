# 第一阶段：构建阶段
FROM highercomve/tpm2-toolbox AS builder
COPY Python-3.10.3.tgz /root
WORKDIR /root
RUN tar -zxvf Python-3.10.3.tgz \
    && cd Python-3.10.3 \
    && ./configure --enable-optimizations \
    && make

# 第二阶段：最终镜像
FROM highercomve/tpm2-toolbox
LABEL authors="seehar"
COPY --from=builder /root/Python-3.10.3 /root/Python-3.10.3
COPY pip.conf /etc/pip.conf
WORKDIR /root/Python-3.10.3
RUN make install \
    && rm -rf /root/Python-3.10.3.tgz /root/Python-3.10.3
