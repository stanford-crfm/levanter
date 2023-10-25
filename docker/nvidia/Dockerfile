FROM ghcr.io/nvidia/jax:nightly-2023-10-25-linux-amd64
LABEL authors="dlwh"

# copy everything from the repo into the container
COPY . /opt/levanter

ARG GIT_USER_EMAIL
ARG GIT_USER_NAME
RUN <<EOF bash -e
git config --global user.email "${GIT_USER_EMAIL}"
git config --global user.name "${GIT_USER_NAME}"
EOF

WORKDIR /opt/levanter

RUN pip install -e .
