# syntax=docker/dockerfile:1

FROM golang:1.26-bookworm AS build

RUN git clone --depth 1 https://github.com/grafana/sigil.git /src

WORKDIR /src/sigil

ENV CGO_ENABLED=0

RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/go/pkg/mod \
    go build -trimpath -ldflags="-s -w" -o /out/sigil ./cmd/sigil

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r sigil && useradd -r -u 10001 -g sigil sigil

WORKDIR /app

COPY --from=build /out/sigil /usr/local/bin/sigil

USER sigil:sigil

EXPOSE 8080
EXPOSE 4317

ENTRYPOINT ["/usr/local/bin/sigil"]
