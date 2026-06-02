FROM haskell:9.4.8 AS build

RUN sed -i 's|deb.debian.org/debian|archive.debian.org/debian|g' /etc/apt/sources.list \
  && sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list \
  && sed -i '/buster-updates/d' /etc/apt/sources.list \
  && apt-get -o Acquire::Check-Valid-Until=false update \
  && apt-get install -y --no-install-recommends libpq-dev pkg-config \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/trader

# Copy only the Haskell project for better caching.
COPY haskell/trader.cabal haskell/trader.cabal
COPY haskell/.build-commit haskell/.build-commit
COPY haskell/app haskell/app
COPY haskell/test haskell/test

WORKDIR /opt/trader/haskell
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cabal update
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cabal build -j1 --disable-optimization exe:trader-hs exe:optimize-equity exe:merge-top-combos
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cp "$(cabal list-bin --disable-optimization exe:trader-hs)" /opt/trader/trader-hs \
  && cp "$(cabal list-bin --disable-optimization exe:optimize-equity)" /opt/trader/optimize-equity \
  && cp "$(cabal list-bin --disable-optimization exe:merge-top-combos)" /opt/trader/merge-top-combos

FROM debian:bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl libgmp10 libpq5 libtinfo6 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/trader/trader-hs /usr/local/bin/trader-hs
# optimize-equity + merge-top-combos are spawned by the auto-optimizer (research role).
COPY --from=build /opt/trader/optimize-equity /usr/local/bin/optimize-equity
COPY --from=build /opt/trader/merge-top-combos /usr/local/bin/merge-top-combos

WORKDIR /opt/trader/haskell
COPY haskell/web/public /opt/trader/haskell/web/public
COPY top-combos.s3.json /opt/trader/haskell/web/public/top-combos.json

ARG TRADER_GIT_COMMIT=""
ENV TRADER_STATE_DIR=/var/lib/trader/state
ENV TRADER_API_ASYNC_DIR=/var/lib/trader/async
ENV TRADER_LSTM_WEIGHTS_DIR=/var/lib/trader/lstm
ENV TRADER_GIT_COMMIT=${TRADER_GIT_COMMIT}

RUN mkdir -p /var/lib/trader/async /var/lib/trader/lstm /var/lib/trader/state /opt/trader/haskell/.tmp/optimizer \
  && chown -R 65532:65532 /var/lib/trader /opt/trader/haskell/.tmp

VOLUME ["/var/lib/trader"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health >/dev/null || exit 1

USER 65532:65532

CMD ["trader-hs", "--serve", "--port", "8080"]
