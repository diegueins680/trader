FROM haskell:8.10.4 AS build

RUN sed -i 's|deb.debian.org/debian|archive.debian.org/debian|g' /etc/apt/sources.list \
  && sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list \
  && sed -i '/buster-updates/d' /etc/apt/sources.list \
  && apt-get -o Acquire::Check-Valid-Until=false update \
  && apt-get install -y --no-install-recommends ca-certificates gnupg wget \
  && wget -qO - https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt buster-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
  && apt-get -o Acquire::Check-Valid-Until=false update \
  && apt-get install -y --no-install-recommends libpq-dev pkg-config \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/trader

# Copy only the Haskell project for better caching.
COPY haskell/trader.cabal haskell/trader.cabal
COPY haskell/app haskell/app
COPY haskell/test haskell/test

WORKDIR /opt/trader/haskell
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cabal update
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cabal build exe:trader-hs exe:optimize-equity exe:merge-top-combos
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cp "$(cabal list-bin trader-hs)" /opt/trader/trader-hs
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cp "$(cabal list-bin optimize-equity)" /opt/trader/optimize-equity
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cp "$(cabal list-bin merge-top-combos)" /opt/trader/merge-top-combos

FROM debian:bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates libgmp10 libtinfo6 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/trader/trader-hs /usr/local/bin/trader-hs
COPY --from=build /opt/trader/optimize-equity /usr/local/bin/optimize-equity
COPY --from=build /opt/trader/merge-top-combos /usr/local/bin/merge-top-combos

WORKDIR /opt/trader/haskell
COPY haskell/web/public /opt/trader/haskell/web/public

ENV TRADER_STATE_DIR=/var/lib/trader/state
ENV TRADER_API_ASYNC_DIR=/var/lib/trader/async
ENV TRADER_LSTM_WEIGHTS_DIR=/var/lib/trader/lstm

RUN mkdir -p /var/lib/trader/async /var/lib/trader/lstm /var/lib/trader/state /opt/trader/haskell/.tmp/optimizer \
  && chown -R 65532:65532 /var/lib/trader /opt/trader/haskell/.tmp

VOLUME ["/var/lib/trader"]

EXPOSE 8080

USER 65532:65532

CMD ["trader-hs", "--serve", "--port", "8080"]
