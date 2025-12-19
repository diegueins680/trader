FROM haskell:8.10.4 AS build

WORKDIR /opt/trader

# Copy only the Haskell project for better caching.
COPY haskell/trader.cabal haskell/trader.cabal
COPY haskell/app haskell/app
COPY haskell/test haskell/test

WORKDIR /opt/trader/haskell
RUN cabal update
RUN cabal build exe:trader-hs
RUN cp "$(cabal list-bin trader-hs)" /opt/trader/trader-hs

FROM debian:bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates libgmp10 libtinfo6 python3 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/trader/trader-hs /usr/local/bin/trader-hs

WORKDIR /opt/trader/haskell
COPY haskell/scripts /opt/trader/haskell/scripts
COPY haskell/web/public /opt/trader/haskell/web/public

ENV TRADER_API_ASYNC_DIR=/var/lib/trader/async
ENV TRADER_LSTM_WEIGHTS_DIR=/var/lib/trader/lstm

RUN mkdir -p /var/lib/trader/async /var/lib/trader/lstm /opt/trader/haskell/.tmp/optimizer \
  && chown -R 65532:65532 /var/lib/trader /opt/trader/haskell/.tmp

EXPOSE 8080

USER 65532:65532

CMD ["trader-hs", "--serve", "--port", "8080"]
