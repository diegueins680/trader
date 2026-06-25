FROM node:22-alpine AS build
WORKDIR /app

ARG TRADER_GIT_COMMIT=""
ENV TRADER_GIT_COMMIT=${TRADER_GIT_COMMIT}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.d/40-trader-config.sh /docker-entrypoint.d/40-trader-config.sh
COPY --from=build /app/dist /usr/share/nginx/html

RUN chmod +x /docker-entrypoint.d/40-trader-config.sh

EXPOSE 8080
