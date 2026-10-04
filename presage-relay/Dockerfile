# Presage relay for the Hazard Intelligence game. Build and run:
#   docker build -t presage-relay presage-relay
#   docker run -p 8787:8787 -e PRESAGE_API_KEY=... -e ALLOWED_ORIGINS=https://your-site.vercel.app presage-relay
# On a cloud host (Render, Fly, Railway, AWS) point it at this Dockerfile and set the same two variables as
# secrets. Cloud hosts run x86: build with --platform linux/amd64 if you build it yourself on a Mac.
FROM node:22-bookworm-slim

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev --no-audit --no-fund && npm cache clean --force \
 && cd node_modules/@smartspectra && arch=$(node -p process.arch) \
 && for d in node-sdk-*; do [ "$d" = "node-sdk-linux-$arch" ] || rm -rf "$d"; done
COPY server.mjs ./

ENV PORT=8787 NODE_ENV=production
EXPOSE 8787
USER node
CMD ["node", "server.mjs"]
