# Dockerfile Optimization Analysis for Haskell Trading Bot

## Current State Summary
- **Current Image Size**: 212MB
- **Base Runtime Image**: debian:bookworm-slim (74.8MB)
- **Binary Sizes**: trader-hs (67MB), optimize-equity (32MB), merge-top-combos (22MB)
- **Runtime Dependencies**: libpq5, libgmp10, libtinfo6, ca-certificates (12.2MB)
- **Static Files**: web/public (~190KB)
- **Build System**: Cabal with cache mounts

---

## Optimization Recommendations

### 1. Build Efficiency & Layer Caching Optimization

#### 1.1 Improve Dependency Caching (HIGH PRIORITY)
**Current Issue**: The Dockerfile copies the entire cabal file and all source code before building, which invalidates the cache whenever any source file changes.

**Recommendation**: Separate dependency installation from source code building
```dockerfile
# Copy only cabal file first to cache dependencies
COPY haskell/trader.cabal haskell/trader.cabal
RUN --mount=type=cache,target=/root/.cabal \
  cabal update && cabal build --only-dependencies

# Then copy source code
COPY haskell/app haskell/app
COPY haskell/test haskell/test
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cabal build exe:trader-hs exe:optimize-equity exe:merge-top-combos
```

**Impact**: Reduces rebuild time from full dependency recompilation to just changed modules when source code changes.

#### 1.2 Consolidate Binary Extraction (MEDIUM PRIORITY)
**Current Issue**: 3 separate RUN commands with identical mount declarations for extracting binaries.

**Recommendation**: Combine into single RUN command
```dockerfile
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cp "$(cabal list-bin trader-hs)" /opt/trader/trader-hs && \
  cp "$(cabal list-bin optimize-equity)" /opt/trader/optimize-equity && \
  cp "$(cabal list-bin merge-top-combos)" /opt/trader/merge-top-combos
```

**Impact**: Reduces layer count and build overhead.

#### 1.3 Remove Redundant cabal update (LOW PRIORITY)
**Current Issue**: `cabal update` runs in a separate layer before build.

**Recommendation**: Combine with dependency installation (see 1.1).

---

### 2. Security Enhancements

#### 2.1 Pin Base Image Versions (HIGH PRIORITY - CRITICAL)
**Current Issue**: 
- `haskell:8.10.4` - pinned ✓
- `debian:bookworm-slim` - **NOT pinned** (uses rolling tag)

**Recommendation**: Use digest pinning for reproducible builds
```dockerfile
FROM debian:bookworm-slim@sha256:<digest>
```

Or use specific version:
```dockerfile
FROM debian:bookworm-20250120-slim
```

**Security Impact**: Prevents unexpected runtime environment changes and supply chain attacks.

#### 2.2 Archive Repository Configuration (MEDIUM PRIORITY)
**Current Issue**: The build stage uses `archive.debian.org` for old Debian buster packages, which is necessary for the old Haskell base image.

**Recommendation**: Consider if this is still needed or if you can update to newer Haskell image:
- `haskell:9.0` or `haskell:9.2` with Debian bullseye
- `haskell:9.4` or `haskell:9.6` with Debian bookworm

**Impact**: Better security posture with maintained packages. If upgrading GHC version, test thoroughly.

#### 2.3 Non-root User Already Implemented ✓
**Status**: Already using UID/GID 65532:65532 (nobody user).

---

### 3. Image Size Reduction Strategies

#### 3.1 Strip Binaries (HIGH PRIORITY)
**Current Issue**: Haskell binaries include debug symbols by default.

**Recommendation**: Strip binaries during build
```dockerfile
RUN --mount=type=cache,target=/root/.cabal \
  --mount=type=cache,target=/opt/trader/haskell/dist-newstyle \
  cp "$(cabal list-bin trader-hs)" /opt/trader/trader-hs && \
  cp "$(cabal list-bin optimize-equity)" /opt/trader/optimize-equity && \
  cp "$(cabal list-bin merge-top-combos)" /opt/trader/merge-top-combos && \
  strip /opt/trader/trader-hs /opt/trader/optimize-equity /opt/trader/merge-top-combos
```

**Expected Reduction**: 30-40% binary size reduction (potentially ~35-50MB saved)
- trader-hs: 67MB → ~40-45MB
- optimize-equity: 32MB → ~20-22MB  
- merge-top-combos: 22MB → ~14-16MB

#### 3.2 Use UPX Compression (MEDIUM PRIORITY - OPTIONAL)
**Recommendation**: Apply UPX compression to binaries for additional size reduction
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends upx && \
    rm -rf /var/lib/apt/lists/*

RUN upx --best --lzma \
    /opt/trader/trader-hs \
    /opt/trader/optimize-equity \
    /opt/trader/merge-top-combos
```

**Expected Reduction**: Additional 50-70% reduction on already-stripped binaries
**Trade-off**: Slower startup time (decompression overhead), but significant size savings

**Total potential savings with strip + UPX**: Could reduce from 212MB to ~130-150MB

#### 3.3 Consider Alpine Base Image (LOW PRIORITY)
**Current Runtime**: debian:bookworm-slim (74.8MB)
**Alternative**: alpine:3.19 (~7MB)

**Challenges**:
- Haskell binaries are dynamically linked against glibc
- Would require static linking or musl-based build
- Significant build changes required

**Recommendation**: Not recommended unless size is critical. The complexity outweighs benefits for this application.

---

### 4. Additional Production Best Practices

#### 4.1 Health Check (HIGH PRIORITY)
**Missing**: No HEALTHCHECK directive

**Recommendation**: Add health check for the web service
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["/usr/local/bin/trader-hs", "--health-check"] || exit 1
```

Or if no built-in health check:
```dockerfile
# Install curl in runtime stage
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

**Impact**: Better orchestration support (Kubernetes, ECS, Docker Compose).

#### 4.2 Build Arguments for Versioning (MEDIUM PRIORITY)
**Recommendation**: Add build arguments for tracking
```dockerfile
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.title="Trader API" \
      org.opencontainers.image.description="LSTM-based trading bot"
```

#### 4.3 Optimize .dockerignore (ALREADY GOOD ✓)
**Status**: .dockerignore properly configured excluding:
- .git, node_modules, dist-newstyle, .stack-work
- Only includes necessary web/public files

#### 4.4 Multi-Architecture Support (OPTIONAL)
**Current**: Single architecture (amd64)

**Recommendation**: Consider buildx for multi-arch if deploying to ARM
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t trader-api .
```

---

## Priority Implementation Order

### Phase 1: Quick Wins (Immediate Implementation)
1. **Strip binaries** - 30-40MB savings, no risk
2. **Pin debian base image digest** - Security improvement
3. **Consolidate binary extraction** - Cleaner Dockerfile
4. **Add HEALTHCHECK** - Production readiness

### Phase 2: Build Optimization (Next Sprint)
1. **Separate dependency caching** - Faster rebuilds
2. **Add build labels/args** - Better tracking

### Phase 3: Advanced Optimization (If Needed)
1. **UPX compression** - If size is critical
2. **Upgrade Haskell version** - If compatible with codebase

---

## Expected Results After Phase 1

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Image Size | 212MB | ~165-175MB | ~20-22% |
| Binary Size | 121MB | ~75-85MB | ~30-38% |
| Build Time (no cache) | Baseline | Same | - |
| Build Time (code change) | Baseline | 50-70% faster | Major |
| Security Score | Good | Better | Pinned versions |
| Production Ready | Good | Better | Health checks |

---

## Implementation Checklist

- [ ] Strip binaries in build stage
- [ ] Pin debian:bookworm-slim with digest
- [ ] Consolidate binary extraction into single RUN
- [ ] Add HEALTHCHECK directive
- [ ] Separate dependency build from source build
- [ ] Add build metadata labels
- [ ] Test optimized image thoroughly
- [ ] Update CI/CD pipeline if needed
- [ ] Optional: Implement UPX compression
- [ ] Optional: Consider Haskell version upgrade

---

## Notes

**Benchmark Testing Recommended**:
- Test stripped binaries don't break functionality
- Verify UPX compressed binaries if implemented
- Measure actual build time improvements
- Test health check endpoint reliability

**Deployment Considerations**:
- Existing volumes will work unchanged
- Environment variables remain the same
- Port mappings unchanged
- Non-root user already configured

The optimizations focus on practical improvements with measurable impact while maintaining application functionality and deployment compatibility.
