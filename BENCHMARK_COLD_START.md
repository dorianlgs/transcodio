# Cold Start Optimization Benchmarking Guide

This guide helps you measure the impact of each optimization on cold start performance.

## Current Baseline: 34.40 seconds

Your goal is to measure each optimization separately to understand its individual impact.

---

## Benchmark Setup

### Prerequisites

1. Have a test audio file ready (e.g., `test_audio.mp3`)
2. Make sure Modal is authenticated: `py -m modal setup`
3. Clear any existing containers between tests

### How to Measure Cold Start Time

Cold start time = time from first request to when transcription begins processing.

**Method 1: Using Modal Logs**
```bash
# Deploy the app
py -m modal deploy modal_app/app.py

# Wait for containers to fully scale down (2+ minutes of inactivity)
# Then run test and check logs
py -m modal run modal_app/app.py test_audio.mp3

# Check logs for "Total initialization time"
py -m modal app logs transcodio-app
```

**Method 2: Using the CLI Tool with Timing**
```bash
# Windows PowerShell
Measure-Command { uv run transcribe_file.py test_audio.mp3 }

# Or manually time with timestamps
uv run transcribe_file.py test_audio.mp3
```

### Forcing a Cold Start

To ensure you're measuring a true cold start:

```bash
# Stop all running containers
py -m modal app stop transcodio-app

# Wait 2-3 minutes for complete shutdown

# Run your test
py -m modal run modal_app/app.py test_audio.mp3
```

---

## Test Scenarios

### Scenario 1: Baseline (Current - No Optimizations)

**Configuration in `config.py`:**
```python
ENABLE_CPU_MEMORY_SNAPSHOT = False
ENABLE_GPU_MEMORY_SNAPSHOT = False
ENABLE_MODEL_WARMUP = False
EXTENDED_IDLE_TIMEOUT = False
```

**Steps:**
1. Edit `config.py` with the above settings
2. Deploy: `py -m modal deploy modal_app/app.py`
3. Wait for complete shutdown (2-3 minutes)
4. Test: `py -m modal run modal_app/app.py test_audio.mp3`
5. Record the "Total initialization time" from logs

**Expected Result:** ~34.40s (your current baseline)

**Record Result:**
```
Test 1 - Baseline
Cold Start Time: _____ seconds
Notes:
```

---

### Scenario 2: CPU Memory Snapshots Only

**Configuration in `config.py`:**
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True   # ← CHANGED
ENABLE_GPU_MEMORY_SNAPSHOT = False
ENABLE_MODEL_WARMUP = False
EXTENDED_IDLE_TIMEOUT = False
```

**Steps:**
1. Edit `config.py` with the above settings
2. Deploy: `py -m modal deploy modal_app/app.py`
   - **IMPORTANT:** The first deploy will create the snapshot (takes normal time)
3. Wait for complete shutdown (2-3 minutes)
4. Test cold start: `py -m modal run modal_app/app.py test_audio.mp3`
   - This should use the snapshot
5. Record the time

**Expected Result:** ~17-24s (30-50% improvement)

**What This Tests:**
- Container state restoration (Python packages, dependencies loaded)
- Model weights loaded to CPU memory (but not GPU)

**Record Result:**
```
Test 2 - CPU Memory Snapshots
Cold Start Time: _____ seconds
Improvement vs Baseline: _____ %
Notes:
```

---

### Scenario 3: GPU Memory Snapshots (Most Important!)

**Configuration in `config.py`:**
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True   # Required for GPU snapshots
ENABLE_GPU_MEMORY_SNAPSHOT = True   # ← CHANGED
ENABLE_MODEL_WARMUP = False
EXTENDED_IDLE_TIMEOUT = False
```

**Steps:**
1. Edit `config.py` with the above settings
2. Deploy: `py -m modal deploy modal_app/app.py`
   - First deploy creates GPU snapshot (takes normal time)
3. Wait for complete shutdown (2-3 minutes)
4. Test cold start: `py -m modal run modal_app/app.py test_audio.mp3`
5. Record the time

**Expected Result:** ~3-5s (85-90% improvement! 🚀)

**What This Tests:**
- Full GPU state restoration
- Model weights already loaded to GPU memory
- CUDA kernels already initialized

**Record Result:**
```
Test 3 - GPU Memory Snapshots
Cold Start Time: _____ seconds
Improvement vs Baseline: _____ %
Improvement vs CPU-only: _____ %
Notes:
```

---

### Scenario 4: GPU Snapshots + Model Warm-up

**Configuration in `config.py`:**
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True
ENABLE_MODEL_WARMUP = True          # ← CHANGED
EXTENDED_IDLE_TIMEOUT = False
```

**Steps:**
1. Edit `config.py` with the above settings
2. Deploy: `py -m modal deploy modal_app/app.py`
   - Watch logs for "Warming up model with dummy forward pass..."
   - Note: Initial deployment will take longer due to warm-up
3. Wait for complete shutdown (2-3 minutes)
4. Test cold start: `py -m modal run modal_app/app.py test_audio.mp3`
5. Record the time

**Expected Result:** ~2-4s (potential additional 1-2s improvement)

**What This Tests:**
- Compiled CUDA kernels captured in snapshot
- First actual transcription doesn't need to compile kernels

**Record Result:**
```
Test 4 - GPU Snapshots + Warm-up
Cold Start Time: _____ seconds
Improvement vs GPU-only: _____ %
Notes:
```

---

### Scenario 5: Extended Idle Timeout (No Snapshots)

**Configuration in `config.py`:**
```python
ENABLE_CPU_MEMORY_SNAPSHOT = False
ENABLE_GPU_MEMORY_SNAPSHOT = False
ENABLE_MODEL_WARMUP = False
EXTENDED_IDLE_TIMEOUT = True        # ← CHANGED
EXTENDED_IDLE_TIMEOUT_SECONDS = 300  # 5 minutes
```

**Steps:**
1. Edit `config.py` with the above settings
2. Deploy: `py -m modal deploy modal_app/app.py`
3. Run first request: `py -m modal run modal_app/app.py test_audio.mp3`
4. Wait 2 minutes (less than timeout)
5. Run second request immediately
6. Record the time

**Expected Result:** <100ms (container is still warm)

**What This Tests:**
- Container stays alive longer between requests
- Trade-off: costs more in idle time vs fewer cold starts

**Record Result:**
```
Test 5 - Extended Idle Timeout
Warm Container Time: _____ ms
Cost Trade-off Notes:
- Idle cost per hour: ~$0.XX
- Useful if requests are < 5 min apart
```

---

### Scenario 6: COMBINED - GPU Snapshots + Warm-up + Extended Timeout

**Configuration in `config.py`:**
```python
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True
ENABLE_MODEL_WARMUP = True
EXTENDED_IDLE_TIMEOUT = True
EXTENDED_IDLE_TIMEOUT_SECONDS = 600  # 10 minutes
```

**Steps:**
1. Edit `config.py` with the above settings
2. Deploy: `py -m modal deploy modal_app/app.py`
3. Test cold start after complete shutdown
4. Test warm container within 10 minutes

**Expected Result:**
- Cold start: ~2-4s
- Warm container: <100ms

**Record Result:**
```
Test 6 - All Optimizations Combined
Cold Start Time: _____ seconds
Warm Container Time: _____ ms
Total Improvement vs Baseline: _____ %
```

---

## Benchmark Results Template

Copy this and fill in your actual measurements:

```
=== COLD START OPTIMIZATION BENCHMARK RESULTS ===

Test Audio File: ________________
Audio Duration: _______ seconds
Whisper Model: large
GPU Type: L4

Scenario 1 - Baseline (No Optimizations)
  Cold Start: _____ seconds

Scenario 2 - CPU Memory Snapshots
  Cold Start: _____ seconds
  Improvement: _____ %

Scenario 3 - GPU Memory Snapshots
  Cold Start: _____ seconds
  Improvement: _____ %

Scenario 4 - GPU Snapshots + Model Warm-up
  Cold Start: _____ seconds
  Improvement: _____ %

Scenario 5 - Extended Idle Timeout (Warm Container)
  Warm Start: _____ ms

Scenario 6 - All Optimizations Combined
  Cold Start: _____ seconds
  Warm Start: _____ ms
  Total Improvement: _____ %

=== RECOMMENDATION ===

Best configuration for your use case:
[Fill in based on your results and request patterns]

Reason:
[Explain why this configuration makes sense]
```

---

## Important Notes

### Snapshot Creation vs Snapshot Use

- **First deployment after enabling snapshots:** Takes normal time (creating snapshot)
- **Subsequent cold starts:** Use the snapshot (fast)

To see the improvement, you need to:
1. Deploy with snapshots enabled
2. Let containers fully shut down
3. Start a new container (this uses the snapshot)

### When Snapshots are Invalidated

Snapshots are automatically recreated when:
- You redeploy with code changes
- You change config values
- You change the container image

Snapshots are NOT invalidated by:
- Volume (model storage) changes
- New audio files being processed

### GPU Snapshot Limitations (Alpha)

From Modal's documentation:
- GPU snapshots are experimental
- Test thoroughly before production use
- May not work with all CUDA operations
- Check Modal docs for latest limitations

---

## Troubleshooting

### "Snapshot not found" errors
- Wait longer for initial deployment to complete
- Check Modal dashboard for snapshot creation status

### Results not improving as expected
- Verify config changes were saved
- Ensure you redeployed after changing config: `py -m modal deploy modal_app/app.py`
- Check Modal logs for snapshot loading messages

### First request after deployment is still slow
- This is expected - it's creating the snapshot
- Second cold start should be fast

---

## Next Steps

After benchmarking:

1. **Choose your optimal configuration** based on:
   - Request frequency (how often are containers cold?)
   - Cost sensitivity (idle costs vs cold start costs)
   - User experience requirements (acceptable wait time)

2. **Update config.py** with your chosen settings

3. **Deploy to production**:
   ```bash
   py -m modal deploy modal_app/app.py
   ```

4. **Monitor in production**:
   - Track cold start times in Modal dashboard
   - Monitor idle costs
   - Adjust timeout values as needed

---

## Cost Considerations

### GPU Idle Costs (NVIDIA L4)
- ~$0.0003/second of GPU time
- Extended timeout of 600s (10 min) = ~$0.18/hour if idle
- vs. cold start cost of 34s = ~$0.01 per cold start

**Break-even analysis:**
- If you get >18 requests per hour, extended timeout saves money
- If you get <18 requests per hour, accept cold starts

**GPU snapshots:** No additional idle cost - best of both worlds!

---

## Recommended Production Configuration

Based on Modal's documented results and typical usage:

```python
# Production-optimized configuration
ENABLE_CPU_MEMORY_SNAPSHOT = True
ENABLE_GPU_MEMORY_SNAPSHOT = True  # Even though alpha, well-tested
ENABLE_MODEL_WARMUP = True         # Marginal cost, good improvement
EXTENDED_IDLE_TIMEOUT = True       # Reduce cold starts further
EXTENDED_IDLE_TIMEOUT_SECONDS = 300  # 5 minutes (balance cost/UX)
```

**Expected production performance:**
- Cold start: ~3-5 seconds (vs 34.40s)
- Warm container: <100ms
- 85-90% reduction in cold start time

Test this thoroughly in your environment before production deployment!
