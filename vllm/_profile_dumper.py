"""Memory profiling dumper for vLLM server during bench.
Started by OpenVINOPredictor.__init__ via env var PROFILE_MEM=1.
"""
import os
import threading
import time
import tracemalloc
import gc

def _dumper_loop(interval=20, top_n=20, log_path="/tmp/profile_mem.log"):
    log = open(log_path, "w", buffering=1)
    log.write("=== Memory profiling started at %s ===\n" % time.strftime("%H:%M:%S"))
    pid = os.getpid()
    while True:
        try:
            time.sleep(interval)
            ts = time.strftime("%H:%M:%S")
            # /proc/PID/status
            try:
                with open(f"/proc/{pid}/status") as f:
                    s = {}
                    for line in f:
                        if line.startswith(("VmRSS", "RssAnon", "RssShmem", "RssFile")):
                            k, v = line.split(":", 1)
                            s[k] = v.strip()
            except Exception:
                s = {}
            log.write("\n=== %s | RSS=%s anon=%s shmem=%s file=%s ===\n" % (
                ts, s.get("VmRSS", "?"), s.get("RssAnon", "?"),
                s.get("RssShmem", "?"), s.get("RssFile", "?")))

            # tracemalloc top
            if tracemalloc.is_tracing():
                snap = tracemalloc.take_snapshot()
                top = snap.statistics("traceback")[:top_n]
                log.write("--- tracemalloc top %d ---\n" % top_n)
                for i, stat in enumerate(top):
                    log.write("%d. size=%.1fMB count=%d\n" % (
                        i+1, stat.size / 1e6, stat.count))
                    for line in stat.traceback.format()[-3:]:
                        log.write("    " + line + "\n")

            # gc stats
            log.write("--- gc: gen0=%d gen1=%d gen2=%d ---\n" % tuple(
                len(gc.get_objects(g)) for g in range(3)))
        except Exception as e:
            log.write("ERR %s: %s\n" % (time.strftime("%H:%M:%S"), e))

def start():
    if os.environ.get("PROFILE_MEM") != "1":
        return
    # Start tracemalloc with shallow stack (depth=10) — too deep blows up overhead
    tracemalloc.start(10)
    print("[profile_dumper] tracemalloc started, dumping every 20s to /tmp/profile_mem.log")
    t = threading.Thread(target=_dumper_loop, daemon=True, name="profile-dumper")
    t.start()
