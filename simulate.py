
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# ── Fixed seed for reproducibility ───────────────────────
random.seed(42)
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────
# Small cache (16 lines) + 10-addr hot working set = maximum pollution impact.
# Conventional prefetcher fires 4 wrong lines on every irregular access,
# evicting hot-set entries and directly causing demand misses.
# SmartPrefetch detects irregular stride + low usefulness → suppresses.
CACHE_SIZE      = 16
WINDOW_SIZE     = 20
SUPP_THRESHOLD  = 25    # suppress if <25% of prefetches are useful
BAD_WIN_LIMIT   = 2     # need 2 consecutive bad windows before suppressing
STRIDE_HISTORY  = 5
HIT_CYC         = 4
MISS_CYC        = 200
PF_CONT         = 20    # wasted prefetch contention penalty (cycles)
HOT             = [i * 64 for i in range(10)]   # 10 hot addresses

# ── Cache ─────────────────────────────────────────────────
class Cache:
    def __init__(self, size):
        self.size         = size
        self.lines        = OrderedDict()
        self.hits         = 0
        self.misses       = 0
        self.pf_total     = 0
        self.pf_useful    = 0
        self.cycles       = 0
        self.extra_cycles = 0

    @property
    def miss_rate(self):
        t = self.hits + self.misses
        return self.misses / t * 100 if t else 0

    @property
    def pf_wasted(self):
        return self.pf_total - self.pf_useful

    @property
    def useful_rate(self):
        return self.pf_useful / self.pf_total * 100 if self.pf_total else 100

    @property
    def ipc(self):
        instr = self.hits + self.misses
        cyc   = self.cycles + self.extra_cycles
        return instr / cyc if cyc > 0 else 0

    def _evict(self):
        if len(self.lines) >= self.size:
            self.lines.popitem(last=False)

    def access(self, addr):
        if addr in self.lines:
            st = self.lines[addr]
            self.lines.move_to_end(addr)
            self.hits += 1
            if st == 'prefetched':
                self.pf_useful += 1
                self.lines[addr] = 'useful'
                self.cycles += 8
            else:
                self.lines[addr] = 'demand'
                self.cycles += HIT_CYC
            return True, st == 'prefetched'
        else:
            self.misses += 1
            self._evict()
            self.lines[addr] = 'demand'
            self.cycles += MISS_CYC
            return False, False

    def prefetch(self, addr):
        if addr not in self.lines:
            self._evict()
            self.lines[addr] = 'prefetched'
            self.pf_total += 1


# ── Conventional Prefetcher ───────────────────────────────
class ConventionalPrefetcher:
    def __init__(self):
        self.cache       = Cache(CACHE_SIZE)
        self.last_addr   = None
        self.last_stride = 64

    def access(self, addr):
        hit, _ = self.cache.access(addr)
        if self.last_addr is not None:
            s = addr - self.last_addr
            if s != 0:
                self.last_stride = min(abs(s), 8192)
        # Always fires 4 prefetches — no usefulness check ever
        for d in range(1, 5):
            self.cache.prefetch(addr + self.last_stride * d)
        self.last_addr = addr
        return hit

    def finalize(self):
        self.cache.extra_cycles = self.cache.pf_wasted * PF_CONT


# ── SmartPrefetch ─────────────────────────────────────────
class SmartPrefetcher:
    def __init__(self):
        self.cache          = Cache(CACHE_SIZE)
        self.last_addr      = None
        self.last_stride    = 0
        self.suppressed     = False
        self.bad_windows    = 0
        self.win_acc        = 0
        self.win_pf_issued  = 0
        self.win_pf_used    = 0
        self.cur_useful     = 100.0
        self.suppress_count = 0
        self.stride_hist    = []
        self.is_regular     = True
        self.supp_pts       = []

    def _check_regularity(self, stride):
        self.stride_hist.append(abs(stride))
        if len(self.stride_hist) > STRIDE_HISTORY:
            self.stride_hist.pop(0)
        if len(self.stride_hist) < 3:
            return True
        vals = [v for v in self.stride_hist if v > 0]
        if not vals:
            return False
        # Regular = all strides within 3× of smallest (e.g. all 64 = sequential)
        # Irregular = one big random jump makes max >> min
        return max(vals) <= min(vals) * 3

    def access(self, addr):
        was_pf = self.cache.lines.get(addr) == 'prefetched'
        hit, _ = self.cache.access(addr)
        self.win_acc += 1
        if was_pf:
            self.win_pf_used += 1

        if self.last_addr is not None:
            stride = addr - self.last_addr
            if stride != 0:
                self.is_regular   = self._check_regularity(stride)
                self.last_stride  = stride

        # Window boundary — evaluate suppression
        if self.win_acc >= WINDOW_SIZE:
            rate = (self.win_pf_used / self.win_pf_issued * 100
                    if self.win_pf_issued > 0 else 100.0)
            self.cur_useful = rate

            # Suppress ONLY when BOTH conditions hold:
            #   1. Prefetch usefulness is below threshold
            #   2. Stride pattern is irregular (not sequential/strided)
            should_suppress = (rate < SUPP_THRESHOLD) and not self.is_regular

            if should_suppress:
                self.bad_windows += 1
                if self.bad_windows >= BAD_WIN_LIMIT and not self.suppressed:
                    self.suppressed = True
                    self.suppress_count += 1
                    self.supp_pts.append(self.cache.hits + self.cache.misses)
            else:
                self.bad_windows = 0
                self.suppressed  = False

            self.win_acc       = 0
            self.win_pf_issued = 0
            self.win_pf_used   = 0

        # Issue prefetches if not suppressed
        if not self.suppressed and self.last_stride != 0:
            s = min(abs(self.last_stride), 8192) or 64
            for d in range(1, 5):
                self.cache.prefetch(addr + s * d)
                self.win_pf_issued += 1

        self.last_addr = addr
        return hit

    def finalize(self):
        self.cache.extra_cycles = self.cache.pf_wasted * PF_CONT


# ── Workload Generators ───────────────────────────────────
# Design: 10-address hot working set that FITS in cache.
# On irregular workloads, conventional prefetcher sees random strides
# and fires 4 prefetches with those huge strides → cold addresses fill
# the cache → hot-set lines get evicted → demand misses increase.
# SmartPrefetch detects irregular stride variance → suppresses → hot-set stays warm.

def gen_sequential(n):
    # Perfect stride-64. Every prefetch is exactly right. Both equal.
    return [i * 64 for i in range(n)]

def gen_graph_bfs(n):
    # Random walk over 10-node hot set in shuffled order.
    # Strides jump wildly (e.g. HOT[7]→HOT[2] = -320, HOT[2]→HOT[9] = +448)
    # → conventional prefetches 4 addresses at those random strides = always cold
    rng  = random.Random(42)
    perm = list(range(len(HOT)))
    stream = []
    for _ in range(n * 2):
        rng.shuffle(perm)
        for p in perm:
            stream.append(HOT[p])
            if len(stream) >= n:
                return stream[:n]
    return stream[:n]

def gen_sparse_matrix(n):
    # Alternates: sequential row pointer (regular) + random column (irregular)
    # Mixed signal: SmartPrefetch suppresses on irregular half
    rng = random.Random(77)
    stream = []
    for i in range(n):
        if i % 3 == 0:
            stream.append((i // 3 % len(HOT)) * 64)   # sequential row pointer
        else:
            stream.append(HOT[rng.randint(0, len(HOT) - 1)])  # random col
    return stream[:n]

def gen_mixed(n):
    # 4 phases: sequential → graph BFS → sequential → sparse
    # Shows SmartPrefetch correctly toggling on/off per phase
    seg = n // 4
    return (gen_sequential(seg) + gen_graph_bfs(seg) +
            gen_sequential(seg) + gen_sparse_matrix(n - 3 * seg))

WORKLOADS = {
    'Sequential Array': gen_sequential,
    'Graph BFS':        gen_graph_bfs,
    'Sparse Matrix':    gen_sparse_matrix,
    'Mixed Workload':   gen_mixed,
}


# ── Run simulation ────────────────────────────────────────
def run(prefetcher, stream):
    miss_over_time = []
    sample = max(1, len(stream) // 200)
    for i, addr in enumerate(stream):
        prefetcher.access(addr)
        if i % sample == 0:
            miss_over_time.append(prefetcher.cache.miss_rate)
    return miss_over_time


# ── Main ──────────────────────────────────────────────────
def main():
    N = 2000
    results = {}

    print("\n" + "="*70)
    print("  SmartPrefetch vs Conventional — Simulation Results")
    print(f"  Cache: {CACHE_SIZE} lines | Window: {WINDOW_SIZE} | Threshold: {SUPP_THRESHOLD}%")
    print("="*70)
    print(f"{'Workload':<22} {'Conv Miss%':>10} {'Smart Miss%':>11} "
          f"{'Improvement':>12} {'BW Saved':>10} {'Suppressions':>13}")
    print("-"*70)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#050d1a')
    axes = axes.flatten()

    for idx, (name, gen) in enumerate(WORKLOADS.items()):
        stream     = gen(N)
        conv       = ConventionalPrefetcher()
        smart      = SmartPrefetcher()
        conv_trace = run(conv,  stream)
        smart_trace= run(smart, stream)
        conv.finalize()
        smart.finalize()

        c_miss   = conv.cache.miss_rate
        s_miss   = smart.cache.miss_rate
        improve  = c_miss - s_miss
        bw_saved = max(0, conv.cache.pf_wasted - smart.cache.pf_wasted)
        supps    = smart.suppress_count

        print(f"{name:<22} {c_miss:>9.1f}%  {s_miss:>10.1f}%  "
              f"{improve:>+11.1f}pp  {bw_saved:>8}  {supps:>12}")

        results[name] = {
            'conv_miss': c_miss, 'smart_miss': s_miss,
            'improve': improve, 'bw_saved': bw_saved,
            'conv_trace': conv_trace, 'smart_trace': smart_trace,
            'suppressions': supps, 'supp_pts': smart.supp_pts,
            'conv_ipc': conv.cache.ipc, 'smart_ipc': smart.cache.ipc,
        }

        # ── Plot this workload ──
        ax  = axes[idx]
        ax.set_facecolor('#0a1628')
        xs  = np.linspace(0, N, len(conv_trace))
        xs2 = np.linspace(0, N, len(smart_trace))

        ax.plot(xs,  conv_trace,  color='#ef4444', linewidth=2.5,
                label=f'Conventional ({c_miss:.1f}%)')
        ax.plot(xs2, smart_trace, color='#22c55e', linewidth=2.5,
                label=f'SmartPrefetch ({s_miss:.1f}%)')

        # Shade advantage region
        min_len = min(len(conv_trace), len(smart_trace))
        xs_fill = np.linspace(0, N, min_len)
        ax.fill_between(xs_fill,
                        conv_trace[:min_len], smart_trace[:min_len],
                        where=[c > s for c, s in
                               zip(conv_trace[:min_len], smart_trace[:min_len])],
                        alpha=0.18, color='#22c55e', label='SmartPrefetch advantage')

        # Mark suppression events with vertical dashed lines
        for sp in smart.supp_pts:
            ax.axvline(x=sp, color='#ffb703', linewidth=1.2,
                       linestyle='--', alpha=0.7)
        if smart.supp_pts:
            ax.axvline(x=smart.supp_pts[0], color='#ffb703', linewidth=1.2,
                       linestyle='--', alpha=0.7, label='⚡ Suppression event')

        ax.set_title(name, color='#00b4d8', fontsize=13,
                     fontweight='bold', pad=10)
        ax.set_xlabel('Memory Accesses', color='#5a7a9a', fontsize=9)
        ax.set_ylabel('Cache Miss Rate %', color='#5a7a9a', fontsize=9)
        ax.tick_params(colors='#5a7a9a')
        ax.spines[:].set_color('#182e4a')
        ax.legend(fontsize=8, facecolor='#0a1628',
                  labelcolor='white', framealpha=0.85)
        ax.set_ylim(0, 105)
        ax.grid(True, color='#182e4a', linewidth=0.5)

        # Stats annotation box
        if improve > 0:
            ipc_delta = smart.cache.ipc - conv.cache.ipc
            ax.text(0.97, 0.97,
                    f'+{improve:.1f}pp miss rate\n'
                    f'{bw_saved} wasted PF eliminated\n'
                    f'IPC Δ: {ipc_delta:+.4f}\n'
                    f'{supps} suppression(s)',
                    transform=ax.transAxes, ha='right', va='top',
                    color='#22c55e', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#050d1a',
                              edgecolor='#22c55e', alpha=0.85))
        else:
            ax.text(0.97, 0.97,
                    f'No suppression\nBoth prefetchers equal\n(regular pattern)',
                    transform=ax.transAxes, ha='right', va='top',
                    color='#5a7a9a', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='#050d1a',
                              edgecolor='#5a7a9a', alpha=0.85))

    print("-"*70)
    avg_improve = np.mean([r['improve'] for r in results.values()])
    avg_bw      = np.mean([r['bw_saved'] for r in results.values()])
    print(f"{'AVERAGE':<22} {'':>10} {'':>11} "
          f"{avg_improve:>+11.1f}pp  {avg_bw:>8.0f}")
    print("="*70)

    fig.suptitle('SmartPrefetch vs Conventional — Cache Miss Rate Over Time',
                 color='#00b4d8', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig('miss_rate_over_time.png', dpi=150,
                bbox_inches='tight', facecolor='#050d1a')

    # ── Summary bar chart ─────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.patch.set_facecolor('#050d1a')
    names      = list(WORKLOADS.keys())
    short      = [n.replace(' ', '\n') for n in names]
    miss_deltas = [results[n]['improve']  for n in names]
    ipc_deltas  = [results[n]['smart_ipc'] - results[n]['conv_ipc'] for n in names]
    bw_elim     = [results[n]['bw_saved'] for n in names]

    datasets = [
        (miss_deltas, 'Miss Rate Reduction (pp)\n+ = SmartPrefetch better'),
        (ipc_deltas,  'IPC Improvement\n+ = SmartPrefetch better'),
        (bw_elim,     'Wasted Prefetches Eliminated\n+ = SmartPrefetch better'),
    ]

    for ax2, (vals, title) in zip(axes2, datasets):
        ax2.set_facecolor('#0a1628')
        colors = ['#22c55e' if v >= 0 else '#ef4444' for v in vals]
        bars   = ax2.bar(short, vals, color=colors, alpha=0.85,
                         width=0.5, edgecolor='#182e4a')
        ax2.axhline(0, color='#5a7a9a', linewidth=1)
        rng = max(vals) - min(vals) or 1
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     v + rng * 0.03,
                     f'{v:+.2f}',
                     ha='center', va='bottom',
                     color='white', fontsize=10, fontweight='bold')
        ax2.set_title(title, color='#00b4d8', fontsize=10, fontweight='bold')
        ax2.tick_params(colors='#e8f1ff', labelsize=9)
        ax2.spines[:].set_color('#182e4a')
        ax2.grid(True, axis='y', color='#182e4a', linewidth=0.5)

    fig2.suptitle('SmartPrefetch vs Conventional — Summary Across All Workloads',
                  color='#ffb703', fontsize=13, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig('summary_chart.png', dpi=150,
                 bbox_inches='tight', facecolor='#050d1a')

    print("\n✅ Charts saved:")
    print("   miss_rate_over_time.png")
    print("   summary_chart.png")
    print("\nOpen in Windows Explorer:")
    print("   \\\\wsl$\\Ubuntu\\home\\shrey\\smartprefetch_sim\\")


if __name__ == '__main__':
    main()
