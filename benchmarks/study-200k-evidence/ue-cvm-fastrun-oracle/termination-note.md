# Intentional bounded stop

The explicit-CVM FAST_RUN oracle was terminated through its owning supervisor on 2026-10-14 at approximately 04:42:00 UTC; all scoped Python/compiler children were gone by 04:42:11. Charged wrapper time: 1361.810505867s (22m41.8s), exit -15. It had no completed compile event, oracle pair, or result.

The first cold C++ compile still had not returned. A second isolated cold primary compile no longer had a defensible completion window before the immutable scientific cutoff; the parent requested reporting corrections without new runs. The queue's automatic primary was disabled and no FAST_RUN primary was launched. This preserves a negative compile-cost observation, not a numerical/correctness or warm-throughput claim. No limits, budget or deadline were reset.

The earlier named-FAST_RUN Numba OOM is separate evidence; see `../ue-fastrun-oracle/systemd-journal.log`. This attempt used an explicit CVM/fast-run Mode object, as shown in its frozen source.
