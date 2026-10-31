# How to Run Track G (xv6 Lottery Scheduler)

## Build and boot

```bash
cd tracks/track_g_xv6/xv6-public
make clean && make qemu-nox CPUS=1
```

## Inside xv6

Run the correctness tests:
```
lotterytest
```
Expected output ends with `lotterytest: ALL PASSED`.

Run the benchmark (copy the printed lines for your report):
```
benchrun
```
Lines beginning with `RESULT,` are machine-readable data.

Run the standard regression tests under round-robin (policy 0):
```
usertests
```
Expected output ends with `ALL TESTS PASSED`.

Switch to lottery and run usertests again:
```
setpolicy 1
usertests
```
Expected output ends with `ALL TESTS PASSED`.

Quit QEMU:
```
Ctrl-a  x
```

## Produce the diff

From the repository root (outside QEMU):
```bash
bash tracks/track_g_xv6/scripts/make_patch.sh
```

The patch is written to `tracks/track_g_xv6/patches/lottery_scheduler.diff`.

## Troubleshooting

- If a script fails with `\r: command not found`, the file has Windows line endings. Fix it with:
  ```bash
  sed -i 's/\r$//' tracks/track_g_xv6/scripts/make_patch.sh
  ```
- Run with a single CPU (`CPUS=1`) to avoid multi-CPU scheduler races during tests.
- If the build fails with `-Werror` related errors on a newer GCC, confirm that `-Werror` has been removed from `CFLAGS` in the Makefile.

