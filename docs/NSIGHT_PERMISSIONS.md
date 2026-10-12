# ERR_NVGPUCTRPERM on Linux

`ncu` can be on `PATH` and still fail with:

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0.
```

That is a driver permission check. It is not a missing metric name and not a missing `ncu` binary. The third A40 package hit this error (`paper/data/a40/nsight_error.log`, `paper/data/a40/nsight_skipped.txt`). `achieved_ai` stayed empty. `torch.profiler` memory is not a substitute.

NVIDIA's note: <https://developer.nvidia.com/ERR_NVGPUCTRPERM>

## Fix (administrator)

Profiling is restricted to admin users unless the NVIDIA kernel module is loaded with `NVreg_RestrictProfilingToAdminUsers=0`.

**Persistent.** As root, write `/etc/modprobe.d/nvidia-profiling.conf`:

```text
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

Then reboot. Unloading `nvidia` while the GPU is in use usually fails, so a reboot is the reliable step.

**Current boot only**, when the module is not already loaded (or you can unload it):

```bash
modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

If `modprobe` reports the module is in use, use the `/etc/modprobe.d/` file and reboot.

## Verify, then re-run

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
ncu --version
```

`RmProfilingAdminOnly` should be `0`. From this repo:

```bash
TRY_NSIGHT=1 bash scripts/run_a40.sh
```

A successful capture writes `results/a40/nsight_dram.json`. If permission is still denied, `nsight_error.log` contains `ERR_NVGPUCTRPERM` again and `achieved_ai` stays empty. Do not fill it from `proxy_ai`.

## 中文

`ncu` 已经在 `PATH` 上，仍可能报 `ERR_NVGPUCTRPERM`：当前用户无权读取 NVIDIA GPU 性能计数器。这不是指标名错误，也不是没安装 `ncu`。第三次 A40 包就是这个错误（`paper/data/a40/nsight_error.log`、`paper/data/a40/nsight_skipped.txt`）。`achieved_ai` 仍为空。`torch.profiler` 的显存不能代替 DRAM 字节。

说明：<https://developer.nvidia.com/ERR_NVGPUCTRPERM>

管理员把内核模块参数 `NVreg_RestrictProfilingToAdminUsers` 设为 0。

持久做法：以 root 写入 `/etc/modprobe.d/nvidia-profiling.conf`：

```text
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

然后重启。GPU 正被占用时卸载 `nvidia` 通常会失败，所以重启是可靠步骤。

模块尚未加载、或可以卸载时，也可以只对本次启动执行：

```bash
modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

若 `modprobe` 提示模块正在使用，改用上面的 `/etc/modprobe.d/` 文件并重启。

检查后再跑：

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
ncu --version
```

`RmProfilingAdminOnly` 应为 `0`。然后在本仓库：

```bash
TRY_NSIGHT=1 bash scripts/run_a40.sh
```

成功时写出 `results/a40/nsight_dram.json`。若权限仍被拒绝，`nsight_error.log` 里会再次出现 `ERR_NVGPUCTRPERM`，`achieved_ai` 保持为空。不要用 `proxy_ai` 填。
