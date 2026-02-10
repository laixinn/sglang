# sgl-kernel 编译指南

> 环境：CUDA 12.6 / SM90 (Hopper) / Python 3.10
>
> 由于 CUDA 12.6 的 `cicc` 编译器在编译 Flash Attention 3 大模板文件时存在间歇性 Segmentation fault，
> 需要使用 `ninja -j1 -k 0` 单线程编译并循环重试。

---

## 步骤 1：开始 / 恢复编译

> ninja 会自动跳过已编译成功的 `.o` 文件，支持断点续编。
> 循环最多重试 5 轮，处理 cicc 间歇性 segfault。

```bash
cd /home/hadoop-djst-algoplat/sglang/sgl-kernel/build && \
export http_proxy=http://10.229.18.27:8412 && \
export https_proxy=http://10.229.18.27:8412 && \
ulimit -s unlimited && \
for i in $(seq 1 10); do \
  echo "=== ninja retry $i ===" && \
  ninja -j1 -k 0 2>&1 | tee /tmp/ninja_resume_${i}.log; \
  FAILS=$(grep -c "FAILED" /tmp/ninja_resume_${i}.log); \
  echo "Failures: $FAILS"; \
  if [ "$FAILS" -eq 0 ]; then echo "ALL COMPILED!"; break; fi; \
  echo "Retrying failed files..."; \
done
```

如果想放后台运行，不占终端：

```bash
nohup bash -c '上面的命令' > /tmp/ninja_bg.log 2>&1 &
```

---

## 步骤 2：查看编译进度

```bash
LOG=$(ls -t /tmp/ninja_resume_*.log 2>/dev/null | head -1); \
[ -z "$LOG" ] && LOG=/tmp/ninja_full_retry.log; \
echo "日志: $LOG" && \
echo "已编译: $(grep -c '^\[' $LOG) 步" && \
echo "失败数: $(grep -c 'FAILED' $LOG)" && \
echo "最新进度:" && grep '^\[' $LOG | tail -3 && \
echo "---" && \
echo "编译进程数: $(ps aux | grep -E 'ninja|nvcc|cicc' | grep -v grep | wc -l)"
```

**关键指标：**
- `[X/Y]` → X 是当前第几个，Y 是本轮总数
- `编译进程数` → 大于 0 表示还在编译，等于 0 表示本轮已结束

---

## 步骤 3：检查是否编译完成

```bash
cd /home/hadoop-djst-algoplat/sglang/sgl-kernel/build && \
ninja -j1 -n 2>&1 | head -5
```

**判断标准：**
- 输出 `ninja: no work to do.` → ✅ 编译完成
- 输出要编译的文件列表或 error → ❌ 还有文件没编好，重新执行步骤 1

---

## 步骤 4：安装 sgl-kernel

编译全部完成后执行：

```bash
cd /home/hadoop-djst-algoplat/sglang/sgl-kernel && \
pip install -e . --no-build-isolation 2>&1 | tail -5
```

---

## 步骤 5：验证安装

```bash
python3 -c "from sgl_kernel import flash_mla_with_kvcache; print('sgl-kernel OK')"
```

输出 `sgl-kernel OK` 即安装成功。

---

## 常见问题

### Q: 编译中途断电 / 关机了怎么办？
A: 直接从步骤 1 开始即可，ninja 会自动跳过已成功编译的文件。

### Q: 某个文件反复 segfault 编不过怎么办？
A: 多重试几轮（增加 `seq 1 10`）。如果某个文件始终失败，可能需要升级 CUDA 到 12.8+。

### Q: 编译成功但 import 报错？
A: 确认执行了步骤 4 的 `pip install -e . --no-build-isolation`。

### Q: CMakeLists.txt 修改说明
当前已注释掉 `common_ops_sm100_build` 目标（SM100/Blackwell 架构），因为本机为 SM90。
如果后续需要恢复，取消 CMakeLists.txt 中相关注释即可。
