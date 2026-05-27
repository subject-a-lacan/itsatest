#!/usr/bin/env bash
# 查看所有任务状态 - 在服务器项目根目录执行

echo "=== 正在运行的 main.py 进程数 ==="
ps aux | grep "main.py" | grep -v grep | wc -l

echo ""
echo "=== 每个 log 的状态 (done=1已完成, done=0未完成) ==="
for f in logs_grid/*.log; do
  done_flag=$(grep -c '\[DONE\]' "$f" 2>/dev/null)
  last_sr=$(grep 'SR=' "$f" 2>/dev/null | tail -1)
  echo "done=${done_flag} | ${last_sr} | $(basename $f .log)"
done

echo ""
echo "=== 已有 log 文件数 / 54 ==="
ls logs_grid/*.log 2>/dev/null | wc -l
