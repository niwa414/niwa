# 3D Regression Checklist (B2 Milestone)

## 文档依赖（必须一致）

- 3D driver spec：`3d_driver_spec.md`
- B2 回归条款：`b2_gate_regression.md`（所有 gate/阈值以此为准，禁止在 checklist 中重写数值）

---

# Stage 3D-0 — Smoke（短跑、只验证链路正确）

## 3D-0A：GS→3D 初始化映射 smoke

**目的**：证明 2D GS/HDF5 → 3D runtime mapping 变量、坐标、矢量投影正确，不产生 NaN/爆炸。

**必跑项**

- 3D 小网格、短 `tlim`（只需 1–2 个输出）
- 初始化来自 GS/HDF5（2D）并在 3D 中生成 `(rho, p, v, B)` 场

**必须通过（Pass/Fail）**

- [ ] 运行不崩溃，输出 1–2 帧 VTK/diagnostics
- [ ] `rho`、`p` 在域内无 NaN/Inf
- [ ] `|B|` 量级合理（与 2D 初值一致的数量级），中心/端部形状合理
- [ ] 轴附近 `r→0` 无异常尖峰（检查 B、v 在轴附近是否出现条纹/爆点）
- [ ] 粗略 ∇·B 诊断不“爆掉”（允许非零，但必须稳定、无发散趋势；CT 一致性后续再看）

**产物**

- 日志（run.log）
- 1–2 帧 VTK
- （可选）`outputs/analysis/init_profile_axis.csv`：轴线上 Bz/Br/Bphi 或 Cartesian B 分量剖面

---

## 3D-0B：EMF 注入 smoke（只验证投影与 ramp 进度对齐）

**目的**：证明 3D Cartesian 下 `Eφ → (Ex,Ey)` 的投影与 CT 更新路径正确，外场 ramp 可控。

**必跑项**

- 关闭 moving-wall（或固定 wall），只打开 EMF 外场
- 使用已验证的 waveform（可用 smooth ramp，但短跑即可）

**必须通过**

- [ ] `Bext_frac_waveform`（脚本/日志）随时间单调增加且与预期进度一致
- [ ] B 场变化方向与外场驱动一致（端部梯度出现）
- [ ] 守恒量不出现明显异常跳变（允许少量波动）

**产物**

- `outputs/analysis/kirtley_scaling*.json/csv`（至少能计算 progress 与 Bext 相关字段）
- smoke 日志

---

## 3D-0C：moving-wall smoke（几何缩容链路正确）

**目的**：证明 3D moving-wall 机制能缩域，且 moving-volume 统计（mass_mv, Etot_mv, vol_mv）正确输出。

**必跑项**

- 开启 moving-wall、关闭 EMF（或 EMF 保持常量），短跑
- 输出 HST 中的 moving-volume 统计

**必须通过**

- [ ] `vol_mv` 随时间下降（方向正确）
- [ ] `mass_mv` 漂移小（作为 smoke：绝对值不做硬阈值，但必须“明显优于 inflow piston”）
- [ ] `Etot_mv` 随 wall 做功上升/变化是连续的，不出现数值爆炸

**产物**

- HST（含 mass_mv/Etot_mv/vol_mv）
- run.log

---

# Stage 3D-1 — Gate60（中等成本，先验证“核心压缩”出现）

**目的**：3D 下在 progress≈0.60 时，`mass_core` 掩膜稳定且核心体积收缩（Vdrop<0），不追指数。

**必跑项**

- 打开：GS→3D 初始化 + EMF + moving-wall（按 spec）
- 跑到 progress≈0.60（不要求 dense cadence）

**必须通过**

- [ ] `tools/analyze_kirtley_scaling.py` 能在 3D 输出上计算 `mass_core`（0.30/0.20）且 `mask_empty_total=0`
- [ ] `Vdrop_window`（建议 0.10–0.60）为负，且 `rho_avg_core` 上升（两档一致趋势）
- [ ] `mask_fraction` 处于合理“核心尺度”（不接近 1，也不接近 0；允许比 2D 有偏移）

**产物**

- 两档 mass_core 的 json/csv（0.30/0.20）
- 简短 Gate60 摘要（3 行：Vdrop、rho_avg、mask_fraction range）

---

# Stage 3D-2 — Gate80 + Fit（dense，执行冻结回归条款）

**目的**：在 3D 下执行 `b2_gate_regression.md` 的完整回归门槛（含拟合段、R²、指数与一致性）。

**必跑项**

- 跑到 progress≥0.80
- dense cadence（确保拟合段 `n_fit ≥ 25` 有机会满足）

**必须通过（严格引用回归条款）**

- [ ] 运行满足 `b2_gate_regression.md` 的全部 pass thresholds（primary 0.30）
- [ ] control 0.20 与 0.30 的一致性满足回归条款
- [ ] 全窗 sanity 计算完成；breathing 若存在，按条款“非阻塞”记录即可

**产物**

- `*.fit.json`（0.30/0.20）
- `full_window_summary.json`（或写进同一个 json）
- 回归判定一行结论：PASS/FAIL + 原因（仅引用条款字段）

---

## 自动化建议（可选，但强烈推荐）

- 把 Stage 3D-0/1/2 变成三个脚本入口（或 Make target）：
  - `run_3d0_smoke.sh`
  - `run_3d1_gate60.sh`
  - `run_3d2_gate80_dense.sh`
- 每个脚本末尾调用 `tools/analyze_kirtley_scaling.py` 并输出 `PASS/FAIL` 的 machine-readable JSON。
