下面是我建议的**后续开发总规划**（按“对标闭环贡献最大 + 返工最少 + 证据可交付”排序）。你按这个节奏推进，每完成一个包就能在 Evidence Pack 里把“对标可信度”往上抬一档。

---

## 总原则（后续所有开发都遵守）

1. **一次只改一个变量**：要么改 case，要么改分析，要么改 C++；不要混改。
2. **所有结论必须能回放**：每个包都有明确 DoD，且写入 `outputs/.../analysis/metrics.json` + `PASSFAIL.json` + `evidence/index.md`。
3. **先修“判定口径”再修“物理能力”**：能用同一条 run 解决的，不重跑；必须动 C++ 才补的，再动。

---

# P0（立刻做，确保 B2.2 ppc 扫描闭环）：分析口径稳健化

### P0.1 修复 ppc-high 的 r2 FAIL（不重跑、只改分析）

**目标**：同一条 `ppc-high` 输出上，把拟合做成“只拟合线性指数段”，而不是混窗导致 r2 低。

**改动点（Python 分析脚本）**

* 增加“滑窗找最佳指数段”的拟合：在候选窗口里对 `log(tilt_amp)` 线性回归，选 **r2 最大**的窗口作为 `r2_fit/gamma_fit`。
* 输出审计字段：`fit_window_start/end`、`fit_window_len`、`r2_fit_best`、`gamma_fit_best`、`residual_std`。

**DoD**

* `r2_fit_best >= 0.9` 且 `fit_points >= 25`（你现在 fit_points=30 足够）
* `PASSFAIL.json` 由 FAIL → PASS（在不改 case、不重跑的前提下）

> 这一包做完，B2.2 就具备“ppc 扫描可交付证据”的判定可信度。

---

# P1（把 B2.2 做成 FULL）：完成 ppc=1/2/4 的可比证据

### P1.1 ppc=4（high）证据闭环

**目标**：ppc=2（mid）PASS 已有；把 ppc=4（high）用 P0 的分析修成 PASS，并固化同口径字段。

**DoD**

* `done=true`、`archive_size_gb`、`wall_time_s`、`gamma_fit`、`r2_fit`、`fit_points`、`tilt_amp_ratio`
* 三点（ppc=1/2/4）齐全：可以在 evidence 里给出 **性能/拟合稳定性**的对比图（哪怕是表格也行）。

---

# P2（对标“最硬核”的物理口径）：B2.1 tilt on/off 做实

### P2.1 “严格 tilt-off”审计闭环（优先不动 C++）

**目标**：off 必须“严格对称”，并且在指标里能证明。

**改动点（case/driver + analysis）**

* off 案例关闭所有非对称来源：dynamic drift / group-centroid axis update / 任何闭环更新（只针对 off）。
* 固定 seed、固定采样顺序/分组初始化。
* 增加审计字段：`tilt_seed_enabled`、`dynamic_drift_enabled`、`initial_asymmetry_metric`。

**DoD**

* 理想：`gamma_on/gamma_off >= 2`
* 若不分离：必须在 metrics 里给出可复核归因（initial_asymmetry 非零、或噪声主导、或某闭环未关干净）。

### P2.2 触发条件：如果 P2.1 仍无法分离 → 进入 C++ 硬补

**触发**：严格 off 后仍不分离且无法归因。
**下一包直接做 P3（m=1 真谱）**，不要再换 seed 赌。

---

# P3（大改代码包 1）：WarpX 侧 m=1/多模能力与 in-situ 谱诊断

### P3.1 m=1 指标先行（最小可用 C++ patch）

**目标**：即使暂时不能完全多模推进，也先把“m=1 可测”做出来，替换代理 tilt。

**改动点（WarpX C++/diagnostics）**

* 输出 `mode_energy_m1(t)` / `tilt_m1_amp(t)` 到轻量 metrics（避免爆盘）。
* 让 Hybrid 路径也能访问该诊断（至少读出 m=1 的投影量）。

**DoD**

* B2.1 on/off 的差异体现在 `m=1` 指标上（off 显著更弱或不增长）
* 用 `m=1` 的线性段拟合得到 `gamma_m1`，并进入 gate 判定

### P3.2 真正多模/外场谱（若必须）

在 P3.1 基础上再评估是否需要：`n_modes>1` 的场更新/注入通路（这是更大工程量的一步）。

---

# P4（大改代码包 2）：Athena++ 侧 EB/blocked face 守恒修复（A4.2 从 partial → full）

### P4.1 blocked faces 严格 no-through-mass

**改动点（Athena++ EB flux）**

* blocked face 的质量通量严格置零或严格按 area fraction 裁剪，并保证守恒口径一致。
* 写入 `blocked_enforcement_mode` 等审计字段。

**DoD**

* `leak_mass_frac_max_geom <= 1e-4`
* `mass_budget_residual_geom_rel <= 1e-4`
* 不再 dt collapse

---

# P5（从“数值连通”到“物理连通”）：A3 闭环响应验证 + 电路模型升级路线

### P5.1 先把闭环响应做成 FULL（不急着 PFN）

**目标**：证明“可观测量→控制量→响应”闭环成立（先选一条你最能稳定定义的链，例如 radius 或 position）。

**DoD**

* 给出一组可复核响应曲线：扰动输入 → observable 变化 → 控制器输出变化 → 系统回正/跟踪
* 重启/重复跑结果一致（误差在你定义的容许范围内）

### P5.2 再升级电路复杂度（PFN/非线性开关）

只有当 P5.1 FULL 后再做，否则会把问题复杂度叠加到不可调试。

---

# P6（“最后一公里”工程能力）：合成诊断 MVP（低成本、高价值）

### P6.1 合成诊断 MVP（先做 2–3 个）

推荐先做：

* **B-dot 探针电压**（由局部/环路磁通变化推导）
* **干涉仪相移**（沿 chord 的密度积分）
* （可选）**中子产额 proxy**（若你已有反应率模型接口）

**DoD**

* 每个算例产出“可直接对齐示波器波形”的 `signal(t)`（写入 analysis/metrics 或单独 JSON）
* 成本可控：不新增大体积 diag（尽量 in-situ reducer）

---

# P7（工程鲁棒性）：合并/压缩压力测试与回归门禁

### P7.1 合并鲁棒性 stress test（A2/B5）

* 双团块不对称（强弱、到达时间偏差、轻微几何偏心）
* 形成一组小 sweep（每组 3–5 点即可）

**DoD**

* 明确“可稳定运行的参数域边界”，并写入 evidence（这是工程对标最关键的“可用范围”）

### P7.2 回归门禁固化（CI 化但不必上 CI）

* 做一个 `tools/run_regression_pack.py`：一键跑核心 case + 汇总 PASS/FAIL + 资源指标

---

# P8（工业化生产力）：HPC 扩容与 I/O/重启一致性

### P8.1 重启一致性（WarpX/Athena++）

**DoD**

* checkpoint/restart 前后：核心指标（gamma、守恒、合成诊断）一致

### P8.2 多节点扩容（性能证据）

**DoD**

* 固定物理设置，给出强/弱缩放数据点（哪怕 2–3 个点也足够支撑对外叙事）

---

## 我给你的“下一步拍板”（从你当前状态出发）

1. **先做 P0.1：修 ppc-high 的拟合窗口/审计字段，让 FAIL → PASS（不重跑）**
2. 然后 **P1：B2.2 ppc sweep FULL**
3. 再做 **P2：B2.1 严格 off 审计**
4. 如果仍不分离，直接进入 **P3（m=1 真谱/诊断）**
5. 并行的第二条主线是 **P4（A4.2 堵漏）**

---

如果你同意，我下一条就按“拍板式”给你一个**P0.1 的具体改动清单**：需要改哪个分析脚本、加哪些字段、PASSFAIL 用哪个新字段判定，以及如何确保不影响 ppc-mid 已通过的口径。你只要回我：你现在用的是哪个脚本产出 `r2_fit`（文件名/路径）。
