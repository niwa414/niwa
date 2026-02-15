# Helion-style Design Trade Study

## 1) 输入基线（metrics来源）
- m27-d1 full requirements: `/Users/ni/Desktop/fusion/outputs/m27-d1-helion-full-requirements-gate/analysis/metrics.json`
- m27-d2 usage scenarios: `/Users/ni/Desktop/fusion/outputs/m27-d2-helion-usage-scenarios-gate/analysis/metrics.json`
- m26-p2 circuit-load integration: `/Users/ni/Desktop/fusion/outputs/m26-p2-circuit-load-integration/analysis/metrics.json`
- m26-d2 magnetic-load interface: `/Users/ni/Desktop/fusion/outputs/m26-d2-magnetic-load-interface/analysis/metrics.json`
- m26-b2 tilt dual-regime: `/Users/ni/Desktop/fusion/outputs/m26-b2-tilt-dual-regime-gate/analysis/metrics.json`
- m26-b3 transport closure: `/Users/ni/Desktop/fusion/outputs/m26-b3-transport-closure-gate/analysis/metrics.json`
- m26-ml1 surrogate: `/Users/ni/Desktop/fusion/outputs/m26-ml1-transport-surrogate/analysis/metrics.json`
- m28-d1 internal parity: `/Users/ni/Desktop/fusion/outputs/m28-d1-helion-internal-parity-gate/analysis/metrics.json`

## 2) 参数窗口（Design Window）

| 设计参数 | Min | Nom | Max | 敏感度判断 | metrics绑定 |
| --- | --- | --- | --- | --- | --- |
| 触发窗口 `t_fire` (ns) | 11.24 | 11.82 | 12.40 | 高（受seed影响） | `m27-d1.recommended_window_start_ns`, `m27-d1.recommended_window_end_ns`, `m27-d1.seed_effect_rel` |
| 建议时序偏移 `Δt` (ns) | 1.16 | 1.16 | 1.16 | 中 | `m27-d1.recommended_timing_shift_ns` |
| 脉宽 `τpulse` (ns) | 2.565 | 2.565 | 2.565 | 低 | `m27-d1.pulse_width_s` |
| 载荷力 `F` (N) | 4.898 | 4.947 | 5.042 | 中 | `m26-d2.force_proxy_min_N`, `m26-d2.force_proxy_mean_N`, `m26-d2.force_proxy_peak_N` |
| 磁压 `p_mag` (Pa) | 1.569 | 1.599 | 1.600 | 中 | `m26-d2.p_mag_mean_Pa`, `m26-d2.p_mag_peak_Pa` |
| 感应电压 `V_ind` (V) | 200 | 69,350 | 69,350 | 中 | `m26-d2.dphi_dt_min_V`, `m26-d2.dphi_dt_peak_V`, `m26-p2.vind_peak_V` |
| 倾斜模态增长/阻尼 `γ` (1/s) | -3.577e8 | -9.227e7 | +2.321e8 | 高（双稳态） | `m26-b2.damping_gamma_on`, `m26-b2.damping_gamma_off`, `m26-b2.growth_gamma_fit` |
| 传输闭环可观测差异 | 0.000 | 0.094 | 0.187 | 高 | `m26-b3.observable_rel_diff` |
| 代理模型 `u2` 误差 | 0.020 | 0.039 | 0.075 | 中-高（样本少） | `m26-ml1.oos_u2_rel_err_mean`, `m26-ml1.oos_u2_rel_err_max`, `m26-ml1.u2_loocv_rel_err_max` |

## 3) 敏感度结论（按影响排序）

| 排名 | 因子 | 量化证据 | 影响 |
| --- | --- | --- | --- |
| S1 | seed扰动 | `seed_effect_rel=0.08475`, `seed_drift_span_rel=0.08850` | 触发窗口与倾斜增长率对seed最敏感，先做seed标定再扩窗。 |
| S2 | 传输闭环差异 | `observable_rel_diff=0.18749` | 运输闭环仍有约18.7%量级差异，影响跨shot外推。 |
| S3 | 代理泛化 | `training_samples=10`, `oos_samples=2`, `oos_u2_rel_err_max=0.03937` | 现阶段可用于工程预筛，但不足以单独做最终签核。 |
| S4 | benchmark对齐 | `benchmark_alignment_r2=0.93634` | 一致性较好但非“高保真封顶”；仍需私有数据绑定。 |
| S5 | 动态drift | `drift_effect_rel=8.68e-7` | 对当前窗口影响可忽略，可降级优先级。 |

## 4) 风险矩阵

| 风险ID | 风险描述 | 严重度 | 证据(metrics) | 建议控制 |
| --- | --- | --- | --- | --- |
| R1 | internal parity 尚不可声明，阻断“全量可采购” | 高 | `m28-d1.internal_parity_claimable=false`, `internal_only_gap_count=3` | 先闭环3个internal-only gap，再放行延后条目。 |
| R2 | GPU runtime 证据缺失导致算力/周期不可承诺 | 高 | `gpu_runtime_proven=false`, `runtime_amrex_gpu_backends=["NONE"]`, `runtime_warpx_compute_modes=["OMP"]` | 建立GPU构建+运行日志证据链并回写manifest。 |
| R3 | 传输模型在未知工况可能偏移 | 中-高 | `observable_rel_diff=0.18749`, `u2_loocv_rel_err_max=0.07495` | 增加OOS样本，先限制采购到“硬件包络”而非“控制算法封闭环”。 |
| R4 | 倾斜模态双稳态对时序窗口有耦合风险 | 中-高 | `growth_gamma_fit=2.321e8`, `damping_gamma_on=-3.577e8` | 先做m=1频段诊断扩展，再放开触发容差。 |
| R5 | 触发窗口受seed影响，可能出现窗口漂移 | 中 | `seed_effect_rel=0.08475`, `recommended_window_start/end_ns` | 采购触发系统时预留可编程延时和抖动预算。 |

## 5) 推荐时序（执行顺序）

1. **T0（立即）**：按当前硬载荷包络启动“可直接采购”条目（高压、力学、基础诊断）。
   - 放行条件：`m26-p2.integrated_gate_pass=true`, `m26-d2.interface_ready=true`, `m27-d2.engineering_interface_outputs_ok=true`。
2. **T1（先补证据）**：完成 GPU runtime proof。
   - 关卡：`gpu_runtime_proven=true` 且 `runtime_gpu_build_backend_detected=true`。
3. **T2（先补实验）**：绑定 private shot dataset 并复算场景一致性。
   - 关卡：`private_shot_dataset_bound=true`，且 `benchmark_alignment_r2` 不低于当前 `0.93634`。
4. **T3（先补模型）**：绑定 private hardware model，打通控制与硬件参数映射。
   - 关卡：`private_hardware_model_bound=true`，并维持 `p20_circuit_r2/p22_circuit_r2≈1.0`。
5. **T4（签核）**：internal parity claimable 后，放行“需补实验/补模型”采购条目。

## 6) Internal-only gap 清单与闭环动作（单列）

| gap | 当前状态 | 闭环动作 | 验收标准(metrics) |
| --- | --- | --- | --- |
| GPU runtime proof | `gpu_runtime_proven=false` | 以GPU后端重建并运行最小闭环case；保留runtime日志与device映射，回写internal manifest。 | `gpu_runtime_proven=true`, `runtime_gpu_build_backend_detected=true`, `runtime_amrex_gpu_backends` 含非`NONE` |
| private shot dataset | `private_shot_dataset_bound=false` | 将私有shot数据与合成诊断 (`synthetic_bdot_points=33`, `synthetic_interferometer_points=129`) 做字段级绑定并做重建回放。 | `private_shot_dataset_bound=true` |
| private hardware model | `private_hardware_model_bound=false` | 建立私有硬件参数模型（线圈/回路/结构）并对齐 `V_ind/F/p_mag` 主接口。 | `private_hardware_model_bound=true`, `internal_parity_claimable=true` |

