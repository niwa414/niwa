# Procurement-ready Spec (Electrical / Structural / Diagnostics)

## A. 可直接采购（基于公开栈已闭环指标）

| Spec ID | 类别 | 采购规格 | Min | Nom | Max | 安全系数 | metrics绑定 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E-01 | 电气 | 线圈-回路瞬态耐压等级 | 69.35 kV | 69.35 kV | 90.16 kV | 1.30 | `m26-d2.dphi_dt_peak_V`, `m26-p2.vind_peak_V` | 可直接采购 |
| E-02 | 电气 | 单脉冲输入能量处理能力 | 0.660 mJ | 1.171 mJ | 1.464 mJ | 1.25 | `m26-d2.e_load_J`, `m26-d2.e_in_J` | 可直接采购 |
| E-03 | 电气 | 回收支路单脉冲能量处理能力 | 0.511 mJ | 0.511 mJ | 0.639 mJ | 1.25 | `m26-d2.e_recaptured_J`, `m26-d2.eta_recaptured` | 可直接采购 |
| E-04 | 电气 | 磁通摆幅接口能力 | 9.101e-5 Wb | 9.101e-5 Wb | 1.092e-4 Wb | 1.20 | `m26-d2.phi_delta_Wb` | 可直接采购 |
| S-01 | 结构 | 径向载荷额定能力 | 4.898 N | 4.947 N | 7.564 N | 1.50 | `m26-d2.force_proxy_min_N`, `m26-d2.force_proxy_mean_N`, `m26-d2.force_proxy_peak_N` | 可直接采购 |
| S-02 | 结构 | 磁压承载额定能力 | 1.569 Pa | 1.600 Pa | 2.399 Pa | 1.50 | `m26-d2.p_mag_mean_Pa`, `m26-d2.p_mag_peak_Pa` | 可直接采购 |
| S-03 | 结构 | 载荷变化率承载能力 | 1.106e8 N/s | 1.106e8 N/s | 1.438e8 N/s | 1.30 | `m26-d2.dforce_dt_peak_N_per_s` | 可直接采购 |
| S-04 | 结构 | 冲量承载能力 | 1.266e-8 N·s | 1.266e-8 N·s | 1.899e-8 N·s | 1.50 | `m26-d2.impulse_proxy_Ns` | 可直接采购 |
| D-01 | 诊断 | B-dot 通道数 | 33 | 33 | 40 | 1.21 | `m27-d2.synthetic_bdot_points` | 可直接采购 |
| D-02 | 诊断 | 干涉仪通道数 | 129 | 129 | 155 | 1.20 | `m27-d2.synthetic_interferometer_points` | 可直接采购 |
| D-03 | 诊断 | 磁探针量程 | 1.986 mT | 2.005 mT | 2.406 mT | 1.20 | `m26-d2.bn_avg_mean_T`, `m26-d2.bn_avg_peak_T` | 可直接采购 |

## B. 需先补实验/补模型后采购

| Spec ID | 类别 | 采购规格 | Min | Nom | Max | 安全系数 | metrics绑定 | 阻塞原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E-05 | 电气 | 触发延时可编程范围 | 11.24 ns | 11.82 ns | 13.64 ns | 1.10 | `m27-d1.recommended_window_start_ns`, `m27-d1.recommended_window_end_ns`, `m27-d1.seed_effect_rel` | 需先做seed-时序实测绑定（窗口对seed敏感） |
| E-06 | 电气 | 驱动响应比可调范围 | 1.20 | 1.30 | 1.40 | 1.08 | `m27-d1.driver_response_ratio` | 需 private hardware model 绑定后再定控制器满量程 |
| S-05 | 结构 | 质心漂移容差 | 0.050 | 0.065 | 0.080 | 1.23 | `m27-d1.centroid_shift_abs`, `m26-b2.seed_effect_rel` | 需 private shot + hardware 联合校准 |
| D-04 | 诊断 | m=1 模态带宽 | 57 MHz | 80 MHz | 114 MHz | 2.00 | `m26-b2.growth_gamma_fit`, `m26-b2.damping_gamma_on` | 需补m=1实测验证（双稳态） |
| D-05 | 诊断 | 线圈波形采样率 | 50 GHz | 50 GHz | 62.5 GHz | 1.25 | `m26-d2.coil_series_len`, `m26-d2.coil_time_end_s` | 需 GPU runtime proof + private shot 数据确认真实时间分辨率需求 |

## C. Internal-only gap 清单及闭环动作（单列）

| gap | 当前指标 | 闭环动作 | 闭环后解锁 |
| --- | --- | --- | --- |
| GPU runtime proof | `m28-d1.gpu_runtime_proven=false`, `runtime_amrex_gpu_backends=["NONE"]`, `runtime_warpx_compute_modes=["OMP"]` | 启用GPU后端构建并产出runtime日志证据，更新internal manifest。 | D-05（采样率）及大算例算力相关交付承诺 |
| private shot dataset | `m28-d1.private_shot_dataset_bound=false` | 私有shot数据与诊断通道字段绑定，完成重建回放并记录哈希。 | E-05、S-05、D-04 |
| private hardware model | `m28-d1.private_hardware_model_bound=false` | 建立私有硬件参数模型，校准 `V_ind/F/p_mag` 映射并回写manifest。 | E-06、S-05，最终 `internal_parity_claimable=true` |

## D. 采购执行备注
- 当前可直接发单范围：A表全部 11 项。
- 延后采购范围：B表 5 项，必须在C表三项gap闭环后发单。
- 发单前复核门限建议：`m26-p2.energy_residual_rel <= 1e-6`（当前 `2.27e-7`），`m27-d2.p20_circuit_r2/p22_circuit_r2 >= 0.999`（当前满足）。

