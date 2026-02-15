# WarpX Smoke Tests (RZ)

快速检查 WarpX 外场驱动链路是否稳定，可在仓库根目录运行：

1. **外场无粒子（字段 I/O 验证）**
   ```bash
   python warpx-driver/warpx_driver.py --mode field-only --max-steps 5 --dt 1e-9 --diag-period 5
   ```
   作用：加载 openPMD B 场（若缺失自动生成均匀 Bz），不生成粒子，可查 `outputs/warpx/diag*` 中的 B 输出。

2. **均匀冷等离子 + 常量 Bz**
   ```bash
   python warpx-driver/warpx_driver.py --mode const-b-plasma --const-B 0.05 --max-steps 20 --dt 1e-9 --diag-period 5
   ```
   作用：在 RZ 网格上放一团均匀电子等离子，常量外加 Bz（默认 0.05 T），验证粒子边界处理是否稳定。

3. **等离子 + openPMD B 文件**
   ```bash
   python warpx-driver/warpx_driver.py --mode bfile-plasma --b-file warpx-driver/B_ext.h5 --max-steps 20 --dt 1e-9 --diag-period 5
   ```
   作用：加载 openPMD B 场并驱动等离子，默认使用 `warpx-driver/B_ext.h5`，若文件缺失会自动生成一致网格的均匀 Bz。

4. **原始 LCR 波形驱动（完整链路）**
   ```bash
   python warpx-driver/warpx_driver.py --mode full-driver --waveform outputs/analysis/lcr_waveform.csv --b-file warpx-driver/B_ext_from_frc.h5 --max-steps 20
   ```
   作用：保留原始动态外场缩放逻辑，使用 CSV 中的 B(t) 插值。

5. **从 Athena 快照生成流体态并播种粒子（fluid-init）**
   ```bash
   # 先从 VTK 导出流体态和 B 场（默认 fold x<0→r>=0，允许重采样到目标网格）
   PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
     python warpx-driver/export_fluid_to_opmd.py --input-vtk outputs/mhd/frc_long_full/frc_merge.block0.out1.00020.vtk --nr 64 --nz 64 --r-min 0 --r-max 8 --z-min -4 --z-max 4 --output-fluid warpx-driver/fluid_init.h5
   PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
     python warpx-driver/export_b_opmd_from_vtk.py --mode from-vtk --input-vtk outputs/mhd/frc_long_full/frc_merge.block0.out1.00020.vtk --nr 64 --nz 64 --r-min 0 --r-max 8 --z-min -4 --z-max 4 --output-bfile warpx-driver/B_ext_from_frc.h5
   # 然后以流体态播种粒子运行短步长（需匹配网格分辨率与几何）
   PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
     python warpx-driver/warpx_driver.py --mode fluid-init --fluid-file warpx-driver/fluid_init.h5 --b-file warpx-driver/B_ext_from_frc.h5 --max-steps 20 --diag-period 5 --ppc 2 --nr 64 --nz 64 --r-max 8 --z-max 8
   ```
   作用：按 VTK 网格播种离子+电子宏粒子，默认零速度（可用 `--use-fluid-velocity` 开启流体速度），验证从 MHD 态到 WarpX 的最小闭环。

6. **m=1 倾斜烟囱（RZ-IM）**
   ```bash
   PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
     python warpx-driver/warpx_driver.py --mode const-b-plasma --solver yee --n-azimuthal-modes 2 \
     --const-B 0.05 --tilt-eps 0.05 --ppc 4 --max-steps 20 --diag-period 5 \
     --monitor-interval 5 --drop-threshold 100 --run-tag tilt_smoke_m1
   ```
   作用：加载常量 Bz，合成倾斜扰动（m=1 权重调制），`n_azimuthal_modes=2` 输出 m0/m1 模能量，检查掉落计数/元数据。若编译了 PSATD，可将 `--solver yee` 改为 `--solver psatd`。

7. **Athena Hall FRC → WarpX fluid-init 一键链路**
   ```bash
   PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
     python warpx-driver/hall_frc_pipeline.py \
       --athena-bin athena-24.0/bin/athena \
       --athena-input athena-24.0/inputs/mhd/athinput.hall_frc_init \
       --vtk-pattern "outputs/mhd/hall_frc_init/hall_frc_init.block0.out1.*.vtk" \
       --nr 64 --nz 64 --r-max 2 --z-max 2 --ppc 2 --max-steps 10 --diag-period 5 \
       --run-tag hall_frc_pipeline --validate
   ```
   作用：可选运行 Athena++ 生成 VTK，自动挑最新 VTK 导出 fluid/B openPMD，并用 WarpX fluid-init 短跑（支持 `--tilt-eps` 时自动加 n_modes=2），最后可选运行元数据校验。

输出位置：WarpX 默认写入 `outputs/warpx/diag*`（带 `--run-tag` 时为 `outputs/warpx/diag_<run-tag>`，也可用 `--metadata-dir/--diag-dir` 指定）。如需更长步数或更改网格，可用 `--max-steps --nr --nz --r-max --z-max` 覆盖默认值。若 `openpmd_api` 不可用，自动生成 B 文件会失败，此时可改用 `--mode const-b-plasma` 先跑通。

## 多 basis 外场（bias + mirror）示例

用于 Belova B2 / IPA A2 这类“端部镜场压缩 + 非均匀 Bz(z,t)”的最小链路验证。

1) 生成两个外场基底（单位幅值，后续用系数叠加）：
```bash
PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
  python warpx-driver/export_b_opmd_from_vtk.py --mode uniform \
  --output-bfile warpx-driver/B_basis_bias.h5 \
  --nr 64 --nz 128 --r-min 0 --r-max 0.25 --z-min -0.25 --z-max 0.25 --Bz-const 1.0

PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
  python warpx-driver/export_b_opmd_from_vtk.py --mode mirror \
  --output-bfile warpx-driver/B_basis_mirror.h5 \
  --nr 64 --nz 128 --r-min 0 --r-max 0.25 --z-min -0.25 --z-max 0.25 \
  --Bz-center 1.0 --mirror-ratio 1.5 --mirror-center-z 0.0 --mirror-half-length 0.25
```

提示：如果你希望“镜比随时间上升，但中心 bias 不变”，推荐用 `--mode mirror-delta` 生成 `B_mirror - B_uniform`，再用系数 `c(t)` 去乘它：
`B_total = B_bias + c(t) * (B_mirror - B_bias)`。

2) 准备一个系数波形（CSV，至少含 `t` 和 `coeff` 两列；支持用 `path:col` 指定列名）：
```bash
cat > scenes/basis_coeff_example.csv <<'CSV'
t,coeff
0.0,0.0
1e-6,1.0
CSV
```

3) 运行（field-only 只验证外场注入；`--induced-E` 会注入与 Aθ 一致的 Eθ 以减少法拉第不一致）：
```bash
PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
  python warpx-driver/warpx_driver.py --mode field-only \
  --nr 64 --nz 128 --r-max 0.25 --z-max 0.5 --max-steps 10 --diag-period 5 \
  --external-basis warpx-driver/B_basis_bias.h5 \
  --external-basis-waveform scenes/basis_coeff_example.csv:coeff \
  --external-basis warpx-driver/B_basis_mirror.h5 \
  --external-basis-const 0.2 \
  --induced-E --run-tag basis_smoke
```

说明：
- 当 `n_azimuthal_modes=1` 时，driver 会优先通过加载第一个 basis 文件来分配 WarpX 的 external-field buffer，再叠加各 basis；这不会强行覆盖自洽场。
- 当 `n_azimuthal_modes>1` 时，WarpX 端 external-field buffer 可能无法分配/加载（取决于构建与版本限制）；此时 driver 会回退到直接写总场（会覆盖自洽场），仅建议用于“外场注入/数值稳定”层面的冒烟验证。

### Scene 一键命令（Belova B2）

`warpx-driver/run_scene.py` 会读取 `scenes/*.yaml`，自动导出流体态，并在场景提供 `fields.external_basis[*].generator` 时自动生成缺失的 basis 文件：
```bash
PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
  python warpx-driver/run_scene.py \
    --scene scenes/belova_b2_compression.yaml \
    --vtk outputs/mhd/belova_b1_merge/belova_b1_merge.block0.out1.00000.vtk \
    --run-tag belova_b2_smoke \
    --run
```
如需强制重建 basis 文件，可加 `--regenerate-basis`。

## 标准化诊断/分析

WarpX RZ diags（thetaMode）：
```bash
python warpx-driver/analyze_fusion_metrics.py --diag-path outputs/warpx/<run> --output-prefix outputs/analysis/fusion_metrics
```
会生成 `outputs/analysis/fusion_metrics.csv` / `.json` / `.png`，包含 centroid、r_s、Psi_max、双核分离、X 点 Et、Hall 四极代理等时间序列。

## GS 平衡初值（B1/B3/C1/C2）

生成一个线性 p(ψ) 的 GS 平衡，并直接输出 openPMD `fluid` + `B`：
```bash
PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
  python warpx-driver/gs_frc_equilibrium.py \
    --nr 128 --nz 256 --r-max 0.25 --z-min -0.25 --z-max 0.25 \
    --b-bias 0.05 --beta-s 0.15 --Ti-eV 100 --Te-Ti 1.0 \
    --output-b warpx-driver/B_ext_gs_frc.h5 \
    --output-fluid warpx-driver/fluid_init_gs_frc.h5
```

然后用 WarpX fluid-init 起跑：
```bash
PYTHONPATH=pic-warpx-25.11/build-rz/lib/site-packages \
  python warpx-driver/warpx_driver.py --mode fluid-init \
    --fluid-file warpx-driver/fluid_init_gs_frc.h5 \
    --b-file warpx-driver/B_ext_gs_frc.h5 \
    --nr 128 --nz 256 --r-max 0.25 --z-max 0.5 --ppc 2 --max-steps 20
```

Athena++ VTK 系列（frc_merge 等 2D cartesian）：
```bash
python tools/analyze_fusion_metrics_athena.py outputs/mhd/<run> --vtk-pattern "frc_merge.block0.out1.*.vtk" \
  --output-prefix outputs/analysis/<run>/fusion_metrics
```
默认按 x1→z、x2→r 解读，并输出同名指标的 CSV/JSON/PNG，便于与 WarpX 对比。
