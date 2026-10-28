<#
Create a dedicated Windows/Python 3.11 environment with the verified legacy
OpenMMLab stack. No global package installation or automatic version migration.
Examples:
  .\scripts\setup_environment.ps1 -Python 'C:\Python311\python.exe'
  .\scripts\setup_environment.ps1 -Python '.\.venv-srd\Scripts\python.exe' -CheckOnly
#>
[CmdletBinding()]
param(
    [string]$Python = 'python',
    [string]$EnvironmentDirectory = '.venv-srd',
    [switch]$CheckOnly,
    [switch]$SkipCudaProbe
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed (exit $LASTEXITCODE). Installation/check stopped without selecting alternative versions."
    }
}

$platformCheck = @'
import struct, sys
assert sys.platform == 'win32', 'This installer targets Windows only; no Linux/macOS fallback is selected.'
assert struct.calcsize('P') == 8, '64-bit Python is required.'
assert sys.version_info[:2] == (3, 11), 'CPython 3.11 is required by the verified cp311 wheel.'
print('Platform accepted:', sys.version.split()[0], sys.platform, 'x64')
'@
Invoke-CheckedPython -Executable $Python -Arguments @('-c', $platformCheck)

if ($CheckOnly) {
    $targetPython = $Python
} else {
    if ([System.IO.Path]::IsPathRooted($EnvironmentDirectory)) {
        $environmentPath = [System.IO.Path]::GetFullPath($EnvironmentDirectory)
    } else {
        $repositoryRoot = Split-Path -Parent $PSScriptRoot
        $environmentPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $EnvironmentDirectory))
    }
    $targetPython = Join-Path $environmentPath 'Scripts\python.exe'
    if (Test-Path -LiteralPath $environmentPath) {
        if (-not (Test-Path -LiteralPath (Join-Path $environmentPath 'pyvenv.cfg'))) {
            throw 'The destination already exists and is not a Python venv. Choose a new dedicated directory.'
        }
    } else {
        Write-Host "Creating dedicated environment: $environmentPath"
        Invoke-CheckedPython -Executable $Python -Arguments @('-m', 'venv', $environmentPath)
    }
    Invoke-CheckedPython -Executable $targetPython -Arguments @('-c', $platformCheck)
    Write-Host 'Installing the explicitly pinned toolchain into the dedicated environment only.'
    Invoke-CheckedPython -Executable $targetPython -Arguments @('-m', 'pip', 'install', 'pip==23.3', 'setuptools==68.0.0', 'wheel==0.41.2')
    Invoke-CheckedPython -Executable $targetPython -Arguments @('-m', 'pip', 'install', 'torch==2.1.1', 'torchvision==0.16.1', '--index-url', 'https://download.pytorch.org/whl/cu118')
    $repositoryRoot = Split-Path -Parent $PSScriptRoot
    Invoke-CheckedPython -Executable $targetPython -Arguments @('-m', 'pip', 'install', '-r', (Join-Path $repositoryRoot 'requirements.txt'))
    Invoke-CheckedPython -Executable $targetPython -Arguments @('-m', 'pip', 'check')
}

$runtimeCheck = @'
import importlib.metadata as metadata
import sys
import numpy as np
import torch
import torchvision
import mmcv
import mmdet
import mmrotate
import e2cnn
from mmcv.ops import box_iou_rotated, nms_rotated

assert torch.__version__.split('+')[0] == '2.1.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.16.1', torchvision.__version__
assert torch.version.cuda == '11.8', ('Expected CUDA 11.8 build', torch.version.cuda)
required = {'mmcv-full':'1.7.2', 'mmdet':'2.28.2', 'mmrotate':'0.3.4', 'e2cnn':'0.2.3', 'numpy':'1.26.4'}
for name, version in required.items():
    assert metadata.version(name) == version, (name, metadata.version(name), version)

devices = ['cpu']
if sys.argv[1] == 'run-cuda':
    assert torch.cuda.is_available(), 'The pinned CUDA build loaded, but no usable NVIDIA CUDA device is available.'
    devices.append('cuda')
for device in devices:
    boxes = torch.tensor([[20., 20., 10., 6., .2]], device=device)
    result = box_iou_rotated(boxes, boxes, clockwise=True)
    assert torch.allclose(result, torch.ones_like(result), atol=1e-5), result
    _, keep = nms_rotated(boxes, torch.ones(1, device=device), .1)
    assert keep.tolist() == [0]
    print('Rotated IoU/NMS passed:', device)
print('Pinned stack accepted:', torch.__version__, 'CUDA', torch.version.cuda, required)
if len(devices) == 1:
    print('CUDA operator execution was explicitly skipped; this is a CPU-only operator check.')
'@
$probeMode = if ($SkipCudaProbe) { 'skip-cuda' } else { 'run-cuda' }
Invoke-CheckedPython -Executable $targetPython -Arguments @('-c', $runtimeCheck, $probeMode)
Write-Host 'The environment check completed. It does not certify model training, dataset accuracy, or the final notebook.'
Write-Host "Notebook launcher: & '$targetPython' -m jupyterlab"
