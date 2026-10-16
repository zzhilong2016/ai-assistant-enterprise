# 把 Anaconda 加入用户 PATH（已存在则跳过）
$anacondaPaths = @('D:\anaconda', 'D:\anaconda\Scripts')
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $userPath) { $userPath = '' }
$entries = $userPath -split ';' | Where-Object { $_ -ne '' }
$changed = $false
foreach ($p in $anacondaPaths) {
    if ($entries -notcontains $p) {
        $userPath = if ($userPath -eq '') { $p } else { "$userPath;$p" }
        $changed = $true
        Write-Output "已添加: $p"
    } else {
        Write-Output "已存在，跳过: $p"
    }
}
if ($changed) {
    [Environment]::SetEnvironmentVariable('Path', $userPath, 'User')
    Write-Output 'PATH 更新完成'
} else {
    Write-Output 'PATH 无需修改'
}
