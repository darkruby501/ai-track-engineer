# Store an Anthropic API key for the coach as a per-user environment variable.
# The key is typed (pasted) at a masked prompt; it is never written to a file.
$ErrorActionPreference = "Stop"
$current = [Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")
if ($current) { Write-Host "A key is already set (ends in ...$($current.Substring($current.Length - 4)))." }
$secure = Read-Host "Paste your Anthropic API key and press Enter" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim() }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
if (-not $key.StartsWith("sk-ant-")) { Write-Host "That doesn't look like an Anthropic key; nothing saved."; exit 1 }
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", $key, "User")
Write-Host "Saved for user $env:USERNAME (ends in ...$($key.Substring($key.Length - 4)))."
Write-Host "Restart the AI Coach window so it picks the key up."
