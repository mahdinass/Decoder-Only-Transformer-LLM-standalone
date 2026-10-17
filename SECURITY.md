# Security

Normal checkpoint loading explicitly uses PyTorch's restricted `weights_only=True`
loader and requires PyTorch >=2.10. Keep PyTorch current with upstream security
fixes. Restricted loading reduces exposure; external checkpoints can still be
malformed, oversized or designed to exhaust memory. Load only checkpoints from
sources you trust and use resource isolation for unfamiliar files.

`convert-legacy --trust-pickle` intentionally permits general pickle loading.
It can execute code. Use it only for your own trusted historical artifacts; normal
loading never falls back to that path. No remote model downloads are automatic.

Secrets review is local. `scripts/audit_history.py` scans all available objects
and project files without printing matched values. A maintained scanner should
also review all intended release refs. A clean local scan cannot clear unavailable
remote logs or artifacts.

If you find a credential, revoke or rotate it first and report the location without
copying the value into an issue, log or pull request. Use a private channel with the
maintainer for vulnerability details; do not create a public issue with exploit
material or private data. No public security-reporting endpoint is configured yet.
