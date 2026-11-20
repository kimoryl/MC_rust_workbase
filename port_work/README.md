# Java 整合包移植临时目录

将待分析或待迁移的 Java 整合包（`.mrpack` / CurseForge `.zip` / Prism 实例目录）导入此处（本目录内除本文件外均被 `.gitignore` 忽略）。

示例命令：
```powershell
horizon-port-pack import "<整合包路径>" .\port_work\pack_a
horizon-port-pack inventory .\port_work\pack_a
horizon-port-pack report .\port_work\pack_a
```
