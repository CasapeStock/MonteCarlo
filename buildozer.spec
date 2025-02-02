[app]
title = Avaliando Ações
package.name = avaliacaoAcoes
package.domain = org.manaia
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 0.1
requirements = python3,kivy==2.2.1,yfinance==0.2.33,numpy==1.24.3,pandas==2.0.1,matplotlib==3.7.1
orientation = portrait
osx.python_version = 3
osx.kivy_version = 2.2.1
fullscreen = 0
android.permissions = INTERNET
android.api = 30
android.minapi = 24
android.sdk = 30
android.ndk = 23b
p4a.branch = master
log_level = 2
android.arch = arm64-v8a
android.release_artifact = aab
android.enable_androidx = True

[buildozer]
log_level = 2
warn_on_root = 1
