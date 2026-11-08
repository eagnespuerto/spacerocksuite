# PyInstaller spec: one-folder Windows build (fast start; the installer wraps it).
# Build: pyinstaller packaging/windows/spacerocksuite.spec --noconfirm
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, "..", ".."))
datas = collect_data_files("spacerocksuite") + collect_data_files("astropy") + collect_data_files("matplotlib")
hidden = collect_submodules("spacerocksuite") + ["srs_core", "erfa"]

a = Analysis(
    [os.path.join(ROOT, "packaging", "windows", "launch.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hidden,
    excludes=["pytest", "IPython", "notebook", "PyQt5", "PySide6", "scipy"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SpaceRockSuite",
          icon=os.path.join(ROOT, "packaging", "icons", "spacerocksuite.ico"),
          console=False, upx=False)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="srs",
          icon=os.path.join(ROOT, "packaging", "icons", "spacerocksuite.ico"),
          console=True, upx=False)
coll = COLLECT(exe, cli, a.binaries, a.datas, name="SpaceRockSuite", upx=False)
