import platform
import os
import shutil
import tempfile
import trimesh

_search_path = os.environ.get("PATH", "")
if platform.system() == "Windows":
    # try to find Blender install on Windows
    # split existing path by delimiter
    _search_path = [i for i in _search_path.split(";") if len(i) > 0]
    for pf in [r"C:\Program Files", r"C:\Program Files (x86)"]:
        pf = os.path.join(pf, "Blender Foundation")
        if os.path.exists(pf):
            for p in os.listdir(pf):
                if "Blender" in p:
                    _search_path.append(os.path.join(pf, p))
    _search_path = ";".join(_search_path)
    # print("searching for blender in: %s", _search_path)

if platform.system() == "Darwin":
    # try to find Blender on Mac OSX
    _search_path = [i for i in _search_path.split(":") if len(i) > 0]
    _search_path.append("/Applications/blender.app/Contents/MacOS")
    _search_path.append("/Applications/Blender.app/Contents/MacOS")
    _search_path.append("/Applications/Blender/blender.app/Contents/MacOS")
    _search_path = ":".join(_search_path)
    print("searching for blender in: %s", _search_path)

_blender_executable = shutil.which("blender", path=_search_path)
_clip_script_path = os.path.join(os.path.dirname(__file__), "blender_diff.py")
_intersect_script_path = os.path.join(os.path.dirname(__file__), "blender_intersect.py")
exists = _blender_executable is not None
if exists:
    _blender_executable = f'"{_blender_executable}"'


def clip_model(model: trimesh.Trimesh, clip: trimesh.Trimesh):
    with tempfile.TemporaryDirectory() as tmpdir:
        if isinstance(model, trimesh.Trimesh):
            model.export(os.path.join(tmpdir, "model.obj"))
            model_path = os.path.join(tmpdir, "model.obj")
        else:
            model_path = model
        if isinstance(clip, trimesh.Trimesh):
            clip.export(os.path.join(tmpdir, "clip.obj"))
            clip_path = os.path.join(tmpdir, "clip.obj")
        else:
            clip_path = clip
        out_path = os.path.join(tmpdir, "out.glb")

        args = [
            _blender_executable,
            "--background",
            "--python",
            _intersect_script_path,
            "DIFFERENCE",
            model_path,
            clip_path,
            out_path,
        ]
        os.system("  ".join(args))

        result = trimesh.load_mesh(out_path)

    return result


def intersect_model(model: trimesh.Trimesh, clip: trimesh.Trimesh):
    with tempfile.TemporaryDirectory() as tmpdir:
        # if True:
        # tmpdir = "E:/Data/上海项目/0419万达/Production_obj_1/building"
        if isinstance(model, trimesh.Trimesh):
            model.export(os.path.join(tmpdir, "model.obj"))
            model_path = os.path.join(tmpdir, "model.obj")
        else:
            model_path = model
        if isinstance(clip, trimesh.Trimesh):
            clip.export(os.path.join(tmpdir, "clip.obj"))
            clip_path = os.path.join(tmpdir, "clip.obj")
        else:
            clip_path = clip
        out_path = os.path.join(tmpdir, "out.glb")
        # model.export(model_path)
        # clip.export(clip_path)

        args = [
            _blender_executable,
            "--background",
            "--python",
            _intersect_script_path,
            "INTERSECT",
            model_path,
            clip_path,
            out_path,
        ]
        os.system(" ".join(args))
        result = trimesh.load_mesh(out_path)

    return result
