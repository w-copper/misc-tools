import argparse
import shutil
import os


def copy_mtl_images(source, target):
    # Copy the mtl file
    shutil.copyfile(source.replace(".obj", ".mtl"), target.replace(".obj", ".mtl"))

    # Copy the images
    with open(source.replace(".obj", ".mtl"), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("map_Kd "):
                image_path = line.split()[1]
                shutil.copyfile(
                    os.path.join(os.path.dirname(source), image_path),
                    os.path.join(os.path.dirname(target), image_path),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="source.obj")
    parser.add_argument("--target", type=str, default="target.obj")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    out_path = (
        args.out if args.out is not None else args.target.replace(".obj", "_copy.obj")
    )
    copy_mtl_images(args.source, out_path)
    with open(out_path, "w") as out:
        out.write(f'mtllib {out_path.replace(".obj", ".mtl")}\n')
        with open(args.source, "r") as f1:
            with open(args.target, "r") as f2:
                for line in f2:
                    if line.startswith("v") and not line.startswith("vt"):
                        out.write(line)

            for line in f1:
                if line.startswith("vt"):
                    out.write(line)
                elif line.startswith("f"):
                    out.write(line)
                elif line.startswith("mtllib"):
                    pass
                elif line.startswith("usemtl"):
                    out.write(line)
