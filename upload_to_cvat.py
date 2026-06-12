#!/usr/bin/env python3
"""
Sube el dataset Fishnet (generado por make_yolo_dataset.py) a un CVAT local.

Por cada split (train/val/test):
  1. Crea un task en el proyecto (las imágenes se leen del file share del
     servidor, montado hacia esta carpeta — no se suben por HTTP).
  2. Sube las anotaciones en formato "Ultralytics YOLO Detection 1.0".

Requiere el venv con cvat-sdk:
    .venv-cvat/bin/python upload_to_cvat.py [--splits val,test] [--limit 50]

--limit N crea un task de prueba con solo las primeras N imágenes del split
(y un zip de anotaciones filtrado a esas imágenes).
"""

import argparse
import os
import re
import tempfile
import zipfile

from cvat_sdk import make_client
from cvat_sdk.api_client import models
from cvat_sdk.core.proxies.tasks import ResourceType

SUBSET_NAME = {"train": "Train", "val": "Validation", "test": "Test"}
ANNOTATION_FORMAT = "Ultralytics YOLO Detection 1.0"


def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="http://localhost:8080")
    p.add_argument("--user", default="admin")
    p.add_argument("--password-file", default=os.path.join(base, ".cvat_admin_pass"))
    p.add_argument("--dataset", default=os.path.join(base, "dataset_yolo"))
    p.add_argument("--project", default="Fishnet FOID")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--limit", type=int, default=None,
                   help="Solo las primeras N imágenes del split (task de prueba)")
    p.add_argument("--image-quality", type=int, default=70)
    return p.parse_args()


def load_classes(dataset_dir):
    names, in_names = {}, False
    with open(os.path.join(dataset_dir, "data.yaml")) as f:
        for line in f:
            if line.strip() == "names:":
                in_names = True
                continue
            if in_names:
                m = re.match(r"\s+(\d+):\s*(\S+)", line)
                if not m:
                    break
                names[int(m.group(1))] = m.group(2)
    if not names:
        raise SystemExit(f"No pude leer las clases de {dataset_dir}/data.yaml")
    return [names[i] for i in sorted(names)]


def split_stems(dataset_dir, split):
    labels_dir = os.path.join(dataset_dir, "labels", split)
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(labels_dir) if f.endswith(".txt"))


def build_annotations_zip(dataset_dir, split, stems, classes, zip_path):
    # Estructura espejo de lo que el propio CVAT/datumaro exporta para un task
    # creado desde el share (frames llamados "images/<uuid>.jpg"):
    #   data.yaml                          -> <Subset>: <Subset>.txt
    #   <Subset>.txt                       -> data/images/<Subset>/images/<uuid>.jpg
    #   labels/<Subset>/images/<uuid>.txt     (el prefijo "data/" de las rutas
    #                                          del .txt se descarta al importar)
    subset = SUBSET_NAME[split]
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    data_yaml = f"{subset}: {subset}.txt\nnames:\n{names_block}\npath: .\n"
    file_list = "\n".join(f"data/images/{subset}/images/{s}.jpg" for s in stems) + "\n"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml)
        zf.writestr(f"{subset}.txt", file_list)
        for s in stems:
            zf.write(os.path.join(dataset_dir, "labels", split, s + ".txt"),
                     f"labels/{subset}/images/{s}.txt")


def get_or_create_project(client, name, classes):
    for proj in client.projects.list():
        if proj.name == name:
            print(f"Proyecto existente '{name}' (id {proj.id})")
            return proj
    proj = client.projects.create(models.ProjectWriteRequest(
        name=name,
        labels=[models.PatchedLabelRequest(name=c) for c in classes],
    ))
    print(f"Proyecto creado '{name}' (id {proj.id})")
    return proj


def upload_split(client, project, dataset_dir, split, classes, args):
    stems = split_stems(dataset_dir, split)
    if args.limit:
        stems = stems[:args.limit]
    task_name = f"foid_{split}" + (f"_smoke{args.limit}" if args.limit else "")

    for task in client.tasks.list():
        if task.name == task_name and task.project_id == project.id:
            print(f"  [{split}] el task '{task_name}' ya existe (id {task.id}), lo salto")
            return

    print(f"  [{split}] creando task '{task_name}' con {len(stems)} imágenes del share ...")
    task = client.tasks.create_from_data(
        spec=models.TaskWriteRequest(
            name=task_name, project_id=project.id, subset=SUBSET_NAME[split]),
        resource_type=ResourceType.SHARE,
        resources=[f"images/{s}.jpg" for s in stems],
        data_params={
            "image_quality": args.image_quality,
            "storage_method": models.StorageMethod("cache"),
            "copy_data": False,
            "sorting_method": models.SortingMethod("lexicographical"),
        },
    )
    print(f"  [{split}] task creado (id {task.id}), subiendo anotaciones ...")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, f"annotations_{split}.zip")
        build_annotations_zip(dataset_dir, split, stems, classes, zip_path)
        task.import_annotations(format_name=ANNOTATION_FORMAT, filename=zip_path)

    print(f"  [{split}] listo: {args.host}/tasks/{task.id}")


def main():
    args = parse_args()
    with open(args.password_file) as f:
        password = f.read().strip()
    classes = load_classes(args.dataset)

    with make_client(args.host, credentials=(args.user, password)) as client:
        project = get_or_create_project(client, args.project, classes)
        for split in args.splits.split(","):
            upload_split(client, project, args.dataset, split.strip(), classes, args)

    print(f"\nRevisa el proyecto en {args.host}/projects/{project.id}")


if __name__ == "__main__":
    main()
