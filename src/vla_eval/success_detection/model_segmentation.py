from ultralytics.models.sam import SAM3SemanticPredictor
import cv2
import json
import numpy as np
import time


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def remove_duplicated_values(masks, confs, iou_thr=0.5):
    ordem = np.argsort(-confs)
    mantidas = []
    for i in ordem:
        if all(mask_iou(masks[i], masks[j]) < iou_thr for j in mantidas):
            mantidas.append(i)
    return mantidas


def base_point(mask):
    ys, xs = np.nonzero(mask)
    y_max = ys.max()
    x_base = xs[ys >= y_max - 3].mean()  # média das últimas linhas
    return float(x_base), float(y_max)


def count_objects(image: np.ndarray, predictor, roi, prompt: str, min_area=200, max_area=50_000):
    predictor.set_image(image)

    # Consulta com vários prompts de texto
    results = predictor(text=prompt)[0]

    contagem = {nome: 0 for nome in roi}
    deteccoes = []

    if results.masks is None:
        return contagem, deteccoes

    masks = results.masks.data.cpu().numpy().astype(bool)
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for i in remove_duplicated_values(masks, confs):
        area = masks[i].sum()
        if not (min_area < area < max_area):
            continue
        x, y = base_point(masks[i])
        for nome, poly in roi.items():
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                contagem[nome] += 1
                deteccoes.append({"regiao": nome, "classe": prompt[classes[i]],
                                "conf": float(confs[i]), "area": int(area)})
                break 

    return contagem, deteccoes


def model_inference(before_image: np.ndarray, after_image: np.ndarray, prompt: list[str], rois): 
    #TODO: Change save_dir to global save path instead of relative.
    overrides = {
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "model": "src/vla_eval/success_detection/model_weights/sam3.pt",
        "quantize": 16,  # Usa FP16 para inferência mais rápida
        "save": True,
        "save_dir": "eval_results/sam3_detections",  # relative to the directory where the script 
    }

    predictor = SAM3SemanticPredictor(overrides=overrides)

    ROIS = {k: np.array(v, np.int32) for k, v in rois.items()}

    inicio, det_ini = count_objects(before_image, predictor=predictor, roi=ROIS, prompt=prompt)
    fim, det_fim = count_objects(after_image, predictor=predictor, roi=ROIS, prompt=prompt)

    total_ini = sum(inicio.values())

    print("Início:", inicio, "| Fim:", fim)
    print("Tarefa completa:", fim["table"] == 0 and fim["container"] == total_ini)

    return inicio, fim