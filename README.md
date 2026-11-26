# Pequi_Mêcanico_UFG_EVAL

Métricas reprodutíveis para avaliar modelos Vision-Language-Action (VLA).

## Uso como pacote pip

Ao entrar em um repositório que implementa uma política de VLA, basta instalar o pacote no ambiente dele:

```bash
pip install eval_vla
```

## Instalação Local

**Recomendação** - Utilizar o uv como gerenciador de pacotes, caso não tenha instalado, siga os passos em https://docs.astral.sh/uv/getting-started/installation/

```bash
git clone https://github.com/Pequi-Mecanico-Research/Eval_VLA.git
cd Eval_VLA
uv venv
source .venv/bin/activate
uv sync
```

Em seguida, somente implemente as métricas dentro do loop de avaliação da sua política, como mostrado em [Uso como repositório local](#uso-como-repositório-local) e nos arquivos de `examples/`.

### Pesos do modelo (SAM3)

O pacote instalado via pip **não inclui os pesos** do SAM3. Eles são necessários nas métricas `success_rate` (modo `"model_segmentation"`) e `human_intervention_rate`.

1. Baixe o arquivo `sam3.pt` no repositório da Meta no Hugging Face: https://huggingface.co/facebook/sam3 (o acesso é restrito: é preciso solicitar acesso na página do modelo e estar logado).
2. Passe o caminho do arquivo no parâmetro `weights` das métricas:

```python
stats.success_rate(..., model="sam3", weights="weights/sam3.pt")
operation.human_intervention_rate(video_path, model="sam3", weights="weights/sam3.pt")
```

Se `weights` não for informado, o pacote procura em `model_weights/<modelo>.pt` dentro do próprio pacote, que só existe ao rodar o repositório com os pesos colocados lá.

### Alternativa leve: YOLO-World

As mesmas métricas aceitam `model="yoloworld"`, bem menor e mais rápido que o SAM3. Os pesos são **baixados automaticamente** no caminho passado em `weights`, caso não existam. O nome do arquivo deve ser um dos oficiais: `yolov8{s,m,l,x}-world.pt` ou `yolov8{s,m,l,x}-worldv2.pt`. Sem `weights`, usa `yolov8s-world.pt`. Na primeira execução o encoder de texto CLIP (ViT-B/32) também é baixado.

```python
stats.success_rate(..., model="yoloworld", weights="weights/yolov8s-world.pt", conf=0.5)
operation.human_intervention_rate(video_path, model="yoloworld", conf=0.5)
```

### Análise de falhas com Hugging Face

A `FailureAnalysis` usa o Ollama por padrão (`mode="ollama"`). Para rodar o Qwen pelo `transformers`, instale o extra e use `mode="hf"`; os pesos são baixados em `model_weights` na primeira execução:

```bash
pip install "eval_vla[hf]"   # no repositório clonado: uv sync --extra hf
```

```python
FailureAnalysis(config, video_path, mode="hf", model_weights="weights/Qwen3.5-9B")
```

O `Qwen/Qwen3.5-9B` ocupa cerca de 19 GB em disco e não cabe inteiro numa GPU de 16 GB; o excedente vai para a RAM, o que deixa a inferência lenta. Um modelo menor pode ser escolhido com `vlm_model="Qwen/Qwen3.5-4B"`.

## Uso ao clonar repositório local

Código utilizado como demonstração de uso em algum repositório que implemente uma política de VLA e gostariamos de avalia-lá:

1 - Configuração da Classe que será utilizada.
2 - Instânciar os Metódos que serão utilizados para avaliar durante o episódio.
3 - Finalizar a execução com log das métricas por episódio e gráficos.

### 1 - Configuração da Classe que será utilizada

```python
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from examples.operation_eval import JOINT_NAMES
from examples.statistical_eval_image_diff import CALIBRATION_DIR, fake_policy, load_or_capture_reference
from examples.widowx_driver_probe import connect_driver
from eval_vla.config import SetupConfig
from eval_vla.metrics.failure_analysis import FailureAnalysis
from eval_vla.metrics.long_horizon import LongHorizon
from eval_vla.metrics.operation import Operation
from eval_vla.metrics.statistical import Statistical
from eval_vla.vision.camera import Camera
from eval_vla.vision.roi_selector import RoiSelector
from eval_vla.vision.viewer import make_viewer

start_eval_time = time.perf_counter()  # marca o início da avaliação completa

config = SetupConfig(  # parâmetros do ambiente/robô, salvos junto com todo resultado
    robot_embodiment={"manufacturer": "Trossen", "model": "WidowX", "dof": 6, "end_effector": "Precision Gripper"},
    camera_configuration={"count": 3, "positions": ["wrist", "front", "high"], "resolution": [640, 480], "wrist": True},
    action_space={"type": "end_effector_pose", "dim": 7},
    control_frequency=10.0,  # Hz do hardware em uso
    object_set={"count": 1, "materials": ["plastic"], "seen_in_training": True},
    scene_layout={"workspace_m": "Grey Table with 100 meters", "lighting": "indoor"},
    reset_procedure="VLA",  # como a cena é reiniciada entre episódios
    success_criteria="objects inside the tots",
    model_name="openpi-finetune",  # formato: modelo_base-finetune
    experiment_name="Full_eval_Complete",  # também nomeia a pasta de resultados
    task_name="Organize_the_table",  # uma das tarefas da classe Statistical ou uma nova
    perturbation_type=None,  # list of: object, layout, camera, lighting, background, initial_state, sensor_noise
)

# uma pasta para a execução inteira: <name>_episodes.csv / <name>.png por família
run_dir = Path("eval_results") / f"{config.experiment_name}_{datetime.now():%d-%m_%H:%M:%S}"
```

### 2 - Instânciar os Metódos que serão utilizados para avaliar durante o episódio

```python
n_episodes, steps_per_episode = 5, 50  # número de episódios e passos da política por episódio
SAM3_WEIGHTS = "weights/sam3.pt"  # pesos do SAM3, baixados de https://huggingface.co/facebook/sam3

stats = Statistical(config, run_dir=run_dir, name="statistical")  # name prefixa os arquivos da família
operation = Operation(config, episodes=n_episodes, run_dir=run_dir, name="operation")
long_horizon = LongHorizon(config, run_dir=run_dir, name="long_horizon")

camera = Camera(0).open()  # 0 = índice da câmera
view = make_viewer("rerun", camera)  # visualização ao vivo dos sinais do robô
driver = connect_driver("192.168.1.2")  # IP do driver do braço

# imagem de referência e regiões de interesse (calibradas uma vez e reutilizadas depois)
reference_image = load_or_capture_reference(camera, CALIBRATION_DIR / config.task_name / "reference.png")
rois = RoiSelector.load_or_create(CALIBRATION_DIR / config.task_name / "rois.json", ["table", "container"], reference_image)

for ep in range(n_episodes):
    episode_dir = stats.episode_dir(ep)  # pasta run_dir/ep_<n>
    video_path = episode_dir / "video.mp4"
    before_frame = camera.capture(episode_dir / "before.png")  # cena antes da tarefa
    camera.start_recording(video_path, preview=view.preview)

    start_time = time.perf_counter()
    for step in range(steps_per_episode):
        inference_start = time.perf_counter()
        fake_policy(obs=None)  # substitua pela chamada da sua política
        stats.inference_latency(inference_start, time.perf_counter(), steps_per_episode=steps_per_episode)

        external_efforts = dict(zip(JOINT_NAMES, (round(e, 2) for e in driver.get_all_external_efforts())))
        # calibração opcional (recomendada): calibration="calibration/collision_gmo.json", ver metrics/README.md
        operation.collision_rate(driver, mode="gmo")  # detecção de colisão pelo momento generalizado (GMO)
        operation.jerks_rate(driver)

        view.update(
            external_effort=external_efforts,
            gmo_residual=dict(zip(JOINT_NAMES, operation.gmo_residual)),
            jerk=dict(zip(JOINT_NAMES, operation.jerks_values)),
        )

    stats.task_completion_time(start_time, time.perf_counter())

    camera.wait_for_enter(f"Episode {ep}: add the objects to the tot before finishing the task, then press Enter...", on_tick=view.update)
    after_frame = camera.capture(episode_dir / "after.png")  # cena depois da tarefa
    camera.stop_recording()

    # sucesso medido comparando antes/depois com segmentação (SAM3) dentro das ROIs
    stats.success_rate(config.task_name, "model_segmentation", before_frame, after_frame, prompt=["Cube", "black clock cube"], rois=rois, model="sam3", weights=SAM3_WEIGHTS)
    operation.reset_failure_rate(driver)
    operation.human_intervention_rate(video_path, prompt=["hand", "human arm"], model="sam3", weights=SAM3_WEIGHTS)  # detecta mão/braço no vídeo

    subtasks_completed = [bool(stats.success_rate_value)]  # aqui a tarefa tem uma única subtarefa
    long_horizon.subtask_completion_rate(subtasks_completed)
    long_horizon.average_completed_sequence_length(subtasks_completed)
    long_horizon.failure_position_distribution(subtasks_completed)
```

### 3 - Finalizar a execução com log das métricas por episódio e gráficos

```python
    # ainda dentro do loop de episódios
    stats.end_episode()  # salva o episódio em ep_<n>, acrescenta uma linha no CSV da execução e atualiza os gráficos
    operation.end_episode()
    long_horizon.end_episode()

    if not stats.success_rate_value:  # só analisa a causa da falha quando o episódio falhou
        analysis = FailureAnalysis(config, video_path, window_s=30.0, debug_video=False,  # window_s: janela de vídeo analisada (s)
                                   run_dir=run_dir, name="failure_analysis")
        print(f"Episode {ep} failure type:", analysis.failure_type)
        analysis.end_episode(episode=ep + 1)  # episode é 1-based no log

# fora do loop: libera os recursos
view.close()
camera.close()
```

Ao final, a pasta `run_dir` contém:

```
run_dir/
├── <name>_episodes.csv      # todos os episódios, uma linha cada (por família)
├── <name>.png               # gráficos da execução
├── ep_0/
│   ├── <name>_results.json  # métricas e config do episódio
│   ├── <name>_episodes.csv  # a linha do episódio
│   └── video.mp4, before.png, after.png, ...
└── ep_1/
```

Os resultados das métricas são limpos a cada `end_episode()`: uma métrica não chamada no episódio não aparece nele.

### Comparação entre avaliações com Weights & Biases (opcional)

O `wandb` já é instalado junto com o pacote; basta fazer o login:

```bash
wandb login   # uma vez, no terminal
```

```python
config = SetupConfig(..., wandb=True, wandb_project="eval-vla")
```

Cada execução vira um run do projeto (com o `SetupConfig`), com as métricas e os gráficos por episódio. As médias de cada métrica vão para o resumo do run (`<name>/<métrica>_mean`), para comparar uma avaliação com outra.

## Adicionando uma nova classe de métricas

1. Criar `src/eval_vla/metrics/<nome>.py` herdando de `EvalVLA`:

   ```python
   from pathlib import Path

   from ..base import EvalVLA, metric
   from ..config import SetupConfig
   from ..plots import plot_minha_familia  # opcional: função que lê o CSV e salva o gráfico


   class MinhaFamilia(EvalVLA):
       def __init__(self, config: SetupConfig, output_dir: str | Path = "eval_results",
                    run_dir: str | Path | None = None, name: str | None = None):
           super().__init__(config, output_dir, run_dir, name)  # cria a pasta e define json_path/csv_path

       def _save_graphs(self) -> None:  # chamado automaticamente em end_episode()
           plot_minha_familia(self.csv_path, self.output_dir, self.config.experiment_name)

       @metric  # guarda o retorno em self._results e o grava no JSON/CSV
       def minha_metrica(self, valor: float) -> dict[str, float]:
           return {"minha_metrica": valor}  # chaves do dict viram colunas do CSV
   ```

2. Usar a classe como as demais, chamando `end_episode()` ao fim de cada episódio:

   ```python
   minha_familia = MinhaFamilia(config, run_dir=run_dir, name="minha_familia")
   minha_familia.minha_metrica(0.9)
   minha_familia.end_episode()
   ```

Métodos decorados com `@metric` devem retornar um `dict`; ele é salvo em `<name>_results.json` e vira uma linha do `<name>_episodes.csv` a cada `end_episode()`.
