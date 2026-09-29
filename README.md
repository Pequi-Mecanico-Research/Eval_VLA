# pequi-eval

Métricas reprodutíveis para avaliar modelos Vision-Language-Action (VLA).

## Instalação

```bash
pip install git+https://github.com/<org>/Eval_VLA.git          # só numpy
pip install "pequi-eval[scipy] @ git+https://github.com/<org>/Eval_VLA.git"  # + Clopper-Pearson
pip install -e ".[dev]"                                         # desenvolvimento
```

## Uso

```python
from pequi_eval import SetupConfig, Statistical

config = SetupConfig(
    robot_embodiment={"name": "ur5e", "gripper": "robotiq_2f85"},
    camera_configuration={"wrist": {"fps": 30}, "front": {"fps": 30}},
    action_space={"type": "ee_delta", "dim": 7},
    control_frequency=10.0,
    object_set={"cube": "red 5cm"},
    scene_layout={"table_height_m": 0.75},
    reset_procedure="manual, object at marked spot",
    success_criteria="cube inside the bowl",
)
ev = Statistical(config, metadata={"model": "openvla-7b", "seed": 0})

for episode in range(50):
    ev.start_trial("pick_cube")
    while not done:
        with ev.measure_latency():
            action = policy(obs)
        ...
    ev.end_trial(success=ok)

ev.trials_per_task()            # {"pick_cube": 50}; avisa se < 50
ev.success_rate("pick_cube")    # rate + IC 95% (method="wilson" | "clopper_pearson")
ev.completion_time("pick_cube") # mean_s, std_s
ev.inference_latency()          # mean_ms, p95_ms
ev.report()                     # todas as métricas, por tarefa e agregadas ("all")
ev.save("results/run.json")     # config + trials brutos + versão/ambiente
ev = Statistical.load("results/run.json")
```

Se o tempo for medido por outro sistema, use `ev.record_trial(task, success, duration_s, latencies_ms)`.

## Várias famílias de métricas

Toda família herda de `EvalVLA` e pode compartilhar o mesmo registro de trials:

```python
robust = Robustness.from_eval(ev)   # mesma config, mesmos trials
```

## Adicionando uma nova família

1. Criar `src/pequi_eval/metrics/<nome>.py`:

   ```python
   from ..base import EvalVLA, metric

   class Robustness(EvalVLA):
       @metric
       def minha_metrica(self, task: str | None = None) -> dict:
           trials = self.log.filter(task)
           ...
   ```

   Não adicione parâmetros obrigatórios ao `__init__`; dados extras por trial vão em
   `record_trial(..., **metadata)` / `start_trial(..., **metadata)`.
2. Exportar em `metrics/__init__.py` e `pequi_eval/__init__.py`.
3. Testes em `tests/`.

Métodos com `@metric` entram automaticamente em `report()`; se aceitarem `task`, são
calculados por tarefa e agregados.
