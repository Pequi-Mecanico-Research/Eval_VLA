# Métricas

Cada classe de métricas herda de `EvalVLA` e cada método decorado com `@metric` é uma métrica.

Estrutura usada para cada método:

- **Para que serve:**
- **O que faz:**
- **Variáveis de entrada:**
- **Retorno:**

---

## Statistical

`statistical.py` — Relatório estatístico: taxa de sucesso, tempo de conclusão da tarefa (média e desvio padrão, em segundos) e latência de inferência no hardware em uso (média e p95).

### `success_rate`

- **Para que serve:** Verificar se a política moveu os objetos da mesa para o recipiente.
- **O que faz:** Conta os objetos nas ROIs antes e depois da ação da política e compara. Possui dois modos (`task_mode`):
  - `"model_segmentation"`: conta os objetos de `prompt` dentro das ROIs usando `model`, em `before_frame` e `after_frame`. Precisa de `prompt` e `rois`.
  - `"image_diff"`: conta os objetos comparando cada frame com `reference_frame` (mesa vazia). Precisa de `reference_frame` e `rois`.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `task_name` | Tarefa avaliada (atualmente apenas `"Organize_the_table"`). |
  | `task_mode` | `"model_segmentation"` ou `"image_diff"`. |
  | `before_frame` | Imagem da mesa logo antes de a política agir. |
  | `after_frame` | Imagem da mesa logo depois de a política agir. |
  | `prompt` | Nomes/descrições dos objetos a segmentar. |
  | `reference_frame` | Imagem da mesa vazia, usada como base no modo `"image_diff"`. |
  | `rois` | Regiões de interesse onde os objetos começam (mesa) e terminam (recipiente). |
  | `model` | Modelo de detecção usado no modo `"model_segmentation"`: `"sam3"` ou `"yoloworld"` (mais leve e rápido). |
  | `debug_plots` | Salva as imagens de debug do modo `"image_diff"`. |
  | `weights` | Caminho do arquivo de pesos (ex.: `sam3.pt`) de `model`, usado em `"model_segmentation"`. Se `None`, procura em `model_weights/<model>.pt` dentro do pacote, que não vem no `pip install`; baixe em https://huggingface.co/facebook/sam3 (ver README da raiz). Só é lido na primeira vez que o detector é criado. Com `"yoloworld"`, o peso é baixado automaticamente nesse caminho se não existir (o nome deve ser oficial, ex.: `yolov8s-world.pt`). |
  | `imgsz` | Tamanho (quadrado) da inferência. Se `None`, usa 640 no `"sam2"` e no `"yoloworld"` e 644 no `"sam3"` (múltiplos do stride). Só é lido na primeira vez que o detector é criado. |
  | `conf` | Confiança mínima de uma detecção. Se `None`, usa 0.25 nos modelos SAM e 0.5 no `"yoloworld"`. Só é lido na primeira vez que o detector é criado. |

- **Retorno:** dict com `successes` (bool), `objects_on_table_before` e `objects_in_container_after` (int, ou `None` para tarefas desconhecidas).

### `task_completion_time`

- **Para que serve:** Medir quanto tempo a política levou para concluir o episódio.
- **O que faz:** Acumula as durações dos episódios e calcula média, mediana e desvio padrão. Chamar uma vez por episódio, depois que a política para de agir.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `start_time` | Início do episódio, de `time.perf_counter()`, em segundos. |
  | `end_time` | Fim do episódio, de `time.perf_counter()`, em segundos. |
  | `precision` | Casas decimais usadas para arredondar os valores retornados. |

- **Retorno:** dict com `task_completion_sec_list` (duração deste episódio) e `task_completion_sec_mean`, `_median` e `_std` sobre todos os episódios até agora, em segundos.

### `inference_latency`

- **Para que serve:** Medir o tempo de inferência da política em cada passo.
- **O que faz:** Acumula as latências ao longo do episódio e calcula média, mediana, desvio padrão e p95. Chamar uma vez por passo, envolvendo apenas a chamada da política.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `inference_time_start` | Tempo logo antes da chamada da política, de `time.perf_counter()`, em segundos. |
  | `inference_time_end` | Tempo logo depois da chamada da política, de `time.perf_counter()`, em segundos. |
  | `steps_per_episode` | Número de passos da política em um episódio, usado para reiniciar as latências acumuladas no próximo episódio. |
  | `precision` | Casas decimais usadas para arredondar os valores retornados. |

- **Retorno:** dict com `inference_latency_sec_list` (latência deste passo) e `inference_latency_sec_mean`, `_median`, `_std` e `_p95` das latências acumuladas, em segundos.

---

## Operation

`operation.py` — Métricas operacionais: taxa de intervenção humana, taxa de colisão, taxa de jerks (movimentos bruscos) e taxa de falha de reset.

### `human_intervention_rate`

- **Para que serve:** Detectar se um humano entrou na cena durante o episódio, ou seja, precisou intervir manualmente.
- **O que faz:** Segmenta o vídeo do episódio procurando o que está em `prompt` e retorna os intervalos de tempo com presença humana. Chamar uma vez por episódio, sobre o vídeo gravado.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `video_path` | Vídeo do episódio. |
  | `model` | Modelo de detecção usado para encontrar o humano no vídeo: `"sam2"`, `"sam3"` ou `"yoloworld"` (mais leve e rápido). |
  | `prompt` | Descrições do que conta como intervenção humana (ex.: `"hand"`, `"human arm"`). |
  | `precision` | Casas decimais usadas para arredondar os valores retornados. |
  | `weights` | Caminho do arquivo de pesos (ex.: `sam3.pt`) de `model`. Se `None`, procura em `model_weights/<model>.pt` dentro do pacote, que não vem no `pip install`; baixe em https://huggingface.co/facebook/sam3 (ver README da raiz). Só é lido na primeira vez que o detector é criado. Com `"yoloworld"`, o peso é baixado automaticamente nesse caminho se não existir (o nome deve ser oficial, ex.: `yolov8s-world.pt`). |
  | `imgsz` | Tamanho (quadrado) da inferência. Se `None`, usa 640 no `"sam2"` e no `"yoloworld"` e 644 no `"sam3"` (múltiplos do stride). Só é lido na primeira vez que o detector é criado. |
  | `conf` | Confiança mínima de uma detecção. Se `None`, usa 0.70 nos modelos SAM e 0.5 no `"yoloworld"`. Só é lido na primeira vez que o detector é criado. |
  | `**detection_kwargs` | Argumentos extras repassados para `ObjectDetection.run`. |

- **Retorno:** dict com `human_interventions` (número de intervalos com humano), `human_intervention_sec` (tempo total com humano), `human_intervention_intervals` (lista de (início, fim) em segundos) e `human_intervention_rate` (fração dos `episodes` com ao menos uma intervenção até agora).

### `collision_rate`

- **Para que serve:** Detectar contato não intencional (colisão) durante o episódio a partir dos torques das juntas do braço.
- **O que faz:** Chamar a cada passo de controle. Guarda o pico do sinal por junta ao longo do episódio e marca colisão quando algum pico passa do limite. Modos (`mode`):
  - `"gmo"`: observador de momento generalizado. O resíduo estima o torque externo usando o modelo dinâmico do braço (URDF) e as posições, velocidades e esforços medidos.
  - `"driver"`: usa os esforços externos já informados pelo driver do braço.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `driver` | Driver do braço Trossen conectado, lido a cada chamada. |
  | `mode` | `"gmo"` ou `"driver"`. |
  | `thresholds` | Valor absoluto máximo aceito por junta do braço, em Nm. |
  | `gain` | Ganho K_O do observador no modo `"gmo"`, em 1/s. Valores maiores reagem mais rápido, mas adicionam ruído. |
  | `precision` | Casas decimais usadas para arredondar os valores retornados. |

- **Retorno:** dict com `collision_mode`, `collision_detected` (neste episódio), `collision_rate` (fração dos `episodes` com colisão até agora), `collision_thresholds` e `collision_values` (pico por junta). Levanta `ValueError` se `mode` não for `"gmo"` nem `"driver"`.

### `jerks_rate`

- **Para que serve:** Detectar movimento brusco (mudança abrupta de aceleração) durante o episódio.
- **O que faz:** Chamar a cada passo de controle. O jerk é a derivada das acelerações ao longo do episódio; guarda o pico por junta e do efetuador final, e marca o episódio quando algum pico passa do limite.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `driver` | Driver do braço Trossen conectado, lido a cada chamada. |
  | `joint_jerk_thresholds` | Jerk máximo aceito por junta, em rad/s³ (m/s³ para a garra). |
  | `linear_jerk_threshold` | Jerk linear máximo aceito do efetuador final, em m/s³. |
  | `angular_jerk_threshold` | Jerk angular máximo aceito do efetuador final, em rad/s³. |
  | `precision` | Casas decimais usadas para arredondar os valores retornados. |

- **Retorno:** dict com `jerks_detected` (neste episódio), `jerks_rate` (fração dos episódios com jerks até agora), `jerks_thresholds` e `jerks_values` (picos por junta, depois linear e angular).

### `reset_failure_rate`

- **Para que serve:** Verificar se o braço voltou à pose de reset (sleep), ou seja, se o procedimento de reset funcionou.
- **O que faz:** Compara as posições atuais das juntas com a pose de reset e marca falha se algum erro passar da tolerância. Chamar uma vez por episódio, quando o braço deveria estar na pose de reset.
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `driver` | Driver do braço Trossen conectado, lido na chamada. |
  | `sleep_positions` | Posições esperadas das juntas na pose de reset, nas unidades do driver. |
  | `tolerance` | Erro absoluto máximo aceito por junta; acima dele o reset conta como falho. |
  | `precision` | Casas decimais usadas para arredondar os valores retornados. |

- **Retorno:** dict com `reset_failed` (neste episódio), `reset_max_joint_error`, `reset_joint_errors` (por junta) e `reset_failure_rate` (fração dos episódios com reset falho até agora).

---

## LongHorizon

`long_horizon.py` — Tarefas de longo horizonte: taxa de conclusão de subtarefas, comprimento médio da sequência concluída e distribuição das posições de falha ao longo da sequência. **Os três métodos ainda não estão implementados.**

### `subtask_completion_rate`

- **Para que serve:** Medir a fração das subtarefas individuais concluídas, independente de a sequência completa ter tido sucesso.
- **O que faz:** Chamar uma vez por episódio. *(Não implementado ainda.)*
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `subtasks_completed` | Um bool por subtarefa, em ordem de execução; `True` se aquela subtarefa foi concluída. |

- **Retorno:** dict com a taxa de conclusão de subtarefas. *(Não implementado ainda.)*

### `average_completed_sequence_length`

- **Para que serve:** Medir o número médio de subtarefas consecutivas concluídas desde o início da sequência (no estilo CALVIN).
- **O que faz:** Chamar uma vez por episódio. *(Não implementado ainda.)*
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `subtasks_completed` | Um bool por subtarefa, em ordem de execução; `True` se aquela subtarefa foi concluída. |

- **Retorno:** dict com o comprimento médio da sequência concluída. *(Não implementado ainda.)*

### `failure_position_distribution`

- **Para que serve:** Mostrar em qual ponto da sequência as falhas se concentram.
- **O que faz:** Histograma do índice da subtarefa (começando em 1) em que cada tentativa falhou pela primeira vez. Chamar uma vez por episódio. *(Não implementado ainda.)*
- **Variáveis de entrada:**

  | Variável | Descrição |
  |---|---|
  | `subtasks_completed` | Um bool por subtarefa, em ordem de execução; `True` se aquela subtarefa foi concluída. |

- **Retorno:** dict com a distribuição das posições de falha. *(Não implementado ainda.)*

---

## FailureAnalysis

`failure_analysis.py` — Análise de falhas. Classifica o episódio em tipos de falha: percepção, compreensão de linguagem, planejamento, preensão (grasp), controle, recuperação e hardware/calibração.

### `vlm_failure_type`

- **Para que serve:** Descobrir, com um modelo visão-linguagem (VLM, via Ollama), qual tipo de falha o vídeo do episódio mostra.
- **O que faz:** Amostra `vlm_frames` frames dos últimos `window_s` segundos do vídeo e os envia ao VLM junto com o nome da tarefa e o critério de sucesso do config. Roda automaticamente quando `FailureAnalysis` é criada.
- **Variáveis de entrada:** O método não recebe argumentos; as entradas são definidas no `__init__`.

  | Variável (`__init__`) | Descrição |
  |---|---|
  | `video_path` | Vídeo do episódio analisado. |
  | `window_s` | Janela final do vídeo (em segundos) usada na análise. |
  | `vlm_model` | Modelo do Ollama usado (padrão `qwen3.5:latest`). |
  | `vlm_host` | Endereço do servidor Ollama. |
  | `vlm_frames` | Número de frames amostrados para o VLM. |
  | `debug_video` | Salva um vídeo com os frames enviados ao VLM. |
  | `vlm_timeout_s` | Tempo limite da requisição ao VLM, em segundos. |
  | `vlm_num_ctx` | Tamanho do contexto (`num_ctx`) do VLM. |

- **Retorno:** dict com `vlm_failure_type` (nome de um `FailureType`, ou `None` se a resposta não foi reconhecida), `vlm_model` e `vlm_raw_reply` (resposta do modelo em texto).
