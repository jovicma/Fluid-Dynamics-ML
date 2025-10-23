# Sequência de Prompts — Projeto **riemann-ml** (Codex/OpenAI)

> **Objetivo:** esta sequência de prompts foi pensada para ser colada, um a um, em um modelo de código (ex.: Codex / GPT com capacidade de geração de código) para implementar um projeto completo envolvendo CFD 1D (Euler/Sod), solver FVM, solução exata, geração de dados, PINN (TensorFlow), FNO (PyTorch/neuraloperator), avaliação, CLI, testes e Docker.
>
> **Dicas de uso:**
> - Envie cada prompt e aguarde o código. Se a resposta for muito grande, peça “continue from last file”.
> - Onde aparecer `{{...}}`, ajuste depois (ex.: nº de camadas, modos de Fourier, etc.).
> - Os prompts pedem para imprimir **apenas blocos de código**, separados por linhas `# ==== path/to/file.ext ====`, quando pertinente.
> - Se a interface não suportar múltiplos arquivos numa mesma resposta, execute o mesmo prompt em partes por seção “Arquivos a criar”.

---

## Prompt 0 — Contexto & Estilo (cole antes de qualquer coisa)

> Você é um engenheiro sênior de ML/CFD. Gere **apenas** blocos de código, separados por linhas do tipo `# ==== path/to/file.ext ====`.
> Linguagem: Python 3.11+.
> Qualidade: PEP8, tipagem, docstrings, logs, testes.
> Projeto: “riemann-ml”.
> Tarefas: criarei uma sequência de prompts; em cada um, crie/edite **exatamente** os arquivos pedidos.
> Não invente dependências exóticas; use o que eu pedir.
> Sempre inclua exemplos de uso em `README.md` e **scripts/`*.sh`** reprodutíveis.

---

## Prompt 1 — Estrutura inicial, configs e dependências

Crie a estrutura de pastas, `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`, configuração de linters/formatadores e um `Dockerfile`.

**Arquivos a criar/alterar**
- `.gitignore` (Python + venv + data)
- `pyproject.toml` (poetry ou hatch; escolha **hatch**)
- `requirements.txt` contendo: `numpy`, `scipy`, `matplotlib`, `h5py`, `pydantic>=2`, `typer[all]`, `rich`, `tqdm`, `pytest`, `pytest-cov`, `coverage`, `pre-commit`, `hydra-core`, `omegaconf`, `tensorboard`, `tensorflow>=2.16`, `torch>=2.2`, `neuraloperator`, `torchvision`, `torchaudio`
- `Dockerfile` (imagem slim + system deps mínimos)
- `README.md` (overview, como instalar, comandos básicos)
- `scripts/dev_install.sh`, `scripts/run_tests.sh`, `scripts/format.sh`
- Pastas: `src/riemann_ml/{core,fvm,exact,data,eval,ml/pinn,ml/fno,utils,configs}`, `tests`, `notebooks`, `data/{raw,processed,artifacts}`

Inclua no README um **logo ASCII** simples e instruções de uso com `hatch` **ou** `pip -r requirements.txt`.

---

## Prompt 2 — Núcleo de física (conversões, fluxo, EOS)

Implemente utilitários do sistema de Euler 1D (γ=1.4 por padrão) com conversões conservativo↔primitivo, fluxo e velocidade do som.

**Arquivos a criar/alterar**
- `src/riemann_ml/core/euler1d.py`
  - `StatePrim`/`StateCons` (pydantic)
  - `prim_to_cons`, `cons_to_prim`
  - `pressure_from_cons`, `sound_speed`, `flux_vector` (para [ρ, ρu, E])
  - Tratamento numérico robusto (clamp de p e ρ > 0, eps)
- `tests/test_euler1d.py` (round-trip prim↔cons e sanidade do fluxo)

Inclua docstrings com fórmulas e referências, e valor default `gamma=1.4`.

---

## Prompt 3 — Solucionador FVM 1D (Lax-Friedrichs/Rusanov)

Implemente um esquema de volumes finitos 1D estável para Euler com Rusanov, CFL, e BCs de saída.

**Arquivos**
- `src/riemann_ml/fvm/solver.py`
  - Funções: `initialize_grid`, `riemann_flux_rusanov`, `advance_one_step`, `simulate`
  - Entradas: malha uniforme [0,1], N células, CFL, Tf, estados iniciais por partes (Riemann) com salto em x0=0.5
  - Saídas: arrays (t, x, q(t_final)) e histórico opcional
- `src/riemann_ml/utils/plotting.py` com helpers `plot_profiles(x, rho,u,p, title, savepath)`
- `tests/test_fvm_rusanov.py` (teste de estabilidade: mass/energy não NaN, dt por CFL)

Inclua um script:
- `scripts/run_fvm_sod.sh` que roda o Sod com N=400, CFL=0.5, Tf=0.2 e salva figuras em `data/artifacts/fvm_sod/`.

---

## Prompt 4 — Solução analítica do Sod (ground truth)

Implemente um solucionador analítico do Sod (clássico) para comparação (perfil em t fixo).

**Arquivos**
- `src/riemann_ml/exact/sod_exact.py`
  - `sod_exact_profile(x, t, left_state, right_state, gamma=1.4)`
  - Retorna ρ, u, p vetoriais; método clássico (rarefação à esq., contato, choque à dir.), robusto e verificado
- `tests/test_sod_exact.py` (valida pontos-chave: pressão intermediária, ordenação das ondas)
- Atualize `scripts/run_fvm_sod.sh` para plotar FVM vs exata (ρ,u,p).

---

## Prompt 5 — CLI com Typer + Hydra (simular, plotar)

Crie uma CLI `riemann-ml` com subcomandos para: `simulate-fvm`, `plot-sod`, `show-config`.

**Arquivos**
- `src/riemann_ml/__init__.py`
- `src/riemann_ml/cli.py` (Typer)
- `src/riemann_ml/configs/fvm.yaml` (Hydra)
- `scripts/cli_examples.sh` com exemplos de uso

---

## Prompt 6 — Geração de dados supervisionados (operador)

Crie um gerador de dataset (NPZ ou HDF5) com amostras de problemas de Riemann (variando ρ_L,p_L,ρ_R,p_R, u=0), resolvendo com FVM até `Tf`.

**Arquivos**
- `src/riemann_ml/data/generate.py`
  - `sample_riemann_ic(M, ranges, seed)` (amostragem LHS ou uniform)
  - `solve_and_store(ic_batch, N, CFL, Tf, out_path, x_grid)`
- `scripts/gen_dataset.sh` (gera, p.ex., 2000 amostras ρ,u,p em grade Nx=512, salva em `data/processed/sod_like.h5`)
- `tests/test_generate.py` (arquivo é criado, shapes corretos, sem NaN/Inf)

---

## Prompt 7 — PINN (TensorFlow 2.x)

Implemente uma PINN vanilla para Euler 1D (saídas ρ, m=ρu, E) com perdas de PDE, IC e BC. Use `tf.GradientTape` e ativação `tanh`.

**Arquivos**
- `src/riemann_ml/ml/pinn/model.py`
  - `PINN(config)` (MLP: {{8 camadas × 32}})
  - Funções auxiliares para montar resíduos `∂q/∂t + ∂f(q)/∂x` via AD
- `src/riemann_ml/ml/pinn/train.py`
  - Geração de pontos: colocação (PDE), IC (t=0), BC (x=0,1)
  - Otimizador Adam (lr={{1e-3}}), logs no TensorBoard, checkpoints
  - Ao final, avalia em malha (x,t=T) e salva perfis/figuras comparando com solução analítica do Sod
- `src/riemann_ml/configs/pinn.yaml` (n_pde, n_ic, n_bc, T=0.2, domínio [0,1], γ, seeds)
- `scripts/train_pinn_sod.sh`

**Critérios**
- Salvar curvas de loss; gráficos ρ,u,p vs exata; erro L2 relativo.

---

## Prompt 8 — FNO (PyTorch/neuraloperator)

Implemente treinamento de FNO 1D como operador da CI → solução em `t=T`. Use o dataset gerado (HDF5).

**Arquivos**
- `src/riemann_ml/ml/fno/dataset.py` (HDF5 → `torch.utils.data.Dataset`)
- `src/riemann_ml/ml/fno/model.py` (instancia FNO 1D de `neuraloperator`; {{4 camadas, 64 canais, 16 modos}})
- `src/riemann_ml/ml/fno/train.py`
  - Batches, Adam(lr={{1e-3}}), loss L2 relativo; validação hold-out; checkpoints
  - Inferência rápida em novas ICs; salva comparações ρ,u,p e métricas
- `src/riemann_ml/configs/fno.yaml`
- `scripts/train_fno.sh`

---

## Prompt 9 — Avaliação unificada e métricas

Implemente módulo de avaliação com: erro L2 relativo, erro na posição do choque, erro no patamar do contato, e geração de figuras lado-a-lado.

**Arquivos**
- `src/riemann_ml/eval/metrics.py` (`relative_l2`, `shock_location_error`, `contact_plateau_error`)
- `src/riemann_ml/eval/report.py` (gera `report.json` + PNGs)
- `scripts/eval_all.sh` (compara FVM, PINN e FNO no caso Sod e em ICs aleatórias do dataset)

---

## Prompt 10 — Testes unitários e de integração

Cubra o mínimo essencial: conversões prim/cons, fluxo positivo de energia, estabilidade do passo FVM curto, PINN forward pass, FNO forward pass e `metrics`.

**Arquivos**
- `tests/test_conversions.py`, `tests/test_flux.py`, `tests/test_metrics.py`, `tests/test_cli.py`
- Ajuste `pyproject.toml`/`pytest.ini` para `pytest -q --maxfail=1 --disable-warnings`

---

## Prompt 11 — Notebook(s) de exploração

Crie notebooks demonstrando: (1) Sod FVM vs exata, (2) PINN training curves e comparação final, (3) FNO generalização.

**Arquivos**
- `notebooks/01_fvm_sod.ipynb`
- `notebooks/02_pinn_sod.ipynb`
- `notebooks/03_fno_operator.ipynb`

Inclua células para carregar figuras/métricas geradas pelos scripts e comentar resultados.

---

## Prompt 12 — Empacotamento e entrada única (main)

Crie um `__main__.py` para permitir `python -m riemann_ml ...` espelhando a CLI.

**Arquivos**
- `src/riemann_ml/__main__.py` (chama Typer CLI)

---

## Prompt 13 — Pre-commit & CI básico

Adicione hooks de `pre-commit` (black, isort, flake8, end-of-file-fixer, trailing-whitespace) e um workflow GitHub Actions simples (lint + tests).

**Arquivos**
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`

---

## Prompt 14 — Docker & Makefile

Finalize Docker e adicione `Makefile` com alvos: `dev`, `test`, `fvm-sod`, `train-pinn`, `train-fno`, `eval`.

**Arquivos**
- `Dockerfile` (atualizar para rodar GPU opcional se disponível; caso contrário, CPU)
- `Makefile` com comandos padronizados

---

## Prompt 15 — Documentação (README completo)

Atualize `README.md` com:
- Diagrama da arquitetura do projeto
- Explicação curta: Euler 1D, problema de Riemann, Sod (com citação), por que PINN vs FNO
- Como reproduzir: 1) FVM, 2) gerar dataset, 3) treinar PINN, 4) treinar FNO, 5) avaliar
- Tabela de resultados esperados (placeholders para preencher após rodar)

---

## Prompt 16 — Sanidade final (scripts end-to-end)

Crie um script **único** que roda tudo “fim-a-fim” no caso Sod (FVM → gráficos; PINN treino curto → gráfico; FNO treino curto → gráfico; avaliação consolidada).

**Arquivos**
- `scripts/run_all_sod_demo.sh` (defina `set -euo pipefail`; salve tudo em `data/artifacts/demo_sod/`)

---

## Prompt 17 — Checagem de reprodutibilidade

Inclua seeds e `hydra` para registrar configs; salve `configs_used.yaml` junto com artefatos; serialize versões (`pip freeze`) em `ENVIRONMENT.txt`.

**Arquivos**
- `src/riemann_ml/utils/repro.py` (`set_global_seeds`, `save_environment`)
- Atualize treinadores/CLI para chamar essas funções
- Atualize scripts para salvar `ENVIRONMENT.txt`

---

## Prompt 18 — iteração extra (opcional)

Implemente:
- BCs alternativas (periódicas)
- Um limitador de fluxo simples (Minmod) para reduzir difusão no FVM (segunda ordem MUSCL opcional)
- Métrica de tempo de inferência (PINN vs FNO vs FVM)

**Arquivos**
- `src/riemann_ml/fvm/second_order.py` (MUSCL + Rusanov)
- `src/riemann_ml/eval/speed.py` (timer padronizado)

---

## Como validar rapidamente (cole no Codex após finalizar prompts 1–9)

> Gere um script `scripts/sanity_quick.sh` que:
> 1) roda `simulate-fvm` no Sod, plota vs exata;
> 2) treina a PINN por 500 épocas e plota;
> 3) gera dataset pequeno (100 amostras, Nx=256), treina FNO por 2 épocas e plota;
> 4) roda `eval_all.sh`;
> 5) imprime paths dos artefatos.

---

### Observações finais
- A sequência mantém alinhamento com boas práticas de engenharia (CLI, testes, Docker, reprodutibilidade) e com o escopo físico (Euler 1D, Sod) e de ML (PINN/FNO).
- Ajuste hiperparâmetros `{{...}}` conforme recursos disponíveis.
