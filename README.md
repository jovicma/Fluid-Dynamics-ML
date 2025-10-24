# Riemann-ML

```
  ____  _                                      _ __  __ _
 |  _ \(_)_ __ ___  _ __   ___ _ __ ___   __ _| |  \/  | |
 | |_) | | '__/ _ \| '_ \ / _ \ '_ ` _ \ / _` | | |\/| | |
 |  _ <| | | | (_) | | | |  __/ | | | | | (_| | | |  | | |___
 |_| \_\_|_|  \___/|_| |_|\___|_| |_| |_|\__,_|_|_|  |_|_____|
                     riemann-ml
```

Riemann-ML é uma plataforma educacional para explorar a dinâmica de fluidos 1D com foco no problema de Riemann. O projeto integra métodos numéricos clássicos (FVM e soluidores exatos) e modelos de aprendizado de máquina (PINNs e operadores neurais) para geração, modelagem e avaliação de soluções.

## Arquitetura em alto nível

```
┌───────────────────────────────┐      ┌─────────────────────────────────┐
│ Entrada de Configuração (CLI) │──────▶     Pipelines (Make/CLI)       │
└───────────────────────────────┘      └──────────────┬──────────────────┘
                                                      │
            ┌─────────────┬─────────────┬─────────────┴─────────────┬──────────────┐
            │             │             │                           │              │
            ▼             ▼             ▼                           ▼              ▼
┌────────────────┐ ┌──────────────┐ ┌──────────────┐      ┌──────────────────┐ ┌───────────────┐
│ Core (Euler)   │ │ FVM Solver   │ │ Exact Solver │      │ PINN (TensorFlow) │ │ FNO (PyTorch) │
└────────▲───────┘ └──────▲───────┘ └──────▲───────┘      └──────────▲───────┘ └───────────────┘
         │                │                │                         │
         │                │                │                         │
         └────────────┬───┴──────┬─────────┴────────────┬────────────┘
                      ▼          ▼                      ▼
             ┌────────────────────────┐        ┌────────────────────────┐
             │ Geração de dados (HDF5)│        │ Avaliação (metricas +  │
             │ + notebooks exploratórios │      │ visualizações)         │
             └───────────────┬─────────┘        └─────────────┬────────┘
                             ▼                               ▼
                  ┌───────────────────────┐       ┌────────────────────────┐
                  │ Artefatos (plots,    │       │ Relatórios / Notebook  │
                  │ checkpoints, métricas)│      │ final (pipeline_run)   │
                  └───────────────────────┘       └────────────────────────┘
```

## Requisitos

- Python 3.11 ou superior
- Git
- Dependências listadas em `requirements.txt`

## Instalação

### Via Hatch (recomendado)

```bash
pip install --upgrade hatch
hatch env create
```

### Via pip + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso rápido

```bash
# Preparação do ambiente de desenvolvimento
scripts/dev_install.sh

# Rodar formatadores e linters
scripts/format.sh

# Executar a suíte de testes
scripts/run_tests.sh
```

## CLI

Após preparar o ambiente, instale o pacote em modo editável (ou dentro do ambiente Hatch) para expor o executável `riemann-ml`:

```bash
pip install -e .                # alternativa geral
# ou, se estiver usando hatch:
hatch run pip install -e .
```

Principais comandos:

- `riemann-ml show-config` — imprime a configuração Hydra usada pelos solvers.
- `riemann-ml simulate-fvm --output data/artifacts/cli_run/solution.npz` — roda o esquema de volumes finitos e salva os resultados.
- `riemann-ml plot-sod --output-dir data/artifacts/cli_run/plots` — gera gráficos comparando FVM e solução analítica.

Se preferir evitar a instalação editável, os comandos também funcionam via `python -m riemann_ml.cli ...` dentro do repositório.

## Estrutura do projeto

```
src/riemann_ml/
├── core/         # Entidades base (malhas, estados termodinâmicos, utilitários numéricos)
├── fvm/          # Implementações dos esquemas de volumes finitos
├── exact/        # Solução analítica do problema de Riemann (ex.: Sod)
├── data/         # Pipelines de geração e gerenciamento de datasets
├── eval/         # Métricas e relatórios de avaliação
├── ml/
│   ├── pinn/     # Redes PINN (TensorFlow)
│   └── fno/      # Operadores de Fourier (PyTorch + neuraloperator)
├── utils/        # Funções auxiliares e log
└── configs/      # Configurações Hydra/OmegaConf

data/
├── raw/          # Dados brutos (não versionado)
├── processed/    # Features prontas
└── artifacts/    # Modelos e saídas finais
```

## Fluxo de trabalho sugerido

1. Gere os dados de referência usando os soluidores FVM e exatos.
2. Treine os modelos PINN/FNO com as configurações definidas em `src/riemann_ml/configs`.
3. Avalie as previsões com as métricas reunidas em `src/riemann_ml/eval`.
4. Use a CLI `riemann-ml` (ou `python -m riemann_ml.cli`) para orquestrar experimentos reprodutíveis.

## Contexto físico e modelo

- **Equações de Euler 1D:** conjunto hiperbólico que modela conservação de massa, quantidade de movimento e energia em um gás ideal. As variáveis conservativas \((\rho, \rho u, E)\) evoluem segundo \( \partial_t q + \partial_x f(q) = 0 \).
- **Problemas de Riemann:** condições iniciais por partes (descontinuidade em \(x_0\)) que produzem ondas de choque, rarefações e contatos.
- **Caso clássico de Sod:** problema introduzido por Gary A. Sod [[Sod, 1978](https://doi.org/10.1016/0021-9991(78)90023-2)], com estados \( (\rho_L,u_L,p_L)=(1.0,0.0,1.0) \) e \( (\rho_R,u_R,p_R)=(0.125,0.0,0.1) \). Serve de teste de convergência para esquemas numéricos.
- **PINN vs FNO:** PINN resolve a PDE diretamente impondo os resíduos via `tf.GradientTape`, útil quando dados são escassos, mas pode exigir tuning pesado. FNO aprende um operador mapeando condições iniciais para o estado final, escalando bem em conjuntos grandes (porém dependente de dados de treino).

## Reproduzindo os experimentos

1. **FVM Sod** – `make fvm-sod`: executa o solver Rusanov e grava os perfis em `data/artifacts/cli_sod`.
2. **Gerar dataset** – `scripts/gen_dataset.sh` (ou `NUM_SAMPLES=2000 ...`): produz HDF5 com soluções para diversas ICs.
3. **Treinar PINN** – `make train-pinn`: usa configuração `pinn.yaml` (TensorFlow).
4. **Treinar FNO** – `make train-fno`: consome dataset HDF5 (PyTorch/neuraloperator).
5. **Avaliar** – `make eval` (ou `scripts/eval_all.sh`): gera métricas/figuras em `data/artifacts/eval`.

Para uma validação rápida, utilize `scripts/sanity_quick.sh`, que encadeia versões curtas de cada etapa.

## Resultados esperados

Preencha a tabela após rodar `make eval` com os artefatos produzidos:

| Modelo | relative_l2_ρ | relative_l2_u | relative_l2_p | Erro choque (Δx) | Erro contato |
|--------|---------------|---------------|---------------|------------------|--------------|
| FVM    | *(preencher)* | *(preencher)* | *(preencher)* | *(preencher)*     | *(preencher)*|
| PINN   | *(preencher)* | *(preencher)* | *(preencher)* | *(preencher)*     | *(preencher)*|
| FNO    | *(preencher)* | *(preencher)* | *(preencher)* | *(preencher)*     | *(preencher)*|

## Docker

```bash
docker build -t riemann-ml .
docker run --rm -it -v $(pwd):/app riemann-ml
```

## Contribuição

1. Crie uma branch a partir de `main`.
2. Execute `scripts/format.sh` antes de submeter.
3. Garanta que `scripts/run_tests.sh` passa sem falhas.

## Licença

Defina a licença do projeto conforme necessário (ex.: MIT, BSD ou outra).
