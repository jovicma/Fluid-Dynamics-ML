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
4. Use a CLI (a ser implementada) para orquestrar experimentos reprodutíveis.

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
