

# **Um Guia Abrangente para a Implementação de Solucionadores de Aprendizado de Máquina para o Problema de Riemann em Dinâmica dos Fluidos**

## **Parte I: Conceitos Fundamentais do Problema de Riemann em Dinâmica dos Gases**

Esta parte estabelece a base física e matemática do problema. É crucial para entender *o que* está sendo resolvido antes de discutir *como* resolvê-lo com aprendizado de máquina. Ela aborda diretamente os aspectos de "Dinâmica dos Fluidos" e "Problema de Riemann" da consulta do usuário.

### **1.1 As Equações de Euler 1D para Escoamento Compressível**

As equações que governam o movimento de um fluido compressível, invíscido (sem atrito interno) e não condutor de calor são conhecidas como as equações de Euler. Em uma dimensão espacial, essas equações são um sistema de leis de conservação hiperbólicas não lineares que surgem dos princípios fundamentais da física: a conservação de massa, momento e energia.1

#### **Forma Conservativa**

A forma mais natural e robusta para expressar essas leis, especialmente para escoamentos que podem desenvolver descontinuidades como ondas de choque, é a forma conservativa. Matematicamente, o sistema é escrito como uma única equação vetorial:

$$\\frac{\\partial \\mathbf{q}}{\\partial t} \+ \\frac{\\partial \\mathbf{f}(\\mathbf{q})}{\\partial x} \= 0$$
onde $t$ é o tempo, $x$ é a coordenada espacial, $\\mathbf{q}$ é o vetor de variáveis conservadas e $\\mathbf{f}(\\mathbf{q})$ é o vetor de fluxo correspondente.1

O vetor de estado $\\mathbf{q}$ é definido como:

$$\\mathbf{q} \= \\begin{pmatrix} \\rho \\\\ \\rho u \\\\ E \\end{pmatrix}$$

Aqui, $\\rho$ é a densidade do fluido (massa por unidade de volume), $u$ é a velocidade do fluido, $\\rho u$ é a densidade de momento linear, e $E$ é a densidade de energia total por unidade de volume.
O vetor de fluxo $\\mathbf{f}(\\mathbf{q})$ descreve o transporte das quantidades conservadas através do espaço e é dado por:

$$\\mathbf{f}(\\mathbf{q}) \= \\begin{pmatrix} \\rho u \\\\ \\rho u^2 \+ p \\\\ u(E \+ p) \\end{pmatrix}$$

Neste vetor, $p$ é a pressão do fluido. O primeiro componente, $\\rho u$, é o fluxo de massa. O segundo, $\\rho u^2 \+ p$, é o fluxo de momento, que tem duas contribuições: o transporte advectivo de momento ($\\rho u^2$) e a força exercida pela pressão ($p$). O terceiro, $u(E \+ p)$, é o fluxo de energia, que também consiste em transporte advectivo de energia ($uE$) e o trabalho realizado pela pressão ($up$).1

#### **Variáveis Primitivas e a Equação de Estado**

Embora a forma conservativa seja matematicamente poderosa, a interpretação física é muitas vezes mais clara em termos das variáveis primitivas: densidade ($\\rho$), velocidade ($u$) e pressão ($p$). O sistema de equações, no entanto, introduz quatro incógnitas ($\\rho$, $u$, $E$, $p$) mas fornece apenas três equações. Para fechar o sistema, é necessária uma relação adicional conhecida como equação de estado, que conecta as variáveis termodinâmicas.2

Para um gás ideal politrópico, a energia total $E$ é a soma da energia cinética e da energia interna:

$$E \= \\frac{1}{2}\\rho u^2 \+ \\rho e$$

onde $e$ é a energia interna específica (energia interna por unidade de massa). A equação de estado relaciona a pressão com a densidade e a energia interna específica:

$$p \= (\\gamma \- 1\) \\rho e$$

onde $\\gamma$ é o índice adiabático (ou razão dos calores específicos, $c\_p/c\_v$), uma constante que depende do gás (para o ar, $\\gamma \\approx 1.4$).2 Combinando essas duas relações, podemos expressar a pressão em termos das variáveis conservadas:

$$p \= (\\gamma \- 1\) \\left( E \- \\frac{1}{2}\\rho u^2 \\right)$$

Esta equação fecha o sistema, permitindo que o fluxo $\\mathbf{f}(\\mathbf{q})$ seja escrito inteiramente em função do vetor de estado $\\mathbf{q}$.

#### **Natureza Hiperbólica e Ondas Características**

A natureza matemática do sistema de equações de Euler é revelada pela análise de sua forma quasilinear. A forma conservativa pode ser reescrita como:

$$\\frac{\\partial \\mathbf{q}}{\\partial t} \+ \\mathbf{A}(\\mathbf{q}) \\frac{\\partial \\mathbf{q}}{\\partial x} \= 0$$
onde $\\mathbf{A}(\\mathbf{q}) \= \\frac{\\partial \\mathbf{f}}{\\partial \\mathbf{q}}$ é a matriz Jacobiana do fluxo. Um sistema de PDEs é classificado como hiperbólico se a matriz Jacobiana $\\mathbf{A}(\\mathbf{q})$ tiver autovalores reais e um conjunto completo de autovetores linearmente independentes.

Para as equações de Euler 1D, os autovalores da matriz Jacobiana são 2:

$$\\lambda\_1 \= u \- c, \\quad \\lambda\_2 \= u, \\quad \\lambda\_3 \= u \+ c$$

onde $c \= \\sqrt{\\frac{\\gamma p}{\\rho}}$ é a velocidade local do som no fluido. Como os autovalores são todos reais, o sistema é de fato hiperbólico. Esses autovalores, conhecidos como velocidades características, representam as velocidades com as quais a informação se propaga através do fluido.
Cada autovalor está associado a um autovetor correspondente, que define um "campo característico". Esses campos descrevem os tipos de ondas que podem existir na solução:

* Os campos associados a $\\lambda\_1 \= u \- c$ e $\\lambda\_3 \= u \+ c$ são chamados de *genuinamente não lineares*. Isso significa que a velocidade da onda depende da própria amplitude da onda, o que leva à formação de ondas de choque ou ondas de rarefação.
* O campo associado a $\\lambda\_2 \= u$ é chamado de *linearmente degenerado*. A velocidade da onda não depende da amplitude, resultando em ondas que se propagam sem mudar de forma, conhecidas como descontinuidades de contato.2

Essa estrutura de ondas é a chave para entender a solução do problema de Riemann.

### **1.2 O Problema de Riemann: Estrutura da Solução Exata**

O problema de Riemann, nomeado em homenagem a Bernhard Riemann, é um problema de valor inicial específico para um sistema de leis de conservação hiperbólicas. Ele é definido por dados iniciais que são constantes por partes, com uma única descontinuidade em um ponto, tipicamente $x=0$.5 A configuração inicial é:

$$\\mathbf{q}(x, 0\) \= \\begin{cases} \\mathbf{q}\_L & \\text{se } x \< 0 \\\\ \\mathbf{q}\_R & \\text{se } x \> 0 \\end{cases}$$
onde $\\mathbf{q}\_L$ e $\\mathbf{q}\_R$ são os vetores de estado constantes à esquerda e à direita da descontinuidade.

Apesar de sua aparente simplicidade, o problema de Riemann é de importância fundamental. Sua solução contém todas as ondas elementares (choques, rarefações e descontinuidades de contato) que podem surgir em soluções mais complexas de leis de conservação hiperbólicas. Por essa razão, é uma ferramenta teórica indispensável e a base para muitos métodos numéricos modernos, como os métodos de volumes finitos.3

#### **Auto-similaridade**

Uma propriedade notável da solução do problema de Riemann para as equações de Euler é que ela é auto-similar. Isso significa que a solução $\\mathbf{q}(x, t)$ depende apenas da razão $\\xi \= x/t$. A solução é constante ao longo de raios que emanam da origem no plano $x-t$.6 Essa propriedade reduz o problema de uma PDE em duas variáveis $(x, t)$ para um sistema de equações diferenciais ordinárias na variável de similaridade $\\xi$. A solução consiste em regiões de estado constante separadas por ondas que também são linhas retas no plano $x-t$.

#### **As Ondas Elementares**

A ruptura da descontinuidade inicial em $t=0$ gera um padrão de ondas que se propagam para a esquerda e para a direita. Para as equações de Euler, a solução geral do problema de Riemann consiste em três ondas separando quatro regiões de estado constante (incluindo os estados iniciais L e R). A estrutura é tipicamente uma onda se movendo para a esquerda (associada a $\\lambda\_1$), uma descontinuidade de contato no meio (associada a $\\lambda\_2$) e uma onda se movendo para a direita (associada a $\\lambda\_3$).7

* **Ondas de Choque:** Uma onda de choque é uma descontinuidade abrupta na qual todas as variáveis primitivas ($\\rho$, $u$, $p$) saltam de um valor para outro. Elas se movem com uma velocidade constante $s$ e são representadas como uma linha reta no plano $x-t$. As ondas de choque são compressivas; o fluido que passa por elas é desacelerado, e sua densidade e pressão aumentam.7 Matematicamente, as relações entre os estados antes e depois do choque são governadas pelas condições de salto de Rankine-Hugoniot, que são a forma integral das leis de conservação através da descontinuidade.3 Além disso, para garantir que a solução seja fisicamente realista (ou seja, que a entropia aumente através do choque), uma condição de entropia deve ser imposta para distinguir os choques físicos dos não físicos.7
* **Ondas de Rarefação (ou Expansão):** Ao contrário dos choques, uma onda de rarefação é uma onda suave e contínua na qual as propriedades do fluido mudam gradualmente. Ela se espalha ao longo do tempo, ocupando uma região em forma de leque no plano $x-t$, delimitada por características que se movem nas velocidades $u\_L \- c\_L$ e $u\_\* \- c\_\*$, por exemplo.6 Elas ocorrem quando o fluido está se expandindo e são caracterizadas por uma diminuição na densidade e na pressão.
* **Descontinuidades de Contato:** Esta é uma interface que separa dois estados de fluido que se movem com a mesma velocidade ($u$) e têm a mesma pressão ($p$), mas podem ter densidades ($\\rho$) e temperaturas diferentes. A descontinuidade de contato é simplesmente transportada (advectada) com a velocidade local do fluido, $u$. Ela é associada ao campo característico linearmente degenerado.2

A combinação específica de choques e rarefações que aparecem na solução depende inteiramente dos estados iniciais $\\mathbf{q}\_L$ e $\\mathbf{q}\_R$.

### **1.3 Estudo de Caso de Referência: O Problema do Tubo de Choque de Sod**

O problema do tubo de choque de Sod é o caso de teste canônico para validar códigos computacionais que resolvem as equações de Euler.11 É um problema de Riemann específico, proposto por Gary Sod em 1978, com condições iniciais que geram uma solução contendo todos os três tipos de ondas elementares: uma onda de rarefação se movendo para a esquerda, uma descontinuidade de contato se movendo para a direita e uma onda de choque se movendo para a direita.13

#### **Condições Iniciais**

O problema modela um tubo longo dividido ao meio por um diafragma. Em $t=0$, o diafragma é rompido instantaneamente. As condições iniciais para o problema padrão de Sod são dadas na Tabela 1\.

**Tabela 1: Condições Iniciais para o Problema do Tubo de Choque de Sod**

| Região | Densidade (ρ) | Pressão (p) | Velocidade (u) |
| :---- | :---- | :---- | :---- |
| Estado Esquerdo ($x \< 0.5$) | 1.0 kg/m³ | 1.0 Pa | 0.0 m/s |
| Estado Direito ($x \> 0.5$) | 0.125 kg/m³ | 0.1 Pa | 0.0 m/s |

*Nota: O índice adiabático é $\\gamma \= 1.4$. A descontinuidade inicial está em $x=0.5$.*

#### **Solução Analítica**

A solução analítica para o problema de Sod pode ser derivada e serve como a "verdade fundamental" (ground truth) para avaliar a precisão dos modelos numéricos e de aprendizado de máquina. A solução em um tempo específico, por exemplo $t=0.2$ s, exibe perfis distintos para densidade, pressão e velocidade. A visualização desses perfis é essencial para a análise qualitativa dos resultados.

As figuras a seguir mostram os perfis da solução analítica. Elas ilustram claramente a onda de rarefação suave à esquerda, a descontinuidade de contato (onde a densidade salta, mas a pressão e a velocidade são contínuas) e a onda de choque abrupta à direita. Qualquer método numérico ou de aprendizado de máquina bem-sucedido deve ser capaz de capturar essas características com alta fidelidade. A incapacidade de um método em representar essas estruturas, seja por excessiva difusão (borrando as descontinuidades) ou pela introdução de oscilações espúrias, indica suas limitações.11

## **Parte II: Paradigmas Clássicos vs. de Aprendizado de Máquina para EDPs Hiperbólicas**

Esta parte conecta o conhecimento existente do usuário, derivado de seu curso e do livro-texto 16, com as novas abordagens de aprendizado de máquina, fornecendo um contexto de alto nível para o projeto.

### **2.1 Uma Introdução aos Solucionadores Clássicos: O Método de Volumes Finitos**

Os métodos numéricos tradicionais para leis de conservação hiperbólicas, como os discutidos no livro-texto de referência 16, são projetados para lidar com soluções que contêm descontinuidades. Entre eles, o Método de Volumes Finitos (FVM) é particularmente proeminente e bem-sucedido.16 A ideia central do FVM é discretizar o domínio espacial em um conjunto de "volumes" ou "células" e trabalhar com as médias das quantidades conservadas em cada célula, em vez dos valores pontuais. A equação de conservação em sua forma integral é então aplicada a cada célula, resultando em uma atualização para a média da célula que depende dos fluxos através de suas interfaces.5

A equação de atualização para a média da célula $i$, $\\bar{\\mathbf{q}}\_i$, ao longo de um passo de tempo $\\Delta t$ é:

$$\\bar{\\mathbf{q}}\_i^{n+1} \= \\bar{\\mathbf{q}}\_i^n \- \\frac{\\Delta t}{\\Delta x} \\left( \\mathbf{F}\_{i+1/2} \- \\mathbf{F}\_{i-1/2} \\right)$$
onde $\\mathbf{F}\_{i\\pm1/2}$ são os fluxos numéricos nas interfaces da célula.

#### **O Papel do Solucionador de Riemann**

O passo crucial e o coração do FVM para leis de conservação é o cálculo desses fluxos numéricos, $\\mathbf{F}\_{i\\pm1/2}$. Como os estados nas células adjacentes $i$ e $i+1$ são, em geral, diferentes, a interface entre elas pode ser vista como uma descontinuidade. A evolução dessa descontinuidade é precisamente o que o problema de Riemann descreve. Portanto, a abordagem padrão é resolver (aproximadamente) um problema de Riemann em cada interface de célula, usando os estados das células vizinhas ($\\bar{\\mathbf{q}}\_i^n$ e $\\bar{\\mathbf{q}}\_{i+1}^n$) como os dados iniciais à esquerda e à direita.5 O fluxo resultante da solução deste problema de Riemann local é então usado como o fluxo numérico na interface.

Essa conexão direta demonstra por que o estudo do problema de Riemann é tão central para a dinâmica dos fluidos computacional (CFD) clássica. Solucionadores de Riemann aproximados, como os de Roe ou HLLC, são algoritmos sofisticados projetados para calcular esses fluxos de forma eficiente e precisa.

#### **Desafios**

Apesar de sua eficácia, os esquemas clássicos enfrentam desafios inerentes. Esquemas simples de primeira ordem, como o método de Lax-Friedrichs, são robustos, mas introduzem uma quantidade significativa de *difusão numérica*, que tende a borrar descontinuidades acentuadas como choques e contatos.14 Por outro lado, esquemas de ordem superior podem capturar descontinuidades de forma mais nítida, mas frequentemente introduzem *oscilações espúrias* não físicas perto delas (um análogo do fenômeno de Gibbs).16 Grande parte da pesquisa em métodos numéricos clássicos tem se concentrado no desenvolvimento de esquemas de "alta resolução" que tentam equilibrar esses dois efeitos, usando limitadores de fluxo para suprimir oscilações enquanto mantêm a nitidez.

### **2.2 A Emergência dos Solucionadores de Redes Neurais**

Nos últimos anos, uma abordagem radicalmente diferente para resolver EDPs emergiu do campo do aprendizado de máquina. Em vez de projetar esquemas de discretização explícitos baseados em expansões de Taylor ou formulações integrais, essa nova abordagem utiliza uma rede neural, um aproximador de função universal, para representar a própria solução da EDP.17

#### **Duas Filosofias Dominantes**

Dentro do paradigma de aprendizado de máquina para EDPs, duas filosofias principais se destacam, e ambas são relevantes para este projeto.

1. **Aprendizado de Função (ex: PINNs):** Nesta abordagem, a rede neural aprende a solução $\\mathbf{q}(x,t)$ para uma *única* instância da EDP. A rede não é treinada com dados de solução conhecidos. Em vez disso, ela é treinada para satisfazer a própria EDP, juntamente com suas condições iniciais e de contorno. A "física" da EDP é codificada diretamente na função de perda durante o treinamento.17
2. **Aprendizado de Operador (ex: DeepONet, FNO):** Esta é uma abordagem mais geral e ambiciosa. A rede neural aprende o *operador de solução* $\\mathcal{G}$, que é o mapeamento de uma classe de funções de entrada (por exemplo, condições iniciais) para as funções de saída correspondentes (as soluções da EDP ao longo do tempo). Uma vez treinado, um operador neural pode resolver instantaneamente qualquer nova instância do problema (com uma nova condição inicial da mesma classe) sem a necessidade de retreinamento.19

A Tabela 2 fornece uma visão geral estratégica, comparando os solucionadores clássicos com essas duas novas abordagens de aprendizado de máquina, destacando suas vantagens e desvantagens relativas.

**Tabela 2: Uma Visão Geral Comparativa dos Solucionadores de EDPs**

| Método | Princípio Primário | Dados de Treinamento | Velocidade de Inferência | Generalização | Tratamento de Choques |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Volumes Finitos** | Discretização, Solucionadores de Riemann | Nenhum | Lenta (iterativa) | Re-resolver para novas CIs | Métodos maduros e robustos |
| **PINN** | Minimização do Resíduo da EDP | Pontos de colocação (sem dados de solução) | Rápida | Re-treinar para novas CIs | Desafiador (PINN vanilla) |
| **DeepONet/FNO** | Aprendizado de Operador | Muitos pares (CI, Solução) | Extremamente Rápida | Generaliza para novas CIs | Dependente da qualidade dos dados de treinamento |

Esta comparação revela uma troca fundamental. Os métodos clássicos não requerem dados, mas são computacionalmente caros para cada nova simulação. Os PINNs evitam a necessidade de dados de solução, mas treinam um modelo para um único cenário. Os operadores neurais oferecem uma velocidade de inferência notável e generalização, mas ao custo de uma fase de treinamento intensiva em dados, que geralmente requer um solucionador clássico para gerar esses dados em primeiro lugar. O projeto proposto explorará essas diferentes facetas, aplicando-as ao desafiador problema de Riemann.

## **Parte III: Arquiteturas de Aprendizado de Máquina para Resolver as Equações de Euler**

Esta é a parte central técnica do relatório, fornecendo um mergulho profundo nos modelos específicos de aprendizado de máquina que o usuário implementará.

### **3.1 Redes Neurais Informadas pela Física (PINNs)**

As Redes Neurais Informadas pela Física (PINNs) representam uma abordagem para resolver problemas diretos e inversos envolvendo equações diferenciais. Em vez de depender puramente de dados, as PINNs incorporam o conhecimento das leis físicas subjacentes, codificadas como equações diferenciais, diretamente na função de perda da rede neural.17

#### **Princípio Central e Arquitetura**

A arquitetura de uma PINN é tipicamente uma rede neural feed-forward padrão, como uma Perceptron de Múltiplas Camadas (MLP). A rede recebe as coordenadas independentes do problema, neste caso o espaço e o tempo $(x, t)$, como entrada e produz a solução da EDP naquelas coordenadas, $\\mathbf{q}\_{pred}(x, t)$, como saída.17 A estrutura consiste em várias camadas ocultas com funções de ativação não lineares, como a tangente hiperbólica ($\\tanh$), que são cruciais para permitir que a rede aproxime as soluções complexas e não lineares das equações de Euler.

#### **A Função de Perda Informada pela Física**

A inovação central das PINNs reside em sua função de perda composta. A rede é treinada para minimizar uma perda que penaliza não apenas o desvio dos dados conhecidos (condições iniciais e de contorno), mas também o grau em que a saída da rede viola as próprias equações diferenciais. A função de perda total, $\\mathcal{L}\_{total}$, é uma soma ponderada de vários termos 17:

$$\\mathcal{L}\_{total} \= \\lambda\_{pde} \\mathcal{L}\_{pde} \+ \\lambda\_{ic} \\mathcal{L}\_{ic} \+ \\lambda\_{bc} \\mathcal{L}\_{bc}$$
onde $\\lambda\_{pde}$, $\\lambda\_{ic}$ e $\\lambda\_{bc}$ são pesos de hiperparâmetros.

* $\\mathcal{L}\_{ic}$ e $\\mathcal{L}\_{bc}$: Estas são as perdas das condições iniciais e de contorno. Elas são calculadas como o erro quadrático médio (MSE) entre a predição da rede e os valores verdadeiros conhecidos nessas regiões. Por exemplo, para a condição inicial $\\mathbf{q}(x, 0\) \= \\mathbf{q}\_0(x)$:
  $$\\mathcal{L}\_{ic} \= \\frac{1}{N\_{ic}} \\sum\_{i=1}^{N\_{ic}} \\| \\mathbf{q}\_{pred}(x\_i, 0\) \- \\mathbf{q}\_0(x\_i) \\|^2$$
* $\\mathcal{L}\_{pde}$: Esta é a perda do resíduo da EDP, o termo "informado pela física". Ela força a rede a satisfazer as equações de Euler em todo o domínio espaço-temporal. O resíduo $\\mathbf{r}(x, t)$ é definido como:
  $$\\mathbf{r}(x, t) := \\frac{\\partial \\mathbf{q}\_{pred}}{\\partial t} \+ \\frac{\\partial \\mathbf{f}(\\mathbf{q}\_{pred})}{\\partial x}$$
  A perda é então o MSE deste resíduo sobre um grande número de pontos de colocação amostrados aleatoriamente dentro do domínio:
  $$\\mathcal{L}\_{pde} \= \\frac{1}{N\_{pde}} \\sum\_{i=1}^{N\_{pde}} \\| \\mathbf{r}(x\_i, t\_i) \\|^2$$

#### **O Papel da Diferenciação Automática (AD)**

A tecnologia chave que torna as PINNs práticas é a diferenciação automática (AD). As bibliotecas de aprendizado de máquina modernas, como TensorFlow e PyTorch, podem calcular automaticamente as derivadas exatas da saída de uma rede neural em relação às suas entradas.17 Para calcular o resíduo da EDP, $\\mathbf{r}(x, t)$, precisamos das derivadas parciais $\\frac{\\partial \\mathbf{q}\_{pred}}{\\partial t}$ e $\\frac{\\partial \\mathbf{q}\_{pred}}{\\partial x}$ (para obter $\\frac{\\partial \\mathbf{f}}{\\partial x}$ através da regra da cadeia). A AD permite que essas derivadas sejam computadas com precisão de máquina diretamente do grafo computacional da rede, sem a necessidade de aproximações por diferenças finitas, que introduziriam erros de discretização na própria definição da perda.

#### **O Desafio das Descontinuidades para PINNs**

Apesar de sua elegância, as PINNs "vanilla" (padrão) enfrentam uma dificuldade fundamental ao tentar representar as soluções do problema de Riemann. A razão reside na natureza das descontinuidades. Uma rede neural, composta por uma série de transformações lineares e funções de ativação suaves (como $\\tanh$), é inerentemente uma função contínua e infinitamente diferenciável. Seus gradientes são sempre bem definidos e finitos.

Por outro lado, a solução verdadeira do problema de Riemann contém ondas de choque e descontinuidades de contato, que são matematicamente descontínuas. Nesses pontos, as derivadas da solução verdadeira são indefinidas ou infinitas (representadas por funções delta de Dirac). Quando uma PINN tenta aprender tal solução, o termo de perda do resíduo da EDP, $\\mathcal{L}\_{pde}$, explode numericamente nas proximidades da descontinuidade, pois a rede tenta, sem sucesso, replicar um gradiente infinito. Isso leva a gradientes patológicos durante o treinamento, fazendo com que o otimizador (por exemplo, Adam) falhe em convergir para uma solução fisicamente correta. O resultado típico é uma solução excessivamente suavizada que falha em capturar a nitidez do choque, ou a convergência para um mínimo local ruim que não representa a física corretamente.26

Essa limitação intrínseca motivou uma área de pesquisa ativa focada em adaptar as PINNs para problemas hiperbólicos. As abordagens incluem a introdução de um termo de viscosidade artificial nas equações para suavizar os choques 26, o uso de decomposição de domínio para isolar as descontinuidades (como nas cPINNs e XPINNs) 27, e a incorporação de princípios de solucionadores de Riemann clássicos (como o de Roe) na arquitetura ou na função de perda para guiar a rede a aprender as relações de salto corretas através das descontinuidades (Locally-Roe PINNs).26 Para este projeto, a implementação de uma PINN vanilla servirá como uma excelente linha de base para demonstrar e entender essa limitação fundamental.

### **3.2 Operadores Neurais: Aprendendo Mapeamentos Entre Espaços de Funções**

Enquanto as PINNs aprendem uma função que representa a solução para uma única instância de uma EDP, os operadores neurais adotam uma abordagem mais geral. Eles são projetados para aprender o próprio operador de solução, $\\mathcal{G}$, que mapeia uma função de entrada (por exemplo, uma condição inicial $\\mathbf{q}\_0(x)$) para uma função de saída (a solução $\\mathbf{q}(x, t)$).19 O objetivo é treinar uma rede que, uma vez treinada, possa prever a solução para *qualquer* condição inicial dentro de uma certa classe de funções, sem a necessidade de retreinamento.

#### **A Necessidade de um "Solucionador no Loop" para Geração de Dados**

A principal consequência dessa abordagem mais geral é a sua dependência de dados. Os operadores neurais são treinados de maneira supervisionada, o que significa que eles requerem um grande conjunto de dados de pares de entrada-saída, ou seja, pares de (condição inicial, solução correspondente).28 Para a maioria dos problemas de EDP, as soluções analíticas não estão disponíveis. Portanto, a geração desse conjunto de dados se torna um passo crítico e computacionalmente intensivo.

Na prática, isso significa que um solucionador numérico clássico e confiável (como um código de volumes finitos, possivelmente implementado usando os métodos do livro-texto 16) deve ser usado para gerar os dados de treinamento. O processo é o seguinte:

1. Definir uma distribuição ou uma gama de possíveis condições iniciais para o problema de Riemann (por exemplo, variando os valores de $\\rho\_L, p\_L, \\rho\_R, p\_R$).
2. Amostrar centenas ou milhares de condições iniciais dessa distribuição.
3. Para cada condição inicial amostrada, executar uma simulação de alta fidelidade usando o solucionador clássico para obter a solução correspondente em um ou mais instantes de tempo.
4. Armazenar cada par (condição inicial, solução) como um exemplo no conjunto de treinamento.30

Este processo revela uma relação simbiótica crucial: os solucionadores clássicos são usados para criar os dados que permitem que os operadores neurais se tornem *modelos substitutos* (surrogate models) rápidos. O custo computacional inicial da geração de dados é amortizado pela velocidade de inferência extremamente rápida do operador treinado, que pode ser ordens de magnitude mais rápido do que o solucionador original.30

### **3.3 Redes de Operadores Profundos (DeepONet)**

O DeepONet é uma das primeiras e mais influentes arquiteturas de operadores neurais. Sua estrutura única é projetada para aproximar operadores entre espaços de funções de dimensão infinita.28 A arquitetura consiste em duas redes neurais separadas que trabalham em conjunto:

* **Rede de Ramificação (Branch Net):** Esta rede processa a função de entrada, $\\mathbf{q}\_0(x)$. A função de entrada contínua é representada por seus valores em um número fixo de "pontos sensores". A rede de ramificação recebe esses valores e os codifica em um vetor de características latentes de dimensão finita. Essencialmente, ela aprende a extrair as características mais importantes da função de entrada.
* **Rede Tronco (Trunk Net):** Esta rede processa a localização no domínio da função de saída, ou seja, as coordenadas $(x, t)$ onde a solução deve ser avaliada. Ela mapeia essas coordenadas para um conjunto de funções de base.

A saída final do DeepONet para uma dada função de entrada $\\mathbf{q}\_0$ avaliada em um ponto $(x, t)$ é calculada pelo produto escalar entre as saídas da rede de ramificação e da rede tronco.28 Matematicamente, a aproximação do operador $\\mathcal{G}$ é:

$$\\mathcal{G}(\\mathbf{q}\_0)(x, t) \\approx \\sum\_{k=1}^{m} b\_k(\\mathbf{q}\_0) \\cdot t\_k(x, t)$$

onde $b\_k$ são as saídas da rede de ramificação e $t\_k$ são as saídas da rede tronco.
Para o problema de Riemann, a função de entrada para a rede de ramificação seria uma representação dos estados iniciais $\\mathbf{q}\_L$ e $\\mathbf{q}\_R$. A rede tronco receberia as coordenadas $(x, t)$, e a rede seria treinada em um grande conjunto de dados de diferentes problemas de Riemann para produzir a solução correta $\\mathbf{q}(x, t)$.34

### **3.4 Operadores Neurais de Fourier (FNO)**

O Operador Neural de Fourier (FNO) é outra arquitetura de operador neural poderosa que se mostrou particularmente eficaz para problemas de dinâmica dos fluidos.30 Em vez de usar convoluções locais como as CNNs ou a estrutura de ramificação/tronco do DeepONet, o FNO opera no domínio da frequência, aproveitando a eficiência da Transformada Rápida de Fourier (FFT).

#### **Arquitetura e Princípio Central**

A arquitetura do FNO consiste em uma sequência de "camadas de Fourier" 20:

1. **Elevação (Lifting):** A função de entrada, definida em uma grade espacial, é primeiro elevada a um espaço de canais de maior dimensão por meio de uma transformação linear pontual (semelhante a uma camada convolucional 1x1).
2. Camada de Fourier: Este é o bloco de construção central do FNO. Ele executa as seguintes operações:
   a. Aplica uma FFT para transformar os dados do domínio espacial para o domínio da frequência.
   b. Aplica uma transformação linear nos modos de Fourier de baixa frequência. Os modos de alta frequência são truncados (filtrados), o que atua como uma forma de regularização e impõe uma suavidade à solução aprendida. Esta operação é uma convolução global no domínio espacial, mas é calculada eficientemente como uma multiplicação pontual no domínio da frequência.
   c. Aplica uma FFT inversa para transformar os dados de volta ao domínio espacial.
3. **Não Linearidade:** Uma função de ativação não linear (como GeLU) é aplicada pontualmente após a camada de Fourier.
4. **Projeção:** Após várias camadas de Fourier, os dados são projetados de volta do espaço de canais de alta dimensão para a dimensão de saída desejada.

#### **Vantagens para a Dinâmica dos Fluidos**

A arquitetura do FNO é especialmente adequada para problemas como as equações de Euler por várias razões 30:

* **Eficiência:** O uso da FFT torna o FNO computacionalmente muito mais eficiente do que os métodos baseados em convoluções espaciais ou em grafos, especialmente para grades de alta resolução.
* **Dependências Globais:** A transformada de Fourier é inerentemente global; cada modo de Fourier depende de todos os pontos no domínio espacial. Isso permite que o FNO capture eficientemente as dependências de longo alcance que são características dos escoamentos de fluidos.
* **Invariância à Resolução:** Como o FNO aprende no domínio da frequência, o operador treinado pode ser avaliado em grades espaciais com resoluções diferentes daquelas usadas no treinamento. Essa propriedade, conhecida como *super-resolução de tiro zero* (zero-shot super-resolution), é uma vantagem significativa sobre os métodos clássicos e outras arquiteturas de aprendizado profundo, que são tipicamente dependentes da malha.20

Para o problema de Riemann, o FNO seria treinado em um conjunto de dados de pares (condição inicial, solução), semelhante ao DeepONet. Sua capacidade de capturar eficientemente a dinâmica global e sua invariância à resolução o tornam um candidato muito promissor para aprender o operador de solução das equações de Euler.

## **Parte IV: Um Guia Prático de Implementação para o Problema do Tubo de Choque de Sod**

Esta parte fornece um guia passo a passo para a construção e execução dos modelos de aprendizado de máquina. É rica em conselhos práticos e sugestões de estrutura de código para facilitar a implementação do projeto.

### **4.1 Ambiente e Frameworks**

A escolha de um ambiente de desenvolvimento adequado e das bibliotecas corretas é o primeiro passo para a implementação bem-sucedida do projeto.

* **Configuração do Ambiente:** Recomenda-se o uso de Python 3.x, que é o padrão de fato para o desenvolvimento de aprendizado de máquina. A gestão de pacotes e ambientes virtuais pode ser feita com ferramentas como conda ou venv para garantir a reprodutibilidade e evitar conflitos de dependência.
* **Bibliotecas de Aprendizado Profundo:** As duas principais opções para este projeto são TensorFlow e PyTorch.
  * **TensorFlow:** É um framework maduro com um ecossistema robusto. Sua API Keras de alto nível é particularmente amigável para iniciantes. Para a implementação de PINNs, a API tf.GradientTape do TensorFlow oferece uma maneira explícita e pedagogicamente clara de calcular as derivadas necessárias para a função de perda informada pela física.17
  * **PyTorch:** É amplamente popular na comunidade de pesquisa por sua flexibilidade e interface mais "pythônica". Seu sistema de diferenciação automática, torch.autograd, é igualmente poderoso para a implementação de PINNs.18 Existem também bibliotecas de alta qualidade construídas sobre o PyTorch para operadores neurais, como a neuraloperator.

Ambos os frameworks são perfeitamente adequados. As diretrizes de implementação aqui apresentadas serão conceitualmente aplicáveis a ambos, com uma ligeira inclinação para a sintaxe do TensorFlow/Keras nos exemplos de PINN, dada a sua clareza para fins de aprendizado.

* **Bibliotecas Essenciais:** Além do framework de aprendizado profundo, duas bibliotecas são indispensáveis:
  * NumPy: Para todas as operações numéricas, manipulação de arrays e pré-processamento de dados.
  * Matplotlib: Para a visualização dos resultados, como plotar os perfis de solução e compará-los com a solução analítica.

### **4.2 Geração e Gerenciamento de Dados**

A estratégia de geração de dados é fundamentalmente diferente para PINNs e para os modelos de aprendizado de operador.

#### **Para PINNs**

As PINNs não requerem dados de solução pré-calculados para o treinamento. Em vez disso, elas são treinadas em conjuntos de "pontos de colocação" amostrados do domínio espaço-temporal do problema.24 A estratégia de geração de dados envolve a criação de três conjuntos distintos de pontos:

1. **Pontos de Resíduo da EDP:** Amostre um grande número de pontos $(x\_r, t\_r)$ aleatoriamente de dentro do domínio computacional (por exemplo, $x \\in , t \\in \[0, 0.2\]$). Esses pontos serão usados para calcular a perda do resíduo da EDP, $\\mathcal{L}\_{pde}$. Uma amostragem por Hipercubo Latino pode fornecer uma cobertura mais uniforme do domínio do que a amostragem aleatória simples.
2. **Pontos da Condição Inicial:** Amostre pontos $(x\_{ic}, 0)$ na fatia de tempo inicial $t=0$. Nesses pontos, a solução é conhecida (as condições iniciais do tubo de choque de Sod). Eles serão usados para calcular a perda da condição inicial, $\\mathcal{L}\_{ic}$.
3. **Pontos das Condições de Contorno:** Amostre pontos $(x\_{bc}, t\_{bc})$ nas fronteiras espaciais do domínio (por exemplo, $x=0$ e $x=1$) para vários instantes de tempo. Eles serão usados para calcular a perda da condição de contorno, $\\mathcal{L}\_{bc}$. Para o problema do tubo de choque em um domínio suficientemente grande, condições de contorno periódicas ou de saída podem ser apropriadas.

#### **Para DeepONet e FNO (Aprendizado de Operador)**

Esses modelos requerem um conjunto de dados supervisionado de pares (condição inicial, solução). A criação deste conjunto de dados é uma etapa de pré-processamento crucial e computacionalmente intensiva.

**Procedimento para Criação do Conjunto de Dados:**

1. **Definir o Espaço de Condições Iniciais:** Determine as faixas para os parâmetros que definem um problema de Riemann. Por exemplo, para problemas do tipo tubo de choque, pode-se variar a densidade e a pressão iniciais nos estados esquerdo e direito, mantendo a velocidade inicial nula: $\\rho\_L \\in \[\\rho\_{min}, \\rho\_{max}\]$, $p\_L \\in \[p\_{min}, p\_{max}\]$, etc.
2. **Amostrar Condições Iniciais:** Gere centenas ou milhares de conjuntos de condições iniciais a partir dessas faixas. Novamente, a amostragem por Hipercubo Latino é uma boa escolha para garantir uma exploração eficiente do espaço de parâmetros.
3. **Resolver Numericamente:** Para cada condição inicial amostrada, resolva numericamente as equações de Euler usando um solucionador clássico robusto. Um método de volumes finitos de primeira ordem com um fluxo de Lax-Friedrichs, como descrito em Edsberg 16 e em outras referências 47, é uma escolha simples e estável para este fim. A simulação deve ser executada até um tempo final fixo, $T\_{final}$.
4. **Armazenar os Dados:** Salve cada simulação como um par de dados. A entrada será uma representação da condição inicial (por exemplo, os valores de $\\rho\_L, p\_L, \\rho\_R, p\_R$). A saída (o "rótulo") será a solução completa ($\\rho(x), u(x), p(x)$) no tempo $T\_{final}$, discretizada em uma grade espacial uniforme. Este conjunto de dados de (CI, Solução) será usado para treinar os modelos de operador neural.

### **4.3 Implementação e Treinamento do Modelo**

#### **Implementação de PINN (TensorFlow/Keras)**

1. **Definição da Rede:** Defina a PINN como uma subclasse de tf.keras.Model. A arquitetura pode consistir em várias camadas Dense com ativação tanh.18 A entrada da rede terá 2 neurônios (para $x$ e $t$) e a saída terá 3 neurônios (para $\\rho, \\rho u, E$).
2. **Construção da Função de Perda:** O passo mais complexo é a implementação de um loop de treinamento personalizado que calcula a perda informada pela física.
   * Dentro de um contexto tf.GradientTape, passe os pontos de colocação pela rede para obter $\\mathbf{q}\_{pred}$.
   * Use a tape para calcular as derivadas de primeira ordem de $\\mathbf{q}\_{pred}$ em relação a $t$ e $x$.
   * Implemente as funções para calcular a pressão $p$ a partir de $\\mathbf{q}\_{pred}$ e o vetor de fluxo $\\mathbf{f}(\\mathbf{q}\_{pred})$.
   * Use a regra da cadeia e as derivadas de $\\mathbf{q}\_{pred}$ para calcular $\\frac{\\partial \\mathbf{f}}{\\partial x}$.
   * Combine as derivadas para formar o resíduo da EDP e calcule $\\mathcal{L}\_{pde}$.
   * Calcule $\\mathcal{L}\_{ic}$ e $\\mathcal{L}\_{bc}$ usando os dados de condição inicial e de contorno.
   * Some as perdas (com pesos) para obter a perda total.
3. **Loop de Treinamento:** Use o otimizador Adam para aplicar os gradientes da perda total em relação aos pesos da rede.

#### **Implementação de FNO (PyTorch)**

1. **Aproveitar Bibliotecas Existentes:** Para um projeto de graduação, é altamente recomendável usar uma biblioteca existente que forneça uma implementação robusta do FNO, como a neuraloperator para PyTorch.42 Tentar implementar as camadas de Fourier com FFTs do zero é complexo e propenso a erros.43
2. **Preparação dos Dados:** Formate o conjunto de dados gerado na etapa 4.2 em tensores adequados para o PyTorch. Para o FNO, a entrada normalmente precisa ter uma dimensão de "canal" que inclua as coordenadas da grade espacial, além dos dados da condição inicial.20
3. **Definição e Treinamento do Modelo:**
   * Instancie o modelo FNO da biblioteca, especificando hiperparâmetros como o número de modos de Fourier a serem mantidos, a largura dos canais e o número de camadas de Fourier.
   * Escreva um loop de treinamento padrão do PyTorch que itera sobre o conjunto de dados em lotes (batches).
   * Em cada iteração, passe um lote de condições iniciais pelo modelo FNO, calcule a predição da solução e compare-a com a solução verdadeira do conjunto de dados usando uma função de perda como o erro relativo L2.20
   * Use o otimizador Adam para retropropagar o erro e atualizar os pesos do modelo.

**Tabela 3: Hiperparâmetros Recomendados para Experimentos Iniciais**

| Modelo | Arquitetura da Rede | Ativação | Otimizador | Taxa de Aprendizagem |
| :---- | :---- | :---- | :---- | :---- |
| **PINN** | 8 camadas ocultas, 32 neurônios cada | tanh | Adam | 1e-3 |
| **FNO** | 4 camadas de Fourier, 64 canais, 16 modos | GeLU | Adam | 1e-3 |

Esta tabela fornece um ponto de partida validado, economizando um tempo significativo de ajuste de hiperparâmetros e permitindo que o usuário obtenha um modelo funcional rapidamente antes de iniciar sua própria otimização.20

### **4.4 Avaliação e Visualização**

A avaliação rigorosa é crucial para entender o desempenho de cada modelo.

* **Avaliação Quantitativa:** A principal métrica de erro deve ser a norma do erro relativo L2, calculada entre a solução prevista pelo modelo para as condições iniciais do tubo de choque de Sod e a solução analítica conhecida. Isso fornecerá uma medida numérica única da precisão do modelo.
* **Avaliação Qualitativa:** A visualização é indispensável para entender *como* e *onde* os modelos falham. Gere gráficos que sobreponham os perfis previstos de densidade, pressão e velocidade com os perfis da solução analítica em um instante de tempo fixo (por exemplo, $t=0.2$ s). Essa comparação visual permitirá uma avaliação detalhada de quão bem cada modelo captura:
  * A suavidade e a extensão da onda de rarefação.
  * A nitidez e a localização correta da descontinuidade de contato.
  * A nitidez, a localização e a amplitude correta da onda de choque.

## **Parte V: Análise do Projeto e Direções Futuras**

Esta parte final incentiva o usuário a refletir criticamente sobre seus resultados e a considerar as implicações mais amplas de seu trabalho, transformando o projeto de uma mera implementação em uma investigação científica.

### **5.1 Análise Comparativa das Abordagens de ML**

Uma vez que os modelos tenham sido implementados e avaliados, a próxima etapa é realizar uma análise comparativa aprofundada de seus desempenhos. Os resultados esperados, com base na teoria e na literatura de pesquisa, fornecem um roteiro para essa análise.

* **PINN:** Espera-se que a implementação de uma PINN vanilla produza uma solução que é uma aproximação suave da solução verdadeira. A rede provavelmente terá grande dificuldade em capturar as descontinuidades acentuadas do problema de Sod. O perfil do choque será provavelmente "borrado" ao longo de vários pontos da grade, e a descontinuidade de contato pode ser quase inteiramente difundida. Isso não é uma falha na implementação, mas uma demonstração da limitação fundamental da arquitetura, como discutido na Parte III. A análise deve focar em quantificar essa difusão e explicar por que ela ocorre, ligando-a à incapacidade da rede de representar gradientes infinitos.
* **DeepONet e FNO:** O desempenho desses modelos de operador será fortemente correlacionado com a qualidade e a diversidade do conjunto de dados de treinamento. Se treinados com um conjunto de dados rico, abrangendo uma ampla gama de problemas de Riemann, espera-se que eles superem a PINN na captura da estrutura geral da solução.
  * **Velocidade de Inferência:** Uma das análises mais importantes será medir o tempo de inferência. Uma vez treinados, tanto o DeepONet quanto o FNO devem ser capazes de prever a solução para uma nova condição inicial em milissegundos, demonstrando sua vantagem como modelos substitutos rápidos em comparação com a execução de um solucionador de volumes finitos do zero.
  * **Precisão:** O FNO pode apresentar uma precisão ligeiramente superior à do DeepONet para este tipo de problema. Sua arquitetura, que opera no espaço de Fourier, é inerentemente boa em capturar as dependências globais e as estruturas de onda presentes em problemas de dinâmica dos fluidos. A análise deve comparar o erro L2 e os perfis visuais de ambos os modelos, discutindo qual deles captura melhor a localização e a nitidez das descontinuidades.

### **5.2 Possíveis Extensões do Projeto**

Um projeto final de excelência não apenas completa a tarefa proposta, mas também aponta para direções futuras interessantes. As seguintes extensões podem transformar este projeto em um trabalho de pesquisa mais substancial.

* **PINNs Avançadas para Choques:** Tendo estabelecido a linha de base com uma PINN vanilla, uma extensão natural é implementar ou pesquisar uma das variantes avançadas de PINN projetadas especificamente para lidar com descontinuidades. Por exemplo, investigar as Conservative PINNs (cPINNs), que usam uma abordagem de decomposição de domínio para impor a conservação do fluxo através das interfaces, pode levar a uma captura de choque muito melhor.27
* **Problemas Inversos:** O framework PINN é excepcionalmente adequado para problemas inversos. Uma extensão fascinante seria usar uma PINN para inferir um parâmetro físico a partir de dados esparsos. Por exemplo, pode-se assumir que o índice adiabático $\\gamma$ das equações de Euler é desconhecido. Ao fornecer à PINN algumas medições esparsas da densidade ou pressão ao longo do tempo, a rede pode ser treinada para encontrar não apenas a solução, mas também o valor de $\\gamma$ que melhor se ajusta aos dados.
* **Explorando Diferentes Problemas de Riemann:** A verdadeira força de um operador neural treinado reside em sua capacidade de generalização. Depois de treinar um FNO ou DeepONet em um conjunto de dados diversificado, pode-se testar seu desempenho em problemas de Riemann clássicos que não foram vistos durante o treinamento. Por exemplo, pode-se avaliar o modelo nas condições iniciais do problema do tubo de choque de Lax, que gera uma estrutura de onda diferente.26 Isso testaria a capacidade de interpolação e extrapolação do operador.
* **Problemas de Riemann Bidimensionais:** Como uma direção de pesquisa mais avançada, pode-se mencionar a extensão para problemas de Riemann em 2D. Esses problemas são significativamente mais complexos, pois a ruptura de uma descontinuidade inicial pode gerar padrões de onda intrincados, incluindo vórtices e instabilidades. A aplicação de modelos como o Geo-FNO, uma variante do FNO projetada para geometrias arbitrárias, a esses problemas é uma área de pesquisa de ponta.37

#### **Referências citadas**

1. The Euler equations of gas dynamics \- Clawpack, acessado em outubro 23, 2025, [https://www.clawpack.org/riemann\_book/html/Euler.html](https://www.clawpack.org/riemann_book/html/Euler.html)
2. Euler\_equations, acessado em outubro 23, 2025, [http://faculty.washington.edu/rjl/riemann\_book/Euler\_equations.html](http://faculty.washington.edu/rjl/riemann_book/Euler_equations.html)
3. Exact solution of the 1D riemann problem in Newtonian and relativistic hydrodynamics \- Revista Mexicana de Física, acessado em outubro 23, 2025, [https://rmf.smf.mx/ojs/index.php/rmf-e/article/download/4695/5956/14426](https://rmf.smf.mx/ojs/index.php/rmf-e/article/download/4695/5956/14426)
4. 1D Euler and MHD Shocks \- Philip Mocz, acessado em outubro 23, 2025, [https://pmocz.github.io/shocks/theory.html](https://pmocz.github.io/shocks/theory.html)
5. Riemann problem \- Wikipedia, acessado em outubro 23, 2025, [https://en.wikipedia.org/wiki/Riemann\_problem](https://en.wikipedia.org/wiki/Riemann_problem)
6. Introduction \- Clawpack, acessado em outubro 23, 2025, [http://www.clawpack.org/riemann\_book/html/Introduction.html](http://www.clawpack.org/riemann_book/html/Introduction.html)
7. Euler equations in 1-D, acessado em outubro 23, 2025, [https://math.tifrbng.res.in/\~praveen/notes/acfd2013/euler\_1d.pdf](https://math.tifrbng.res.in/~praveen/notes/acfd2013/euler_1d.pdf)
8. Shock–contact–shock solutions of the Riemann problem for dilute granular gas | Journal of Fluid Mechanics | Cambridge Core, acessado em outubro 23, 2025, [https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/shockcontactshock-solutions-of-the-riemann-problem-for-dilute-granular-gas/E7BC28E0762BD9D078E2483770949525](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/shockcontactshock-solutions-of-the-riemann-problem-for-dilute-granular-gas/E7BC28E0762BD9D078E2483770949525)
9. Waves, structures, and the Riemann problem for a system of hyperbolic conservation laws \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/pdf/2510.02070](https://arxiv.org/pdf/2510.02070)
10. The Riemann Problem The Riemann problem for (scalar or system of) conservation laws is the following ut \+ f(u) \= 0, u0(x) \= ul, acessado em outubro 23, 2025, [http://www.math.ualberta.ca/\~xinweiyu/527.1.08f/lec22.pdf](http://www.math.ualberta.ca/~xinweiyu/527.1.08f/lec22.pdf)
11. Sod shock tube \- Wikipedia, acessado em outubro 23, 2025, [https://en.wikipedia.org/wiki/Sod\_shock\_tube](https://en.wikipedia.org/wiki/Sod_shock_tube)
12. DISCLAIMER This report was prepared as an account of work sponsored by an agency of the United States Government. Neither the Un \- OSTI.GOV, acessado em outubro 23, 2025, [https://www.osti.gov/servlets/purl/2510952](https://www.osti.gov/servlets/purl/2510952)
13. Sod Shock Tube Problem \- Validation Case \- SimFlow CFD Software, acessado em outubro 23, 2025, [https://help.sim-flow.com/validation/sod-shock](https://help.sim-flow.com/validation/sod-shock)
14. 1.1.8.4. Sod's Test Problems: The Shock Tube Problem \- The Visual Room, acessado em outubro 23, 2025, [http://www.thevisualroom.com/01\_barba\_theory/sods\_test\_problem.html](http://www.thevisualroom.com/01_barba_theory/sods_test_problem.html)
15. How to get exact solution to Sod shock tube test? \- Physics Stack Exchange, acessado em outubro 23, 2025, [https://physics.stackexchange.com/questions/423758/how-to-get-exact-solution-to-sod-shock-tube-test](https://physics.stackexchange.com/questions/423758/how-to-get-exact-solution-to-sod-shock-tube-test)
16. Introduction to Computation and Modeling for Differential \-- Lennart Edsberg \-- ( WeLib.org ).pdf
17. Physics Informed Neural Networks (PINNs) \- Kaggle, acessado em outubro 23, 2025, [https://www.kaggle.com/code/newtonbaba12345/physics-informed-neural-networks-pinns](https://www.kaggle.com/code/newtonbaba12345/physics-informed-neural-networks-pinns)
18. Physics-informed Neural Networks (PINN), acessado em outubro 23, 2025, [https://i-systems.github.io/tutorial/KSNVE/220525/01\_PINN.html](https://i-systems.github.io/tutorial/KSNVE/220525/01_PINN.html)
19. DeepONet for Solving PDEs: Generalization Analysis in Sobolev Training \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/html/2410.04344v1](https://arxiv.org/html/2410.04344v1)
20. Solve PDE Using Fourier Neural Operator \- MATLAB & Simulink \- MathWorks, acessado em outubro 23, 2025, [https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html](https://www.mathworks.com/help/deeplearning/ug/solve-pde-using-fourier-neural-operator.html)
21. Multi-Step Physics-Informed Deep Operator Neural Network for Directly Solving Partial Differential Equations \- MDPI, acessado em outubro 23, 2025, [https://www.mdpi.com/2076-3417/14/13/5490](https://www.mdpi.com/2076-3417/14/13/5490)
22. nguyenkhoa0209/pinns\_tutorial: Tutorials for Physics-Informed Neural Networks \- GitHub, acessado em outubro 23, 2025, [https://github.com/nguyenkhoa0209/pinns\_tutorial](https://github.com/nguyenkhoa0209/pinns_tutorial)
23. Implementation of a Physics Informed Neural Network (PINN) written in Tensorflow v2, which is capable of solving Partial Differential Equations. \- GitHub, acessado em outubro 23, 2025, [https://github.com/kochlisGit/Physics-Informed-Neural-Network-PINN-Tensorflow](https://github.com/kochlisGit/Physics-Informed-Neural-Network-PINN-Tensorflow)
24. Using physics informed neural networks (PINNs) to solve parabolic PDEs \- Colab, acessado em outubro 23, 2025, [https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN\_Solver.ipynb](https://colab.research.google.com/github/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb)
25. Tensorflow tutorial for Physics Informed Neural Networks \- George Miloshevich, acessado em outubro 23, 2025, [https://georgemilosh.github.io/blog/2022/distill/](https://georgemilosh.github.io/blog/2022/distill/)
26. An approximate Riemann Solver Approach in Physics-Informed Neural Networks for hyperbolic conservation laws \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/html/2506.11959v2](https://arxiv.org/html/2506.11959v2)
27. Physics-Informed Neural Networks: Bridging the Divide Between Conservative and Non-Conservative Equations \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/html/2506.22413v1](https://arxiv.org/html/2506.22413v1)
28. DeepONet as a Multi-Operator Extrapolation Model: Distributed Pretraining with Physics-Informed Fine-Tuning \- UCLA Mathematics, acessado em outubro 23, 2025, [https://ww3.math.ucla.edu/wp-content/uploads/2025/02/2411.07239v1.pdf](https://ww3.math.ucla.edu/wp-content/uploads/2025/02/2411.07239v1.pdf)
29. An Efficient Method for Solving Two-Dimensional Partial Differential Equations with the Deep Operator Network \- MDPI, acessado em outubro 23, 2025, [https://www.mdpi.com/2075-1680/12/12/1095](https://www.mdpi.com/2075-1680/12/12/1095)
30. Fourier Neural Operator for Parametric Partial Differential Equations \- OpenReview, acessado em outubro 23, 2025, [https://openreview.net/forum?id=c8P9NQVtmnO](https://openreview.net/forum?id=c8P9NQVtmnO)
31. A Large Time Step Numerical Method for the Euler Equations using Deep Learning, acessado em outubro 23, 2025, [https://www.iccfd.org/iccfd11/assets/pdf/papers/ICCFD11\_Paper-4202.pdf](https://www.iccfd.org/iccfd11/assets/pdf/papers/ICCFD11_Paper-4202.pdf)
32. Learning the solution operator of parametric partial differential equations with physics-informed DeepONets \- PMC \- PubMed Central, acessado em outubro 23, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8480920/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8480920/)
33. Neural Basis Functions for Accelerating Solutions to High Mach Euler Equations \- OpenReview, acessado em outubro 23, 2025, [https://openreview.net/pdf?id=dvqjD3peY5S](https://openreview.net/pdf?id=dvqjD3peY5S)
34. Fusion-DeepONet: A Data-Efficient Neural Operator for Geometry-Dependent Hypersonic and Supersonic Flows | Request PDF \- ResearchGate, acessado em outubro 23, 2025, [https://www.researchgate.net/publication/396550142\_Fusion-DeepONet\_A\_Data-Efficient\_Neural\_Operator\_for\_Geometry-Dependent\_Hypersonic\_and\_Supersonic\_Flows](https://www.researchgate.net/publication/396550142_Fusion-DeepONet_A_Data-Efficient_Neural_Operator_for_Geometry-Dependent_Hypersonic_and_Supersonic_Flows)
35. CF DeepONet Deep Operator Neural Networks For Solving Compressible Flows \- Scribd, acessado em outubro 23, 2025, [https://www.scribd.com/document/893442131/CF-DeepONet-Deep-Operator-Neural-Networks-for-Solving-Compressible-Flows](https://www.scribd.com/document/893442131/CF-DeepONet-Deep-Operator-Neural-Networks-for-Solving-Compressible-Flows)
36. R-Adaptive DeepONet: Learning Solution Operators for PDEs with, acessado em outubro 23, 2025, [https://global-sci.com/pdf/article/91750/r-adaptive-deeponet-learning-solution-operators-for-pdes-with-discontinuous-solutions-using-an-r-adaptive-strategy.pdf](https://global-sci.com/pdf/article/91750/r-adaptive-deeponet-learning-solution-operators-for-pdes-with-discontinuous-solutions-using-an-r-adaptive-strategy.pdf)
37. Fourier Neural Operator with Learned Deformations for PDEs on General Geometries \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/html/2207.05209v2](https://arxiv.org/html/2207.05209v2)
38. Fourier Neural Operator Networks for Solving Reaction–Diffusion Equations \- MDPI, acessado em outubro 23, 2025, [https://www.mdpi.com/2311-5521/9/11/258](https://www.mdpi.com/2311-5521/9/11/258)
39. Fourier Neural Operator Networks: A Fast and General Solver for the Photoacoustic Wave Equation \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/pdf/2108.09374](https://arxiv.org/pdf/2108.09374)
40. Burgers Optimization with a PINN \- Physics-based Deep Learning, acessado em outubro 23, 2025, [https://physicsbaseddeeplearning.org/physicalloss-code.html](https://physicsbaseddeeplearning.org/physicalloss-code.html)
41. PINNs Part 1: Existing Work, acessado em outubro 23, 2025, [https://www.kylerofharrison.com/pinn\_p1/pinn\_p1.html](https://www.kylerofharrison.com/pinn_p1/pinn_p1.html)
42. neuraloperator/neuraloperator: Learning in infinite dimension with neural operators. \- GitHub, acessado em outubro 23, 2025, [https://github.com/neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)
43. anac0der/fno\_from\_scratch: Implementation of Fourier Neural Operator from scratch, acessado em outubro 23, 2025, [https://github.com/anac0der/fno\_from\_scratch](https://github.com/anac0der/fno_from_scratch)
44. Physics Informed Neural Networks (PINNs) for Solving System of ODEs \- A Beginner's Tutorial \- YouTube, acessado em outubro 23, 2025, [https://www.youtube.com/watch?v=gXv1SGoL04c](https://www.youtube.com/watch?v=gXv1SGoL04c)
45. Solve PDE Using Physics-Informed Neural Network \- MATLAB & Simulink \- MathWorks, acessado em outubro 23, 2025, [https://uk.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html](https://uk.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html)
46. Solving Burgers' Equation With PINN | by M.Hamxa \- Medium, acessado em outubro 23, 2025, [https://medium.com/@hamxa26/solving-burgers-equation-with-neural-network-e7405c95b2d2](https://medium.com/@hamxa26/solving-burgers-equation-with-neural-network-e7405c95b2d2)
47. The advection equation \- Numerical Methods for Engineers, acessado em outubro 23, 2025, [http://lrhgit.github.io/tkt4140/allfiles/digital\_compendium/.\_main021.html](http://lrhgit.github.io/tkt4140/allfiles/digital_compendium/._main021.html)
48. Four-Quadrant Riemann Problem for a 2 × 2 System Involving Delta Shock \- MDPI, acessado em outubro 23, 2025, [https://www.mdpi.com/2227-7390/9/2/138](https://www.mdpi.com/2227-7390/9/2/138)
49. \[2505.20361\] Solving Euler equations with Multiple Discontinuities via Separation-Transfer Physics-Informed Neural Networks \- arXiv, acessado em outubro 23, 2025, [https://arxiv.org/abs/2505.20361](https://arxiv.org/abs/2505.20361)
