# cutlass-tutorial-kr

[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)의 **CuTe DSL** 공식 교육용 노트북을
한국어로 번역하고, Google Colab에서 직접 실행한 결과와 유의점을 덧붙인 Jupyter 노트북
모음입니다.

원문 기준: [`NVIDIA/cutlass` examples/python/CuTeDSL/notebooks](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) (BSD-3-Clause). 각 노트북 파일이 인용하는 정확한 커밋은 해당 노트북 상단 주석 참고.

## 노트북

| 파일 | 내용 | Colab |
|---|---|---|
| [`notebooks/01_hello_world.ipynb`](notebooks/01_hello_world.ipynb) | `@cute.kernel`/`@cute.jit` 기초, 스레드 인덱싱, `cute.printf` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/master/notebooks/01_hello_world.ipynb) |
| [`notebooks/02_control_flow.ipynb`](notebooks/02_control_flow.ipynb) | 메타 루프 vs 스테이지드 루프, `range_constexpr`/`range`, `const_expr` 분기 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/master/notebooks/02_control_flow.ipynb) |
| [`notebooks/03_diagnostics.ipynb`](notebooks/03_diagnostics.ipynb) | 컴파일러 진단 `warnings{...}`/`remarks{...}`, 심각도 3단계, `CompilerDiagnosticError`, `CUTE_DSL_ARCH` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/master/notebooks/03_diagnostics.ipynb) |
| [`notebooks/04_zero_cost_abstraction.ipynb`](notebooks/04_zero_cost_abstraction.ipynb) | 클래스·다형성이 트레이스 시점 파이썬이라 컴파일되며 사라짐을 PTX로 확인 (`KeepPTX`, `__ptx__`) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/master/notebooks/04_zero_cost_abstraction.ipynb) |
| [`notebooks/05_array_concepts.ipynb`](notebooks/05_array_concepts.ipynb) | 원문 `2_primitives/01_array_concepts` — `cutlass.Array`, 네 가지 GPU 메모리 공간(local/shared/global/constant), 슬라이스 = 벡터화 load/store | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/master/notebooks/05_array_concepts.ipynb) |
| [`notebooks/06_vector_concepts.ipynb`](notebooks/06_vector_concepts.ipynb) | 원문 `2_primitives/02_vector_concepts` — `cutlass.Vector`(레지스터) vs `cutlass.Array`(메모리), SIMD 산술, `vector.where`/`full`, `reduce("add")` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/master/notebooks/06_vector_concepts.ipynb) |

원본 공식 커리큘럼은 `1_dsl_features`(DSL 기능) → `2_primitives`(Array/Vector/TMA) →
`3_kernels`(softmax, stencil, Blackwell MMA) 순으로 이어집니다. 이 저장소는 `1_dsl_features`의 4개를 모두 번역했고, `2_primitives`는 `02_vector_concepts`까지
(저장소 안 번호는 이어서 05, 06) 진행했습니다. `3_kernels`는 아직 손대지 않았습니다.

## 실행 환경

- **GPU**: L4(Colab Pro/Pro+)에서 검증. `python/CuTeDSL/cutlass/base_dsl/enums.py`의
  `Arch` enum이 `sm_80`(Ampere)부터 시작하고 `sm_75`(Turing/T4)가 파일 전체에 아예 없음을
  코드로 확인했습니다 — 무료 T4 티어는 아키텍처상 배제되고, sm_80 이상(A100/L4 등)이
  필요합니다.
- **노트북3 예외**: 섹션 1(`elect.sync`/`cp.async.bulk`)은 sm_90 이상 명령어라 L4(sm_89)에선 그대로 컴파일이 실패합니다(실측). 그래서 노트북3은 `CUTE_DSL_ARCH=sm_90a`로 *컴파일 타깃만* 덮어씌워 L4에서 검증했습니다(컴파일만 하고 실행은 안 함). 실제 sm_90 GPU 검증은 아직 못 했습니다.
- **패키지**: `nvidia-cutlass-dsl==4.8.0` (기본 설치는 CUDA 12.x 바이너리, `cu13` extra는
  CUDA 13.x 바이너리로 교체됨 — 실측은 Colab CUDA 13.0 환경에서 기본 설치로 검증).

## 라이선스

이 저장소의 코드·설명은 [MIT License](LICENSE)로 배포합니다.
원문 발췌 부분의 출처·라이선스는 [NOTICE](NOTICE)를 참고하세요.
