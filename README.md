# cutlass-tutorial-kr

[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)의 **CuTe DSL** 공식 교육용 노트북을
한국어로 번역하고, Google Colab에서 직접 실행한 결과와 유의점을 덧붙인 Jupyter 노트북
모음입니다.

원문 기준: [`NVIDIA/cutlass` examples/python/CuTeDSL/notebooks](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/notebooks) (BSD-3-Clause)

## 노트북

| 파일 | 내용 | Colab |
|---|---|---|
| [`notebooks/01_hello_world.ipynb`](notebooks/01_hello_world.ipynb) | `@cute.kernel`/`@cute.jit` 기초, 스레드 인덱싱, `cute.printf` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexxony/cutlass-tutorial-kr/blob/main/notebooks/01_hello_world.ipynb) |

원본 공식 커리큘럼은 `1_dsl_features`(DSL 기능) → `2_primitives`(Array/Vector/TMA) →
`3_kernels`(softmax, stencil, Blackwell MMA) 순으로 이어집니다. 이 저장소는 아직
`1_dsl_features/01_hello_world`만 번역했습니다.

## 실행 환경

- **GPU**: L4(Colab Pro/Pro+)에서 검증. 원문은 "any CUDA GPU"라 명시하므로 T4 등 다른
  GPU에서도 동작할 가능성이 높지만, 이 저장소는 L4로만 실측했습니다.
- **패키지**: `nvidia-cutlass-dsl` (CUDA 12.x 기본, CUDA 13.x는 `cu13` extra).

## 라이선스

이 저장소의 코드·설명은 [MIT License](LICENSE)로 배포합니다.
원문 발췌 부분의 출처·라이선스는 [NOTICE](NOTICE)를 참고하세요.
