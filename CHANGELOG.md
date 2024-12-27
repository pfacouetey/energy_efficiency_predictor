# CHANGELOG


## v0.1.1 (2024-12-27)

### Bug Fixes

- **code**: Add automatic release versionning
  ([`b6e83f6`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/b6e83f6d4bd40889626e188d1b9e043207fa9acd))


## v0.1.0 (2024-12-26)

### Chores

- **structure**: Initialize project
  ([`81f714d`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/81f714d9f98b0c52815e2e8ec7679f84927f94de))

### Features

- **data**: Define data loading process and configure adequate tests
  ([`0a81f34`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/0a81f346cf1e09ef99554556107feb801e5ff11f))

- **engineering**: Proceed to features scaling, and configure adequate tests
  ([`7ddbbd3`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/7ddbbd3da7668b3c8d0bbd6a7bb6f5c8360b6533))

- **form**: Add decsion tree regressor, duplicate main notebook and update readme file
  ([`deda0a2`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/deda0a2a8573cc004a3bcaa28b50f9c16d7d0d2a))

- **linear-models**: Perform pca regression
  ([`58c687b`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/58c687bba8ef1a3949b4261d3fbb24abc427a690))

- **models**: Define pca regression and its regularized version
  ([`3a61a2b`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/3a61a2b122c67983bab4f9d8e00dc91ae601b80d))

- **notebook**: Add more analysis on training, validation, test features and targets
  ([`bf4ce5e`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/bf4ce5eee327bf24a02bb454f12293aa22d8a364))

- **notebook**: Perform analysis between features and targets
  ([`01d3ade`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/01d3aded3443c3826035651709163621dfec8f44))

### Refactoring

- Add newline at end of file
  ([`325fb3f`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/325fb3fb37752fc59fb2cd921f65e15d437d1303))

It is recommended to put a newline at the end of the file.

- Add newline at end of file
  ([`3f051fb`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/3f051fb51513f62a5697921f918179c92043cc1e))

It is recommended to put a newline at the end of the file.

- Refactor unnecessary `else` / `elif` when `if` block has a `return` statement
  ([`522a816`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/522a816c637bbe7d4775ff5ec76188b0b7cc1719))

The use of `else` or `elif` becomes redundant and can be dropped if the last statement under the
  leading `if` / `elif` block is a `return` statement. In the case of an `elif` after `return`, it
  can be written as a separate `if` block. For `else` blocks after `return`, the statements can be
  shifted out of `else`. Please refer to the examples below for reference.

Refactoring the code this way can improve code-readability and make it easier to maintain.

- Refactor unnecessary `else` / `elif` when `if` block has a `return` statement
  ([`eff55ce`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/eff55ce6de8d062083901b07adb984cea91b11f2))

The use of `else` or `elif` becomes redundant and can be dropped if the last statement under the
  leading `if` / `elif` block is a `return` statement. In the case of an `elif` after `return`, it
  can be written as a separate `if` block. For `else` blocks after `return`, the statements can be
  shifted out of `else`. Please refer to the examples below for reference.

Refactoring the code this way can improve code-readability and make it easier to maintain.

- Remove unnecessary lambda expression
  ([`b54c19c`](https://github.com/pfacouetey/energy_efficiency_predictor/commit/b54c19cfb43bda5b6f35c5d087b5049e0f139308))

A lambda that calls a function without modifying any of its parameters is unnecessary. Python
  functions are first-class objects and can be passed around in the same way as the resulting
  lambda. It is recommended to remove the lambda and use the function directly.
