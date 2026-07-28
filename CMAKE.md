# CMakeList

`CMakeLists.txt` is the configuration file used by CMake to build the C++ project.

CMake is a build-system generator. reads the instructions in `CMakeLists.txt` and creates the platform-specific build files needed to compile the project.

* `PRIVATE` — Applies the compiler options only to the `linear_regression` target.
* `PUBLIC` — Applies options to the target and to targets that link against it.
* `INTERFACE` — Applies options only to targets that depend on the target.
* `/O2` — allows speed-focused optimization in Microsoft Visual C++.
* `/W4` — allows a high level of compiler warnings in Microsoft Visual C++.
* `else()` — Uses the following configuration when the compiler is not MSVC, usually GCC or Clang.
* `-O0` — Disables compiler optimization.
* `-O1` — allows basic compiler optimization.
* `-O2` — allows strong optimization with a balance between speed and executable size.
* `-O3` — allows aggressive speed-focused optimization in GCC and Clang.
* `-Os` — Optimizes the program to reduce executable size.
* `-Og` — allows optimization while preserving a more useful debugging experience.
* `-Wall` — allows a broad set of common compiler warnings.
* `-Wextra` — allows additional warnings not included in `-Wall`.
* `-Wpedantic` — Warns when code does not strictly follow the selected ISO C++ standard.
* `set(CMAKE_CXX_STANDARD 17)` — Configures the project to use the C++17 standard.
* `set(CMAKE_CXX_EXTENSIONS OFF)` — Disables compiler-specific C++ extensions to improve portability.
