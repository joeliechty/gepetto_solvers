# Robotics & ML Agent Engineering Guidelines

**Mission**
You are an expert AI software engineer assisting with a robotics and machine learning codebase. Your goal is to rapidly iterate on research while maintaining strict, production-grade engineering standards. Eliminate business context or narrative fluff; focus strictly on technical constraints, clean architecture, and execution.

### 1. Repository Structure & Compartmentalization
*   **The `src` Layout:** All core package code must be housed within a `src/<package_name>/` directory. Never pollute the repository root.
*   **Granular Compartmentalization:** Ban mega-files. Break logic down into small, single-purpose methods and tightly scoped files. This ensures isolated debugging and makes finding specific modules (e.g., a specific causal transformer layer or GTSAM factor) instantaneous.
*   **Separation of Concerns:** Place executable utilities in `scripts/` and tests in `tests/` or colocated with the target module.
*   **The Three-Tier Architecture:** Use `core/` (foundational infrastructure), `projects/` (isolated active research; no cross-dependencies), and `experimental/` (sandboxes).

### 2. C++/Python Mixed Codebases
*   **Clean Architectural Boundaries:** Strictly separate pure C++ logic from the binding boilerplate and the Python wrapper. Store core C++ algorithms in `include/` and `src/` directories. Store the `PYBIND11_MODULE` or `NB_MODULE` definitions in a dedicated `src/bindings/` directory. Store the Python wrapper package in `python/<package_name>/`.
*   **Build Stack:** Use `scikit-build-core` as the `pyproject.toml` build backend to bridge Python packaging with CMake. A `CMakeLists.txt` file (requiring CMake 3.15 or higher) is strictly required to compile the C++ source and bindings. Do not use `setup.py`.
*   **Binding Choices:** Default to `nanobind` for new, performance-critical codebases to ensure faster compilation and smaller binaries. Use `pybind11` only if integrating with legacy ecosystems.
*   **Memory & Type Safety:** Explicitly manage memory ownership when passing structures across the C++/Python boundary (e.g., via `take_ownership` or `reference` policies). Avoid deep copying standard template library (STL) containers; pass memory buffers by reference for high-performance data to prevent overhead. Do not return raw pointers to Python if the object is internally managed by a smart pointer.

### 3. README-Driven Development & Maintenance
*   **README as the Command Center:** The README is the starting point for planning any repository update. Before writing code for a new feature, model, or refactor, outline the proposed architecture, file changes, and execution plan directly in the README.
*   **High-Fidelity Documentation:** Keep READMEs highly detailed and strictly up-to-date with current installation steps, environment setups (e.g., Apptainer/Docker container instructions, ROS 2 workspaces), and execution commands.
*   **Git Hygiene & Security:** Proactively maintain the `.gitignore` file. Never commit personal information, API keys, hardcoded absolute local paths, or any sensitive credentials. Do not commit weights, large datasets, or localized visualization artifacts.

### 4. Debugging & State Visibility
*   **Aggressive Introspection:** When debugging, print verbose diagnostic information. Do not just print errors; explicitly print tensor shapes, data types, min/max bounds, and intermediate states (especially when debugging Riemannian manifold operations, coordinate frame shifts, or flow matching decoders).
*   **Visual Debugging:** Always default to visual validation for robotics and ML data. Generate plots to verify 3D trajectories, optimization loss curves, bounding boxes, or distribution matches before assuming the underlying math is correct.
*   **Human-Centric Interfaces:** Ensure every API, logging output, and configuration interface is easily human-readable and writable. Avoid opaque serialized objects when a clear, structured text representation will suffice.
*   **Plan Debugging Steps:** For human programmers poor debugging wastes time. For agents poor debugging wastes tokens. In both cases it leads to messier code. Make small plans for how to isolate bugs and fix them rather than blindly running tests in an effor to uncover the bugs. Think carefully about what information would be useful for isolating the problem in the code.

### 5. Dependency Management & Configuration
*   **Single Source of Truth:** Use a PEP-compliant `pyproject.toml` file. Do not use `setup.py` or `setup.cfg`.
*   **ML Configuration Paradigm:** Use pure Python configuration (e.g., `ml_collections`) or typed dataclasses (`tyro`/`Hydra`) for deep learning parameters.
*   **Robotics/Hardware Paradigm:** Use YAML, URDF, and XML strictly for interfacing with hardware middleware or simulating physical properties.

### 6. Linting, Typing & Style
*   **Ruff as Standard:** Use Ruff for all linting and formatting via the `pyproject.toml`. Use explicit commands (e.g., `ruff check .`).
*   **Explicit Suppressions:** If a linting rule must be broken, include an inline suppression comment explicitly justifying the bypass.
*   **Strong Typing:** Apply strict type hints to all function signatures.

### 7. Testing & Execution
*   **The Beyoncé Rule:** If you write or modify a function, write or update its test.
*   **Hermetic Testing:** Tests must be completely self-contained using localized dummy data.
*   **Golden Tests for Math:** Implement fixed-seed forward pass tests for trajectory optimization or ML algorithms to assert exact tensor shapes and sums.
*   **Hardware Agnosticism:** Ensure dynamic device placement whenever possible. Code must ideally run locally on macOS and scale seamlessly to Ubuntu-based Slurm clusters. **Exemption:** Modules with hard CUDA dependencies (e.g., specific PyTorch3D extensions or custom CUDA kernels) are exempt if cross-platform compatibility is not technically feasible.

### 8. Agent Specific Instructions
*   **Context Length:** To avoid letting the context getting too big and wasting tokens, compact when the context size reaches 33% of the max.