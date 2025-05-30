# Template Python Operator

Cell-wise mean calculation implemented in Python.

## Python operator - Development workflow

* Set up [the Tercen Studio development environment](https://github.com/tercen/tercen_studio)
* Create a new git repository based on the [template Python operator](https://github.com/tercen/template-python-operator)
* Open VS Code Server by going to: http://127.0.0.1:8443
* Clone this repository into VS Code (using the 'Clone from GitHub' command from the Command Palette)
* Load the environment and install core requirements by running the following commands in the terminal:

  ```bash
  source ./config/.pyenv/versions/3.9.0/bin/activate
  pip install -r requirements.txt
  ```

* Develop your operator. Note that you can interact with an existing data step by specifying arguments to the `TercenContext` function:

  ```python
  tercenCtx = ctx.TercenContext(
      workflowId="YOUR_WORKFLOW_ID",
      stepId="YOUR_STEP_ID",
      username="admin",  # if using the local Tercen instance
      password="admin",  # if using the local Tercen instance
      serviceUri = "http://tercen:5400/"  # if using the local Tercen instance 
  )
  ```

* Generate requirements

  ```bash
  python3 -m tercen.util.requirements . > requirements.txt
  ```

* Push your changes to GitHub: triggers CI GH workflow
* Tag the repository: triggers Release GH workflow
* Go to tercen and install your operator

### Helpful Commands

#### Install Tercen Python Client

```bash
python3 -m pip install --force git+https://github.com/tercen/tercen_python_client@0.7.1
```

#### Wheel

Though not strictly mandatory, many packages require it.

```bash
python3 -m pip install wheel
```

---

## operator.json

The `operator.json` file defines metadata and configuration for this operator. Key sections:

- **name**: Identifier of the operator (e.g., `my_compressor`).
- **description**: A short summary of the operator's purpose.
- **tags**: Keywords to categorize the operator in Tercen.
- **authors**: List of authors or organization responsible.
- **urls**: Related URLs, such as repository or documentation.
- **container**: Docker registry path and tag for the operator image.
- **properties**: Configuration parameters for the operator:
  - `ColorNumber` (integer, default 3): Number of color segments to compress to.
  - `MaxIteration` (integer, default 15): Maximum iterations in compression.
  - `precision` (double, default 0.1): Convergence threshold for algorithm.
- **operatorSpec**: Specification for integration in Tercen:
  - `ontologyUri` and `ontologyVersion` (not shown here).
  - **inputSpecs**: Defines inputs such as a `CrosstabSpec`:
    - **metaFactors**: Expected data factors mapped to roles (e.g., Y coordinate, X coordinate, colors).
    - **axis**: One or more axes of the crosstab, each with its own metaFactors.
  - **outputSpecs**: (if present) defines outputs generated by the operator.


Refer to the `operator.json` file in the repository for the full details and examples of each field.