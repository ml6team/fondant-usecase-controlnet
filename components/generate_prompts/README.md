# Generate prompts

### Description
Component that generates a set of seed prompts

### Inputs / outputs

**This component consumes no data.**

**This component produces:**

- prompts
    - text: string

### Arguments

The component takes the following arguments to alter its behavior:

| argument | type | description | default |
| -------- | ---- | ----------- | ------- |
| n_rows_to_load | int | Optional argument that defines the number of rows to load. Useful for testing pipeline runs on a small scale | / |

### Usage

You can add this component to your pipeline using the following code:

```python
from fondant.pipeline import ComponentOp


generate_prompts_op = ComponentOp.from_registry(
    name="generate_prompts",
    arguments={
        # Add arguments
        # "n_rows_to_load": 0,
    }
)
pipeline.add_op(generate_prompts_op, dependencies=[...])  #Add previous component as dependency
```

