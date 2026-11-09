# Blocks

::: gpex.blocks
    options:
      members:
        - Locations
        - FunctionalBlock
        - FunctionalQuery
        - ObservationDesignBlock
        - ObservationDesign
        - ObservationBlock
        - FunctionalDataset
        - StructuredValues
        - IndexMap

`ObservationDesignBlock.variance` stores the scalar measurement-error variance.
Use a fixed scalar or a scalar `PositiveParameter` to fit it. The same variance
applies to every location and output component in that block.

The block's `variances` property evaluates that parameter and repeats it over
the functional output shape. Collections expose these measurement-error
variances in canonical scalar-output order through `variance_flat`.

Observation designs reserve `max_size` locations and keep a dynamic active
`count`. See [JAX and observation capacity](../concepts/jax-and-capacity.md)
for capacity, padding and compilation behavior. Functional queries are unreserved.
