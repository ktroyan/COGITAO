from typing import List, Optional
from pydantic import BaseModel, Field, model_validator, field_validator


class ConfigValidator(BaseModel):
    min_n_shapes_per_grid: int = Field(..., ge=1)
    max_n_shapes_per_grid: int = Field(..., ge=1)

    min_grid_size: int = Field(..., ge=1)
    max_grid_size: int = Field(..., ge=1)

    n_examples: int = Field(..., ge=1)

    allowed_transformations: Optional[List[str]] = None
    allowed_combinations: Optional[List[List[str]]] = None

    min_transformation_depth: Optional[int] = Field(default=None, ge=0)
    max_transformation_depth: Optional[int] = Field(default=None, ge=0)

    shape_compulsory_conditionals: List[str]

    @field_validator("allowed_transformations")
    @classmethod
    def validate_allowed_transformations(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

    @field_validator("allowed_combinations")
    @classmethod
    def validate_allowed_combinations(cls, v):
        if v is not None and len(v) == 0:
            return None
        return v

    @model_validator(mode="after")
    def check_constraints(self):
        min_shapes = self.min_n_shapes_per_grid
        max_shapes = self.max_n_shapes_per_grid
        if max_shapes < min_shapes:
            raise ValueError("max_n_shapes_per_grid must be >= min_n_shapes_per_grid")

        min_grid = self.min_grid_size
        max_grid = self.max_grid_size
        if max_grid < min_grid:
            raise ValueError("max_grid_size must be >= min_grid_size")

        allowed_trans = self.allowed_transformations
        allowed_combs = self.allowed_combinations

        # Check that exactly one of allowed_transformations or allowed_combinations is provided
        if (allowed_trans is None and allowed_combs is None) or (
            allowed_trans is not None and allowed_combs is not None
        ):
            raise ValueError(
                "Exactly one of allowed_combinations or allowed_transformations must be provided (not both or neither)."
            )

        min_depth = self.min_transformation_depth
        max_depth = self.max_transformation_depth

        if allowed_trans is not None:
            if min_depth is None or max_depth is None:
                raise ValueError(
                    "When allowed_transformations is provided, min_transformation_depth and max_transformation_depth must also be set"
                )
            if max_depth < min_depth:
                raise ValueError(
                    "max_transformation_depth must be >= min_transformation_depth"
                )
        else:
            if min_depth is not None or max_depth is not None:
                raise ValueError(
                    "min_transformation_depth and max_transformation_depth must be set to None when allowed_transformations is None"
                )

        return self
