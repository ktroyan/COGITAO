import numpy as np
import random
import h5py
import pandas as pd

from . import hdf5_utils
from .shapes.base import Shape
from .general_utils import (
    position_shape_in_world,
    randomly_add_shape_to_world,
)
from .conditionals.single_shape_conditionals import (
    conditionals_dict as single_shape_conditionals_dict,
)
from .transformations.shape_transformations import (
    transformations_dict,
    transformations_constraints,
)
from .utils.config_validation import ConfigValidator


class Generator:
    def __init__(self, config: dict | ConfigValidator, debug_mode=False):
        if isinstance(config, ConfigValidator):
            self.config = config
        else:
            self.config = ConfigValidator(**config)
        self.transformations_dict = transformations_dict  ## Loads transformations_dict into the class (from the ...transformation.py files)
        self.transformations_constraints = transformations_constraints

        self.conditionals_dict = single_shape_conditionals_dict  ## Loads conditionals_dict into the class (from the ...conditionals.py files)

        self.shape_conditionals_table, self.conditionals_names = (
            hdf5_utils.load_conditions()
        )
        self.shape_conditionals_table_df = pd.DataFrame(self.shape_conditionals_table)
        self.conditionals_names = {k: v for v, k in enumerate(self.conditionals_names)}
        self.shape_file = h5py.File(hdf5_utils.SHAPE_DATASET_PATH)
        self.subset_shapes()
        self.max_trials_for_function_combination = (
            15  # Number of trials to generate a specific function combination
        )
        self.max_trials_for_configuration = 30  # Number of trials to sample from a specific configuration before quitting
        # and advising the user to maybe update the config
        self.debug_mode = debug_mode

    def subset_shapes(self):
        """Subset shapes based on the compulsory conditions specified in the config in order to reduce search
        space during generation"""
        conditional_indices_of_interest = []
        for c in self.config.shape_compulsory_conditionals:
            conditional_indices_of_interest.append(self.conditionals_names[c])
        condition_satisfaction_mask = (
            self.shape_conditionals_table[:, conditional_indices_of_interest] == 1
        )
        condition_satisfaction_mask_sum = np.sum(condition_satisfaction_mask, axis=-1)
        satisfying_rows = np.where(
            condition_satisfaction_mask_sum == len(conditional_indices_of_interest)
        )[0].tolist()
        self.possible_shapes = np.arange(self.shape_conditionals_table.shape[0])[
            satisfying_rows
        ]
        self.shape_conditionals_table = self.shape_conditionals_table[satisfying_rows]

    def get_compatible_shape_rows(
        self,
        shape_conditionals_to_satisfy: list,
        shape_conditionals_not_to_satisfy: list,
    ) -> list:
        """
        Sample shapes rows from the library of available shapes given compulsory conditionals and constraints from
        sequence of transformations

        This function presents some redundancy with the function "subset_shapes" above. However, it is kept
        separately in order to allow for the possibility of having different sets of conditionals for different
        tasks. Most importantly, it is kept for the sake of selecting shapes *not to satisfy*. subset_shapes
        fullfills the purpose of, on the get go of the class instance (based on the config), subsetting to only
        the shapes that are compatible with the compulsory conditionals.

        Parameters:
        conditionals_to_satisfy (list): shape conditionals that sample shapes must satisfy
        conditionals_not_to_satisfy (list): shape conditionals that sample shapes must NOT satisfy

        Returns:
        compatible_shape_rows (list): shape rows that are compatible to position
        """

        ## Verify there is no overlap between conditions to satisfy and not to satisfy
        if (
            len(
                set(shape_conditionals_to_satisfy)
                & set(shape_conditionals_not_to_satisfy)
            )
            != 0
        ):
            raise Exception(
                "There is overlap between conditions to satisfy and conditions not \
                            to satisfy. Check your config file!"
            )

        ## "Conditional indices" serve the purpose of searching through the shape table and corresponding conditionals.
        conditionals_to_satisfy_indices = []
        for c in shape_conditionals_to_satisfy:
            conditionals_to_satisfy_indices.append(self.conditionals_names[c])

        conditionals_not_to_satisfy_indices = []
        for c in shape_conditionals_not_to_satisfy:
            conditionals_not_to_satisfy_indices.append(self.conditionals_names[c])

        conditions_ones = [
            self.shape_conditionals_table_df[col] == 1
            for col in conditionals_to_satisfy_indices
        ]
        conditions_zeros = [
            self.shape_conditionals_table_df[col] == 0
            for col in conditionals_not_to_satisfy_indices
        ]

        if conditions_ones:
            combined_conditions = conditions_ones[0]
            for condition in conditions_ones[1:]:
                combined_conditions &= condition

        # For the zeros, start with the first condition and chain with &
        if conditions_zeros:
            for condition in conditions_zeros:
                combined_conditions &= condition

        # Filter the DataFrame based on the combined conditions
        return self.shape_conditionals_table_df[combined_conditions].index.tolist()

    def randomly_sample_shapes(
        self, compatible_shape_rows: list, n_shapes_wanted: int
    ) -> list[Shape]:
        """
        Given the compatible shape rows, randomly sample shapes to position in a grid

        Parameters:
        compatible_shape_rows (list):  output from self.get_compatible_shape_rows()
        n_shapes_wanted (list):
        """
        shapes_to_position = []
        for i in range(n_shapes_wanted):
            random_row = int(random.choice(compatible_shape_rows))
            shape_grid = hdf5_utils.load_shape(random_row, self.shape_file)
            shapes_to_position.append(Shape(shape_grid))
        return shapes_to_position

    def sample_transform_suite(self):
        """
        Sample a transformation suite for the task. The transformation suite is a list of transformations
        that will be applied to the shapes in the grid. The transformations are sampled from the
        transformations_dict and the transformations_constraints dictionaries. The transformations are
        sampled in a way that respects the constraints specified in the transformations_constraints dictionary."""

        if (
            self.config.allowed_combinations is not None
        ):  ## If allowed combinations are specified, sample from them
            return random.choice(self.config.allowed_combinations)

        elif (
            self.config.allowed_transformations is not None
        ):  ## If allowed transformations are specified, sample from them
            compatible_transforms = list(self.transformations_dict.keys())
            compatible_transforms = [
                t
                for t in compatible_transforms
                if t in self.config.allowed_transformations
            ]

            depth = random.randint(
                self.config.min_transformation_depth,
                self.config.max_transformation_depth,
            )
            transform_suite = []
            for k in range(depth):
                transform_suite.append(random.sample(compatible_transforms, 1)[0])

                ## Remove incompatible transforms from the pool of compatible transforms once transform selected.
                ## This will basically remove "translate_down" from pool of possible transforms if "translate_up" is selected, for example.

                transforms_to_remove = self.transformations_constraints[
                    transform_suite[-1]
                ]["incompatible_transforms"]
                for t in transforms_to_remove:
                    if t in compatible_transforms:
                        compatible_transforms.remove(t)

                ## If transform cannot be applied twice, remove it from compatible transforms
                if (
                    self.transformations_constraints[transform_suite[-1]][
                        "can_be_applied_multiple_times"
                    ]
                    == False
                ):
                    compatible_transforms.remove(transform_suite[-1])

            return transform_suite
        else:
            raise Exception(
                "Please specify allowed transformations or allowed combinations in the config"
            )

    def get_shape_constraints_from_rule_sampled(self, transform_suite: list) -> list:
        """
        get all shape conditionals which cannot be associated with the chosen task transforms

        Parameters:
        transform_suite (list): the transformation sequence for the task. e.g. ["translate_up", "fill_holes"]

        Returns:
        shape_constraints (list): list of shape conditionals NOT not to sample for the given rule
        """
        shape_constraints = []
        for t in transform_suite:
            shape_constraints += transformations_constraints[t]["incompatible_shapes"]
        return list(dict.fromkeys(shape_constraints))

    def set_up_initial_grid(
        self, compatible_shape_rows: list
    ) -> tuple[np.ndarray, list[Shape]]:
        """
        Set up initial grid

        Parameters:
        compatible_shape_rows (list): as computed in self.get_compatible_shape_rows()

        Returns:
        main_grid (np.ndarray): grid with positionned shape
        positionned_shapes (list[Shape]): list of Shapes.
        """
        grid_size = [
            random.randint(self.config.min_grid_size, self.config.max_grid_size),
            random.randint(self.config.min_grid_size, self.config.max_grid_size),
        ]
        main_grid = np.zeros(grid_size)

        n_shapes_wanted = random.randint(
            self.config.min_n_shapes_per_grid, self.config.max_n_shapes_per_grid
        )

        shapes_to_position = self.randomly_sample_shapes(
            compatible_shape_rows=compatible_shape_rows, n_shapes_wanted=n_shapes_wanted
        )
        positionned_shapes = []
        for s in shapes_to_position:
            main_grid, positionned_s = randomly_add_shape_to_world(main_grid, s)
            positionned_shapes.append(positionned_s)
        return main_grid, positionned_shapes

    def apply_transform_suite_to_grid(
        self, transform_suite, input_grid, shapes_positionned
    ):
        """
        Apply the transformation suite to the grid and return the output grid
        Parameters:
        transform_suite (list): the transformation sequence for the task. e.g. ["translate_up", "fill_holes"]
        input_grid (np.ndarray): grid with positionned shape
        shapes_positionned (list[Shape]): list of Shapes.

        Returns:
        output_grid (np.ndarray): grid with positionned shape
        """
        output_grid = np.zeros_like(input_grid)
        full_grid_sequence = [
            input_grid.copy()
        ]  # To keep track of the full sequence of grids
        for original_shape in shapes_positionned:
            transformed_shape = original_shape
            for t in transform_suite:
                transformed_shape = self.transformations_dict[t](transformed_shape)

            if (
                transformed_shape.is_null == False
            ):  ### Invalidate the whole initial grid
                output_grid = position_shape_in_world(output_grid, transformed_shape)
            else:
                output_grid = None  # If the shape is null, we invalidate the whole grid
            full_grid_sequence.append(output_grid.copy())
        return output_grid, full_grid_sequence

    def apply_transform_suite_to_grid_2(
        self, transform_suite, input_grid, shapes_positionned
    ):
        """
        Apply the transformation suite to the grid and return the output grid.
        This version loops over transformations first, then shapes.

        Parameters:
        transform_suite (list): the transformation sequence for the task. e.g. ["translate_up", "fill_holes"]
        input_grid (np.ndarray): grid with positioned shapes
        shapes_positionned (list[Shape]): list of Shapes.

        Returns:
        output_grid (np.ndarray): grid with positioned shapes
        full_grid_sequence (list): sequence of grid states after each transformation
        """
        output_grid = np.zeros_like(input_grid)
        full_grid_sequence = [input_grid.copy()]

        # Start with the original shapes
        current_shapes = shapes_positionned

        for t in transform_suite:
            transformed_shapes = []
            for shape in current_shapes:
                transformed_shape = self.transformations_dict[t](shape)
                transformed_shapes.append(transformed_shape)
            current_shapes = transformed_shapes

            # Build grid for this transformation step
            temp_grid = np.zeros_like(input_grid)
            for shape in current_shapes:
                if not shape.is_null:
                    temp_grid = position_shape_in_world(temp_grid, shape)
                else:
                    temp_grid = None
                    break
            full_grid_sequence.append(
                temp_grid.copy() if temp_grid is not None else None
            )

        # Final output grid is the last valid temp_grid
        output_grid = (
            full_grid_sequence[-1] if full_grid_sequence[-1] is not None else None
        )
        return output_grid, full_grid_sequence

    def generate_single_task(self):
        task = {"pairs": [], "transform_suite": None}
        n_config_trials = 0
        while n_config_trials < self.max_trials_for_configuration:
            transform_suite = self.sample_transform_suite()

            # Probably some edits needed here --> not a practical way of operating
            task_shape_constraints = self.get_shape_constraints_from_rule_sampled(
                transform_suite
            )
            compatible_shape_rows = self.get_compatible_shape_rows(
                shape_conditionals_to_satisfy=self.config.shape_compulsory_conditionals,
                shape_conditionals_not_to_satisfy=task_shape_constraints,
            )

            failed_transform_trials = 0
            generated_pairs = []
            while (
                failed_transform_trials < self.max_trials_for_function_combination
                and len(generated_pairs) < self.config.n_examples
            ):
                try:
                    input_grid, shapes_positionned = self.set_up_initial_grid(
                        compatible_shape_rows=compatible_shape_rows
                    )
                    output_grid, full_grid_sequence = (
                        self.apply_transform_suite_to_grid_2(
                            transform_suite, input_grid, shapes_positionned
                        )
                    )
                    to_append = {
                        "input": input_grid,
                        "output": output_grid,
                        "n_shapes": len(shapes_positionned),
                        "grid_size": input_grid.shape,
                        "full_grid_sequence": full_grid_sequence,
                    }
                    generated_pairs.append(to_append)
                    failed_transform_trials = 0  # If we successfully generate a grid, we reset the failed trials
                except Exception as e:
                    if self.debug_mode:
                        print(e)
                    failed_transform_trials += 1

            if len(generated_pairs) == self.config.n_examples:
                n_config_trials = (
                    0  # If we successfully generate a grid, we reset the config trials
                )
                return {
                    "pairs": generated_pairs,
                    "full_grid_sequence": full_grid_sequence,
                    "transformation_suite": transform_suite,
                }
            elif failed_transform_trials >= self.max_trials_for_function_combination:
                n_config_trials += 1
            else:
                print("Something went wrong with the generation of the task.")

        print(
            "Failed to generate a task with the current configuration. Please consider updating the config file. \n \
              Config is the following:",
            self.config,
        )
        return {}
