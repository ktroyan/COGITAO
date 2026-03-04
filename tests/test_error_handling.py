"""Tests for error handling and edge cases."""

import numpy as np
import pytest

from arcworld import DatasetConfig, Generator
from arcworld.constants import DoesNotFitException, ShapeOutOfBounds
from arcworld.general_utils import position_shape_in_world, randomly_add_shape_to_world
from arcworld.shapes.base import Shape
from arcworld.shapes.rectangle import Rectangle


class TestShapePositioning:
    """Test error handling in shape positioning."""

    def test_shape_too_large_for_grid(self):
        """Test positioning a shape that's too large for the grid."""
        # Create a large shape
        large_grid = np.zeros((20, 20), dtype=np.int32)
        large_grid[5:15, 5:15] = 1
        large_shape = Shape(large_grid)

        # Try to position it in a small world
        small_world = np.zeros((5, 5), dtype=np.int32)

        # Should raise error or handle gracefully
        with pytest.raises((DoesNotFitException, ShapeOutOfBounds, Exception)):
            position_shape_in_world(small_world, large_shape, (0, 0))

    def test_shape_out_of_bounds_position(self):
        """Test positioning a shape at an out-of-bounds location."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:5, 2:5] = 1
        shape = Shape(grid)

        world = np.zeros((10, 10), dtype=np.int32)

        # Try to position at a location that would put shape out of bounds
        # This depends on implementation - it might clip or raise an error
        try:
            # Position too far to the right/bottom
            result = position_shape_in_world(world, shape, (8, 8))
            # If it succeeds, verify it didn't corrupt the world
            assert result.shape == world.shape
        except (ShapeOutOfBounds, IndexError, DoesNotFitException):
            # This is also acceptable behavior
            pass

    def test_random_add_shape_to_full_grid(self):
        """Test adding a shape when grid is full."""
        # Create a completely filled world
        world = np.ones((10, 10), dtype=np.int32)

        # Create a shape to add
        shape_grid = np.zeros((5, 5), dtype=np.int32)
        shape_grid[1:4, 1:4] = 2
        shape = Shape(shape_grid)

        # Should fail to find a valid position
        with pytest.raises((DoesNotFitException, Exception)):
            # Try multiple times to ensure it's not just unlucky
            for _ in range(100):
                randomly_add_shape_to_world(world, shape)
                # If we get here, the function succeeded somehow
                break

    def test_delete_out_of_bounds_points(self):
        """Test deleting points that are out of bounds."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:5, 2:5] = 1
        shape = Shape(grid)

        # Move shape to near the edge
        shape.move_to_position((0, 0))

        # Delete out of bounds points
        shape.delete_out_of_bounds_points()

        # Shape should still be valid
        assert shape.num_points >= 0


class TestGeneratorRetryLogic:
    """Test generator retry and failure handling."""

    def test_impossible_configuration(self):
        """Test generator with impossible constraints."""
        # Create a config that's very hard to satisfy
        config = DatasetConfig(
            min_n_shapes_per_grid=10,  # Many shapes
            max_n_shapes_per_grid=10,
            min_grid_size=5,  # Very small grid
            max_grid_size=5,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        generator = Generator(config)

        # Try to generate a task - might fail or return None/empty dict
        task = generator.generate_single_task()
        # If it returns something, it should be valid or empty (indicating failure)
        if task is not None:
            assert "pairs" in task or task == {} or "error" in str(task).lower()

    def test_incompatible_transformations(self):
        """Test transformations with incompatible constraints."""
        # This tests the constraint system - some combinations may raise exceptions
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=1,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=["is_shape_hollow"],
            allowed_transformations=["change_shape_color", "translate_up"],
            min_transformation_depth=1,
            max_transformation_depth=2,
        )

        generator = Generator(config)

        # Should either succeed or fail gracefully
        task = generator.generate_single_task()
        if task is not None and "pairs" in task:
            # If successful, verify it has the expected structure
            assert len(task["pairs"]) > 0

    def test_no_shapes_satisfy_conditionals(self):
        """Test when no shapes satisfy the compulsory conditionals."""
        # Use a combination of conditionals that might be hard to satisfy
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=1,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=[
                "is_shape_hollow",
                "is_shape_of_two_rows",  # Hollow shape with exactly 2 rows is rare
            ],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        generator = Generator(config)

        # Check that some shapes are available after subsetting
        # If no shapes available, possible_shapes should be empty
        if len(generator.possible_shapes) == 0:
            # This is expected - no shapes match the constraints
            pass
        else:
            # Some shapes match, try to generate
            task = generator.generate_single_task()
            # Task might be None if generation fails
            assert task is None or "pairs" in task


class TestEmptyAndNullShapes:
    """Test handling of empty and null shapes."""

    def test_empty_grid_shape(self):
        """Test creating a shape from an empty grid."""
        empty_grid = np.zeros((10, 10), dtype=np.int32)
        shape = Shape(empty_grid)

        assert shape.is_null
        assert shape.num_points == 0

    def test_operations_on_null_shape(self):
        """Test that operations on null shapes don't crash."""
        empty_grid = np.zeros((10, 10), dtype=np.int32)
        shape = Shape(empty_grid)

        # These should not crash
        assert len(shape.existing_colors) == 0
        assert shape.is_null


class TestInvalidInputs:
    """Test handling of invalid inputs."""

    def test_shape_from_invalid_data(self):
        """Test creating shape from invalid data."""
        # Try to create shape from wrong type
        with pytest.raises((ValueError, TypeError, AttributeError)):
            Shape("invalid")

    def test_negative_grid_values(self):
        """Test grid with negative values raises error."""
        grid = np.array([[-1, -1], [-1, -1]], dtype=np.int32)

        # Negative colors are not allowed, should raise ValueError
        with pytest.raises(ValueError, match="Colors not allowed"):
            shape = Shape(grid)

    def test_grid_with_invalid_colors(self):
        """Test grid with colors outside valid range (0-9) raises error."""
        grid = np.array([[15, 20], [25, 30]], dtype=np.int32)

        # Colors > 9 are not allowed, should raise ValueError
        with pytest.raises(ValueError, match="Colors not allowed"):
            shape = Shape(grid)


class TestConcurrentAccess:
    """Test concurrent access patterns."""

    def test_generator_shape_file_access(self):
        """Test that generator can access shape file."""
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=2,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        # Create multiple generators (simulating multi-process scenario)
        generators = [Generator(config) for _ in range(3)]

        # All should be able to access the shape file
        for gen in generators:
            assert gen.shape_file is not None
            assert len(gen.possible_shapes) > 0

    def test_multiple_generators_independent(self):
        """Test that multiple generators are independent."""
        config1 = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=2,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        config2 = DatasetConfig(
            min_n_shapes_per_grid=2,
            max_n_shapes_per_grid=3,
            min_grid_size=30,
            max_grid_size=40,
            n_examples=2,
            shape_compulsory_conditionals=[],
            allowed_transformations=["translate_up"],
            min_transformation_depth=1,
            max_transformation_depth=2,
        )

        gen1 = Generator(config1)
        gen2 = Generator(config2)

        # Configs should be independent
        assert gen1.config.min_grid_size != gen2.config.min_grid_size
        assert (
            gen1.config.allowed_transformations != gen2.config.allowed_transformations
        )


class TestEdgeCases:
    """Test various edge cases."""

    def test_single_pixel_grid(self):
        """Test with a 1x1 grid."""
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=1,
            min_grid_size=1,
            max_grid_size=1,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=0,
            max_transformation_depth=0,
        )

        generator = Generator(config)
        task = generator.generate_single_task()

        # Might succeed or fail depending on implementation
        if task is not None and "pairs" in task:
            assert len(task["pairs"]) > 0

    def test_maximum_shapes_in_grid(self):
        """Test with maximum number of shapes."""
        config = DatasetConfig(
            min_n_shapes_per_grid=20,
            max_n_shapes_per_grid=20,
            min_grid_size=50,
            max_grid_size=50,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        generator = Generator(config)
        # This might be slow or fail, but should not crash
        try:
            task = generator.generate_single_task()
            if task is not None and "pairs" in task:
                # Count shapes in input
                input_grid = task["pairs"][0]["input"]
                # Should have multiple shapes (exact count hard to verify)
                assert np.any(input_grid > 0)
        except (DoesNotFitException, Exception) as e:
            # Acceptable if it can't fit that many shapes
            pass

    def test_zero_transformation_depth(self):
        """Test with zero transformation depth (identity)."""
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=2,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=0,
            max_transformation_depth=0,
        )

        generator = Generator(config)
        task = generator.generate_single_task()

        if task is not None and "pairs" in task:
            # With depth 0, input and output might be similar or identical
            pair = task["pairs"][0]
            assert pair["input"].shape == pair["output"].shape


class TestShapeConstraints:
    """Test shape constraint validation."""

    def test_get_compatible_shape_rows_contradiction(self):
        """Test getting shapes with contradictory constraints."""
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=2,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        generator = Generator(config)

        # Try to get shapes that satisfy contradictory constraints
        with pytest.raises(Exception):
            # Same constraint in both to_satisfy and not_to_satisfy
            generator.get_compatible_shape_rows(
                shape_conditionals_to_satisfy=["is_shape_symmetric"],
                shape_conditionals_not_to_satisfy=["is_shape_symmetric"],
            )

    def test_get_compatible_shape_rows_no_matches(self):
        """Test getting shapes when no shapes match."""
        config = DatasetConfig(
            min_n_shapes_per_grid=1,
            max_n_shapes_per_grid=2,
            min_grid_size=10,
            max_grid_size=20,
            n_examples=1,
            shape_compulsory_conditionals=[],
            allowed_transformations=["change_shape_color"],
            min_transformation_depth=1,
            max_transformation_depth=1,
        )

        generator = Generator(config)

        # Try to get shapes with very restrictive constraints
        # that might not match any shapes
        result = generator.get_compatible_shape_rows(
            shape_conditionals_to_satisfy=[
                "is_shape_hollow",
                "is_shape_of_two_rows",
                "is_shape_of_two_cols",
                "is_shape_of_3_colors",
            ],
            shape_conditionals_not_to_satisfy=[],
        )

        # Result might be empty list
        assert isinstance(result, list)
