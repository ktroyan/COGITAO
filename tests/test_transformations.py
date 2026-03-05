"""Tests for individual shape transformations."""

import numpy as np
import pytest

from arcworld.shapes.base import Shape
from arcworld.transformations.shape_transformations import (
    change_shape_color,
    crop_bottom_side,
    crop_contours,
    crop_left_side,
    crop_right_side,
    crop_top_side,
    double_down,
    double_left,
    double_right,
    double_up,
    empty_inside_pixels,
    extend_contours_different_color,
    extend_contours_same_color,
    fill_holes_different_color,
    fill_holes_same_color,
    mirror_horizontal,
    mirror_vertical,
    pad_bottom,
    pad_left,
    pad_right,
    pad_shape,
    pad_top,
    quadruple_shape,
    rot90,
    translate_down,
    translate_left,
    translate_right,
    translate_up,
)


@pytest.fixture
def simple_square():
    """Create a simple 3x3 blue square."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # Blue square
    return Shape(grid)


@pytest.fixture
def hollow_square():
    """Create a hollow square."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[1:6, 1:6] = 1  # Outer square
    grid[2:5, 2:5] = 0  # Hollow center
    return Shape(grid)


@pytest.fixture
def rectangle_shape():
    """Create a 2x4 rectangle."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[3:5, 2:6] = 2  # Red rectangle
    return Shape(grid)


@pytest.fixture
def multicolor_shape():
    """Create a shape with multiple colors."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # Blue
    grid[3, 3] = 2  # Red center
    return Shape(grid)


class TestTranslations:
    """Test translation transformations."""

    def test_translate_up(self, simple_square):
        original_pos = simple_square.current_position
        transformed = translate_up(simple_square)
        new_pos = transformed.current_position
        assert new_pos[0] == original_pos[0] - 1
        assert new_pos[1] == original_pos[1]
        assert transformed.num_points == simple_square.num_points

    def test_translate_down(self, simple_square):
        original_pos = simple_square.current_position
        transformed = translate_down(simple_square)
        new_pos = transformed.current_position
        assert new_pos[0] == original_pos[0] + 1
        assert new_pos[1] == original_pos[1]
        assert transformed.num_points == simple_square.num_points

    def test_translate_right(self, simple_square):
        original_pos = simple_square.current_position
        transformed = translate_right(simple_square)
        new_pos = transformed.current_position
        assert new_pos[0] == original_pos[0]
        assert new_pos[1] == original_pos[1] + 1
        assert transformed.num_points == simple_square.num_points

    def test_translate_left(self, simple_square):
        original_pos = simple_square.current_position
        transformed = translate_left(simple_square)
        new_pos = transformed.current_position
        assert new_pos[0] == original_pos[0]
        assert new_pos[1] == original_pos[1] - 1
        assert transformed.num_points == simple_square.num_points

    def test_multiple_translations(self, simple_square):
        """Test chaining translations."""
        shape = simple_square
        shape = translate_up(shape)
        shape = translate_right(shape)
        shape = translate_down(shape)
        shape = translate_left(shape)
        # Should return to original position
        assert shape.current_position == simple_square.current_position


class TestRotations:
    """Test rotation transformations."""

    def test_rot90_square(self, simple_square):
        """Square should look the same after rotation."""
        transformed = rot90(simple_square)
        assert transformed.num_points == simple_square.num_points
        # For a square, the shape should remain the same size
        assert transformed.n_rows == simple_square.n_rows
        assert transformed.n_cols == simple_square.n_cols

    def test_rot90_rectangle(self, rectangle_shape):
        """Rectangle should swap dimensions."""
        original_rows = rectangle_shape.n_rows
        original_cols = rectangle_shape.n_cols
        transformed = rot90(rectangle_shape)
        # After 90 degree rotation, rows and cols should swap
        assert transformed.n_rows == original_cols
        assert transformed.n_cols == original_rows
        assert transformed.num_points == rectangle_shape.num_points

    def test_rot90_four_times(self, rectangle_shape):
        """Four 90-degree rotations should return to original."""
        shape = rectangle_shape
        for _ in range(4):
            shape = rot90(shape)
        assert shape.n_rows == rectangle_shape.n_rows
        assert shape.n_cols == rectangle_shape.n_cols
        assert shape.num_points == rectangle_shape.num_points


class TestFilling:
    """Test filling transformations."""

    def test_fill_holes_same_color(self, hollow_square):
        original_points = hollow_square.num_points
        filled = fill_holes_same_color(hollow_square)
        # Should have more points after filling
        assert filled.num_points > original_points
        # Should still be the same color
        assert len(filled.existing_colors) == len(hollow_square.existing_colors)

    def test_fill_holes_different_color(self, hollow_square):
        original_points = hollow_square.num_points
        original_colors = len(hollow_square.existing_colors)
        filled = fill_holes_different_color(hollow_square)
        # Should have more points after filling
        assert filled.num_points > original_points
        # Should have one additional color
        assert len(filled.existing_colors) == original_colors + 1

    def test_fill_no_holes(self, simple_square):
        """Filling a solid shape should not change it."""
        filled = fill_holes_same_color(simple_square)
        assert filled.num_points == simple_square.num_points


class TestEmptying:
    """Test emptying transformations."""

    def test_empty_inside_pixels(self, simple_square):
        emptied = empty_inside_pixels(simple_square)
        # Should have fewer points (only outer pixels remain)
        assert emptied.num_points < simple_square.num_points
        assert emptied.num_points > 0  # Should not be completely empty

    def test_empty_already_hollow(self, hollow_square):
        """Emptying a hollow shape should reduce it further."""
        emptied = empty_inside_pixels(hollow_square)
        assert emptied.num_points <= hollow_square.num_points


class TestMirroring:
    """Test mirroring transformations."""

    def test_mirror_horizontal(self, rectangle_shape):
        mirrored = mirror_horizontal(rectangle_shape)
        assert mirrored.num_points == rectangle_shape.num_points
        assert mirrored.n_rows == rectangle_shape.n_rows
        assert mirrored.n_cols == rectangle_shape.n_cols

    def test_mirror_vertical(self, rectangle_shape):
        mirrored = mirror_vertical(rectangle_shape)
        assert mirrored.num_points == rectangle_shape.num_points
        assert mirrored.n_rows == rectangle_shape.n_rows
        assert mirrored.n_cols == rectangle_shape.n_cols

    def test_mirror_twice_returns_original(self, rectangle_shape):
        """Mirroring twice should return to original shape."""
        twice_h = mirror_horizontal(mirror_horizontal(rectangle_shape))
        assert twice_h.indexes == rectangle_shape.indexes

        twice_v = mirror_vertical(mirror_vertical(rectangle_shape))
        assert twice_v.indexes == rectangle_shape.indexes


class TestCropping:
    """Test cropping transformations."""

    def test_crop_left_side(self, rectangle_shape):
        cropped = crop_left_side(rectangle_shape)
        assert cropped.num_points < rectangle_shape.num_points
        assert cropped.n_cols < rectangle_shape.n_cols

    def test_crop_right_side(self, rectangle_shape):
        cropped = crop_right_side(rectangle_shape)
        assert cropped.num_points < rectangle_shape.num_points
        assert cropped.n_cols < rectangle_shape.n_cols

    def test_crop_top_side(self, rectangle_shape):
        cropped = crop_top_side(rectangle_shape)
        assert cropped.num_points < rectangle_shape.num_points
        assert cropped.n_rows < rectangle_shape.n_rows

    def test_crop_bottom_side(self, rectangle_shape):
        cropped = crop_bottom_side(rectangle_shape)
        assert cropped.num_points < rectangle_shape.num_points
        assert cropped.n_rows < rectangle_shape.n_rows

    def test_crop_contours(self, simple_square):
        """Crop contours should hollow out the shape."""
        cropped = crop_contours(simple_square)
        assert cropped.num_points < simple_square.num_points


class TestExtending:
    """Test extending transformations."""

    def test_extend_contours_same_color(self, simple_square):
        extended = extend_contours_same_color(simple_square)
        # Should have more points after extending
        assert extended.num_points > simple_square.num_points
        # Should maintain same number of colors
        assert len(extended.existing_colors) == len(simple_square.existing_colors)

    def test_extend_contours_different_color(self, simple_square):
        original_colors = len(simple_square.existing_colors)
        extended = extend_contours_different_color(simple_square)
        # Should have more points
        assert extended.num_points > simple_square.num_points
        # Should have one additional color
        assert len(extended.existing_colors) == original_colors + 1


class TestRecoloring:
    """Test color change transformations."""

    def test_change_shape_color(self, simple_square):
        original_color = simple_square.most_frequent_color
        recolored = change_shape_color(simple_square)
        new_color = recolored.most_frequent_color
        # Color should be different
        assert new_color != original_color
        # Number of points should remain the same
        assert recolored.num_points == simple_square.num_points

    def test_change_color_preserves_pattern(self, multicolor_shape):
        """Color change should preserve relative color patterns."""
        original_colors = len(multicolor_shape.existing_colors)
        recolored = change_shape_color(multicolor_shape)
        # Should still have the same number of distinct colors
        assert len(recolored.existing_colors) == original_colors


class TestPadding:
    """Test padding transformations."""

    def test_pad_top(self, simple_square):
        padded = pad_top(simple_square)
        assert padded.n_rows > simple_square.n_rows

    def test_pad_bottom(self, simple_square):
        padded = pad_bottom(simple_square)
        assert padded.n_rows > simple_square.n_rows

    def test_pad_left(self, simple_square):
        padded = pad_left(simple_square)
        assert padded.n_cols > simple_square.n_cols

    def test_pad_right(self, simple_square):
        padded = pad_right(simple_square)
        assert padded.n_cols > simple_square.n_cols

    def test_pad_shape(self, simple_square):
        """Pad shape should increase both dimensions."""
        padded = pad_shape(simple_square)
        assert padded.n_rows > simple_square.n_rows
        assert padded.n_cols > simple_square.n_cols


class TestDoubling:
    """Test doubling transformations."""

    def test_double_right(self, simple_square):
        doubled = double_right(simple_square)
        # Should roughly double in width
        assert doubled.n_cols >= simple_square.n_cols * 2 - 1
        # Points should roughly double
        assert doubled.num_points >= simple_square.num_points * 1.5

    def test_double_left(self, simple_square):
        doubled = double_left(simple_square)
        assert doubled.n_cols >= simple_square.n_cols * 2 - 1
        assert doubled.num_points >= simple_square.num_points * 1.5

    def test_double_up(self, simple_square):
        doubled = double_up(simple_square)
        assert doubled.n_rows >= simple_square.n_rows * 2 - 1
        assert doubled.num_points >= simple_square.num_points * 1.5

    def test_double_down(self, simple_square):
        doubled = double_down(simple_square)
        assert doubled.n_rows >= simple_square.n_rows * 2 - 1
        assert doubled.num_points >= simple_square.num_points * 1.5

    def test_quadruple_shape(self, simple_square):
        """Quadruple should create 4 copies."""
        quadrupled = quadruple_shape(simple_square)
        # Should have significantly more points (close to 4x)
        assert quadrupled.num_points >= simple_square.num_points * 3


class TestTransformationInvariants:
    """Test properties that should be preserved across transformations."""

    def test_color_preservation(self, simple_square):
        """Most transformations preserve the color set."""
        # Transformations that should preserve colors
        transforms = [
            translate_up,
            translate_down,
            translate_left,
            translate_right,
            rot90,
            mirror_horizontal,
            mirror_vertical,
        ]

        original_colors = set(simple_square.existing_colors)
        for transform in transforms:
            transformed = transform(simple_square)
            # Should have same colors (though may be in different positions)
            assert set(transformed.existing_colors) == original_colors

    def test_point_count_preservation(self, simple_square):
        """Some transformations should preserve point count."""
        transforms = [
            translate_up,
            translate_down,
            translate_left,
            translate_right,
            rot90,
            mirror_horizontal,
            mirror_vertical,
        ]

        original_points = simple_square.num_points
        for transform in transforms:
            transformed = transform(simple_square)
            assert transformed.num_points == original_points
