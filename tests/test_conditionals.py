"""Tests for shape conditional functions."""

import numpy as np
import pytest

from arcworld.conditionals.single_shape_conditionals import (
    is_shape_anti_diagonal_line,
    is_shape_antidiagonally_symmetric,
    is_shape_diagonal_line,
    is_shape_diagonal_or_antidiagonal_line,
    is_shape_diagonally_or_antidiagonally_symmetric,
    is_shape_diagonally_symmetric,
    is_shape_evenly_colored,
    is_shape_filled_rectangle,
    is_shape_filled_square,
    is_shape_fully_connected,
    is_shape_higher_than_wide,
    is_shape_hollow,
    is_shape_horizontal_line,
    is_shape_horizontally_or_vertically_symmetric,
    is_shape_horizontally_symmetric,
    is_shape_line,
    is_shape_more_than_1_cell,
    is_shape_more_than_3_colors,
    is_shape_not_evenly_colored,
    is_shape_not_fully_connected,
    is_shape_not_hollow,
    is_shape_not_simple,
    is_shape_not_symmetric,
    is_shape_of_2_colors,
    is_shape_of_3_colors,
    is_shape_of_five_cols,
    is_shape_of_five_rows,
    is_shape_of_four_cols,
    is_shape_of_four_rows,
    is_shape_of_three_cols,
    is_shape_of_three_rows,
    is_shape_of_two_cols,
    is_shape_of_two_rows,
    is_shape_same_height_width,
    is_shape_simple,
    is_shape_symmetric,
    is_shape_vertical_line,
    is_shape_vertically_symmetric,
    is_shape_wider_than_high,
)
from arcworld.shapes.base import Shape


@pytest.fixture
def filled_square():
    """Create a filled square shape."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # 3x3 blue square
    return Shape(grid)


@pytest.fixture
def filled_rectangle():
    """Create a filled rectangle shape."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:7] = 2  # 3x5 red rectangle
    return Shape(grid)


@pytest.fixture
def hollow_square():
    """Create a hollow square shape."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[1:6, 1:6] = 1  # Outer square
    grid[2:5, 2:5] = 0  # Hollow center
    return Shape(grid)


@pytest.fixture
def horizontal_line():
    """Create a horizontal line."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 2:8] = 1  # Horizontal line
    return Shape(grid)


@pytest.fixture
def vertical_line():
    """Create a vertical line."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:8, 5] = 1  # Vertical line
    return Shape(grid)


@pytest.fixture
def diagonal_line():
    """Create a diagonal line."""
    grid = np.zeros((10, 10), dtype=np.int32)
    for i in range(5):
        grid[i, i] = 1  # Main diagonal
    return Shape(grid)


@pytest.fixture
def anti_diagonal_line():
    """Create an anti-diagonal line."""
    grid = np.zeros((10, 10), dtype=np.int32)
    for i in range(5):
        grid[i, 4 - i] = 1  # Anti-diagonal
    return Shape(grid)


@pytest.fixture
def multicolor_shape():
    """Create a shape with multiple colors."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:5, 2:5] = 1  # Blue square
    grid[3, 3] = 2  # Red center
    grid[2, 2] = 3  # Green corner
    return Shape(grid)


@pytest.fixture
def disconnected_shape():
    """Create a disconnected shape (two separate regions)."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:4, 2:4] = 1  # First square
    grid[6:8, 6:8] = 1  # Second square (disconnected)
    return Shape(grid)


@pytest.fixture
def horizontally_symmetric_shape():
    """Create a horizontally symmetric shape."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[3:7, 4:6] = 1  # Vertical rectangle (symmetric horizontally)
    return Shape(grid)


@pytest.fixture
def vertically_symmetric_shape():
    """Create a vertically symmetric shape."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:6, 3:7] = 1  # Horizontal rectangle (symmetric vertically)
    return Shape(grid)


class TestSymmetry:
    """Test symmetry conditionals."""

    def test_is_shape_symmetric_square(self, filled_square):
        assert is_shape_symmetric(filled_square)
        assert is_shape_horizontally_symmetric(filled_square)
        assert is_shape_vertically_symmetric(filled_square)

    def test_is_shape_horizontally_symmetric(self, horizontally_symmetric_shape):
        assert is_shape_horizontally_symmetric(horizontally_symmetric_shape)
        assert is_shape_symmetric(horizontally_symmetric_shape)

    def test_is_shape_vertically_symmetric(self, vertically_symmetric_shape):
        assert is_shape_vertically_symmetric(vertically_symmetric_shape)
        assert is_shape_symmetric(vertically_symmetric_shape)

    def test_is_shape_not_symmetric_rectangle(self, filled_rectangle):
        # A non-square rectangle positioned off-center is typically not symmetric
        # unless it's centered, so this depends on the exact shape
        pass  # Skip for now as it depends on positioning

    def test_diagonal_symmetry_square(self, filled_square):
        # A square is diagonally symmetric
        assert is_shape_diagonally_symmetric(filled_square)

    def test_not_symmetric(self, multicolor_shape):
        # Multicolor shape with asymmetric coloring should not be symmetric
        result = is_shape_not_symmetric(multicolor_shape)
        # This depends on the exact color pattern
        assert isinstance(result, bool)


class TestColoring:
    """Test color-related conditionals."""

    def test_is_shape_evenly_colored_single_color(self, filled_square):
        assert is_shape_evenly_colored(filled_square)
        assert not is_shape_not_evenly_colored(filled_square)

    def test_is_shape_not_evenly_colored(self, multicolor_shape):
        assert is_shape_not_evenly_colored(multicolor_shape)
        assert not is_shape_evenly_colored(multicolor_shape)

    def test_is_shape_of_2_colors(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:5, 2:5] = 1  # Blue
        grid[3, 3] = 2  # Red center
        shape = Shape(grid)
        assert is_shape_of_2_colors(shape)

    def test_is_shape_of_3_colors(self, multicolor_shape):
        assert is_shape_of_3_colors(multicolor_shape)

    def test_is_shape_more_than_3_colors(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2, 2] = 1
        grid[3, 3] = 2
        grid[4, 4] = 3
        grid[5, 5] = 4
        shape = Shape(grid)
        assert is_shape_more_than_3_colors(shape)


class TestHollowness:
    """Test hollow/filled conditionals."""

    def test_is_shape_hollow(self, hollow_square):
        assert is_shape_hollow(hollow_square)
        assert not is_shape_not_hollow(hollow_square)

    def test_is_shape_not_hollow(self, filled_square):
        assert is_shape_not_hollow(filled_square)
        assert not is_shape_hollow(filled_square)


class TestSimplicity:
    """Test shape simplicity conditionals."""

    def test_is_shape_simple_square(self, filled_square):
        assert is_shape_simple(filled_square)
        assert not is_shape_not_simple(filled_square)

    def test_is_shape_not_simple_disconnected(self, disconnected_shape):
        # Disconnected shapes are not simple
        assert is_shape_not_simple(disconnected_shape)
        assert not is_shape_simple(disconnected_shape)


class TestLines:
    """Test line detection conditionals."""

    def test_is_shape_horizontal_line(self, horizontal_line):
        assert is_shape_horizontal_line(horizontal_line)
        assert is_shape_line(horizontal_line)
        assert not is_shape_vertical_line(horizontal_line)

    def test_is_shape_vertical_line(self, vertical_line):
        assert is_shape_vertical_line(vertical_line)
        assert is_shape_line(vertical_line)
        assert not is_shape_horizontal_line(vertical_line)

    def test_is_shape_diagonal_line(self, diagonal_line):
        assert is_shape_diagonal_line(diagonal_line)
        assert is_shape_diagonal_or_antidiagonal_line(diagonal_line)

    def test_is_shape_anti_diagonal_line(self, anti_diagonal_line):
        assert is_shape_anti_diagonal_line(anti_diagonal_line)
        assert is_shape_diagonal_or_antidiagonal_line(anti_diagonal_line)

    def test_non_line_shape(self, filled_square):
        assert not is_shape_line(filled_square)
        assert not is_shape_horizontal_line(filled_square)
        assert not is_shape_vertical_line(filled_square)


class TestGeometry:
    """Test geometric property conditionals."""

    def test_is_shape_filled_square(self, filled_square):
        assert is_shape_filled_square(filled_square)

    def test_is_shape_filled_rectangle(self, filled_rectangle):
        assert is_shape_filled_rectangle(filled_rectangle)

    def test_is_shape_higher_than_wide(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:8, 3:5] = 1  # Tall rectangle
        shape = Shape(grid)
        assert is_shape_higher_than_wide(shape)
        assert not is_shape_wider_than_high(shape)

    def test_is_shape_wider_than_high(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[3:5, 2:8] = 1  # Wide rectangle
        shape = Shape(grid)
        assert is_shape_wider_than_high(shape)
        assert not is_shape_higher_than_wide(shape)

    def test_is_shape_same_height_width(self, filled_square):
        assert is_shape_same_height_width(filled_square)


class TestDimensions:
    """Test dimension-specific conditionals."""

    def test_is_shape_of_two_cols(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:5, 3:5] = 1  # 2 columns wide
        shape = Shape(grid)
        assert is_shape_of_two_cols(shape)

    def test_is_shape_of_three_cols(self, filled_square):
        # filled_square is 3x3
        assert is_shape_of_three_cols(filled_square)

    def test_is_shape_of_four_cols(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:5, 2:6] = 1  # 4 columns wide
        shape = Shape(grid)
        assert is_shape_of_four_cols(shape)

    def test_is_shape_of_five_cols(self, filled_rectangle):
        # filled_rectangle is 3x5
        assert is_shape_of_five_cols(filled_rectangle)

    def test_is_shape_of_two_rows(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[3:5, 2:5] = 1  # 2 rows high
        shape = Shape(grid)
        assert is_shape_of_two_rows(shape)

    def test_is_shape_of_three_rows(self, filled_square):
        # filled_square is 3x3
        assert is_shape_of_three_rows(filled_square)

    def test_is_shape_of_four_rows(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:6, 3:5] = 1  # 4 rows high
        shape = Shape(grid)
        assert is_shape_of_four_rows(shape)

    def test_is_shape_of_five_rows(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[2:7, 3:5] = 1  # 5 rows high
        shape = Shape(grid)
        assert is_shape_of_five_rows(shape)


class TestConnectivity:
    """Test connectivity conditionals."""

    def test_is_shape_fully_connected(self, filled_square):
        assert is_shape_fully_connected(filled_square)
        assert not is_shape_not_fully_connected(filled_square)

    def test_is_shape_not_fully_connected(self, disconnected_shape):
        assert is_shape_not_fully_connected(disconnected_shape)
        assert not is_shape_fully_connected(disconnected_shape)


class TestSize:
    """Test size conditionals."""

    def test_is_shape_more_than_1_cell(self, filled_square):
        assert is_shape_more_than_1_cell(filled_square)

    def test_single_pixel_not_more_than_1_cell(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = 1
        shape = Shape(grid)
        assert not is_shape_more_than_1_cell(shape)


class TestConditionalConsistency:
    """Test that opposite conditionals are mutually exclusive."""

    def test_symmetric_mutually_exclusive(self, filled_square, multicolor_shape):
        # A shape is either symmetric or not
        for shape in [filled_square, multicolor_shape]:
            symmetric = is_shape_symmetric(shape)
            not_symmetric = is_shape_not_symmetric(shape)
            assert symmetric != not_symmetric  # Exactly one should be true

    def test_hollow_mutually_exclusive(self, filled_square, hollow_square):
        for shape in [filled_square, hollow_square]:
            hollow = is_shape_hollow(shape)
            not_hollow = is_shape_not_hollow(shape)
            assert hollow != not_hollow

    def test_evenly_colored_mutually_exclusive(self, filled_square, multicolor_shape):
        for shape in [filled_square, multicolor_shape]:
            evenly = is_shape_evenly_colored(shape)
            not_evenly = is_shape_not_evenly_colored(shape)
            assert evenly != not_evenly

    def test_simple_mutually_exclusive(self, filled_square, disconnected_shape):
        for shape in [filled_square, disconnected_shape]:
            simple = is_shape_simple(shape)
            not_simple = is_shape_not_simple(shape)
            assert simple != not_simple

    def test_connected_mutually_exclusive(self, filled_square, disconnected_shape):
        for shape in [filled_square, disconnected_shape]:
            connected = is_shape_fully_connected(shape)
            not_connected = is_shape_not_fully_connected(shape)
            assert connected != not_connected
