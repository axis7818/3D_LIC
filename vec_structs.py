from math import floor, sqrt

from vec_ops import build_average_field

class Point(object):
    """
    Represnts a 3 dimensional cartesian coordinate.
    """

    def __init__(self, x, y, z):

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __eq__(self, other):

        return issubclass(type(other), Point) and self.x == other.x \
            and self.y == other.y and self.z == other.z

    def __ne__(self, other):

        return not self == other

    def __add__(self, other):

        if not issubclass(type(other), Point):
            raise ValueError

        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __str__(self):

        return "<{}: x={}, y={}, z={}>".format(self.__class__.__name__,
            self.x, self.y, self.z)

    def is_zero(self):

        return self.x == 0 and self.y == 0 and self.z == 0

    def as_indices(self):
        return int(self.x), int(self.y), int(self.z)

class Vector(Point):
    """
    Represents a 3 dimensional cartesian vector.
    """

    def scale(self, val):

        self.x *= val
        self.y *= val
        self.z *= val

    def length(self):

        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):

        length = self.length()
        if length != 0:
            self.scale(1.0 / length)

class VectorField(object):
    """
    Wraps a numpy array of shape (x, y, z, 3).
    """

    def __init__(self, data):

        if data.shape[3] != 3:
            raise ValueError

        self.field = data
        self.shape = data.shape[0:3]

    def __str__(self):
        return "<{}: shape={}>".format(self.__class__.__name__, self.shape)

    def contains_point(self, point):

        return 0 <= point.x and point.x < self.shape[0] \
            and 0 <= point.y and point.y < self.shape[1] \
            and 0 <= point.z and point.z < self.shape[2]

    def get_raw(self, x, y, z):

        assert self.contains_point(Point(x, y, z))
        return Vector(x, y, z)

    def get(self, point):
        """
        Retrieves an interpolated vector from the field.
        """

        assert self.contains_point(point)

        # indices in each dimension of the cell
        i = int(floor(point.x))
        j = int(floor(point.y))
        k = int(floor(point.z))
        i1 = i + 1 if i < self.shape[0] - 1 else i
        j1 = j + 1 if j < self.shape[1] - 1 else j
        k1 = k + 1 if k < self.shape[2] - 1 else k

        # weights for each dimension
        u = point.x - i
        v = point.y - j
        w = point.z - k

        # trilinear interpolation on each vector component
        vector = (u * v * w * self.field[i,j,k]) \
            + (u * v * (1.0 - w) * self.field[i,j,k1]) \
            + (u * (1.0 - v) * w * self.field[i,j1,k]) \
            + (u * (1.0 - v) * (1.0 - w) * self.field[i,j1,k1]) \
            + ((1.0 - u) * v * w * self.field[i1,j,k]) \
            + ((1.0 - u) * v * (1.0 - w) * self.field[i1,j,k1]) \
            + ((1.0 - u) * (1.0 - v) * w * self.field[i1,j1,k]) \
            + ((1.0 - u) * (1.0 - v) * (1.0 - w) * self.field[i1,j1,k1])

        return Vector(*vector)

    def _do_integration(self, seed, delta_t, steps, forwards):

        result = []
        direction = 1.0 if forwards else -1.0

        point = seed
        while steps > 0 and self.contains_point(point):

            result.append(point)

            offset = self.get(point)
            if offset.is_zero():
                break

            offset.scale(direction * delta_t)

            point += offset
            steps -= 1

        # Special indexing to remove the seed from the list
        return result[1:] if forwards else result[:0:-1]

    def make_streamline(self, seed, delta_t, steps, forwards=True,
            backwards=True):
        """
        Create a streamline in the field based on Euler integration.

        Args:
            seed: a Point to start the streamline.
            delta_t: the time step between points.
            steps: the number of integration steps to perform in
                either direction.
            forwards: whether or not to generate the stream line in the
                forwards direction.
            backwards: whether or not to generate the stream line in the
                backwards direction.

        Returns:
            A list of Points that descibe the streamline from back to front.
        """

        assert self.contains_point(seed)
        assert delta_t > 0 and steps > 0
        assert forwards or backwards

        front = self._do_integration(seed, delta_t, steps, True) \
            if forwards else []

        back = self._do_integration(seed, delta_t, steps, False) \
            if backwards else []

        return back + [seed] + front

    @staticmethod
    def build_average_field(fields):

        assert len(fields) > 0

        result = np.zeros(*(fields[0].shape + [3]))

        for field in fields:
            result += field

        result /= 1.0 * len(fields)

        return result

class VectorEnsemble(object):

    def __init__(self, fields):

        self.fields = fields
        self.average_field = VectorField.build_average_field(fields)

        self.shape = fields[0].shape
        self.member_count = len(fields)

    def __str__(self):

        return "<{}: shape={}, member_count: {}>".format(
            self.__class__.__name__, self.shape, self.member_count)
