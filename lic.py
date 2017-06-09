import numpy as np

def lic(field, time_step, detail, show_progress=False):
    """
    Generates a 3 dimensional LIC of values between 0 and 1.

    Args:
        field: The VectorField instance to make a LIC from.
        time_step: The time step to use in the streamlines.
        detail: The length of streamline integration. Higher values yield a more
            detailed LIC, but take longer to create.

    Returns:
        A numpy array of the same shape as the field and type float16
    """

    noise = np.random.rand(*field.shape)
    result = np.empty(field.shape, dtype=np.float16)

    first = True
    total_seeds = float(noise.size)

    for i in range(noise.shape[0]):
        i_count = i * noise.shape[1] * noise.shape[2]

        for j in range(noise.shape[1]):
            j_count = j * noise.shape[2]

            for k in range(noise.shape[2]):

                count = k + i_count * j_count
                progress = float(count) / total_seeds
                if show_progress:
                    print_progress(progress, first)
                first = False

                streamline = field.make_streamline(Point(i, j, k), time_step,
                    detail)

                # TODO: weight the average
                value = 0.0
                for point in streamline:
                    value += noise[point.as_indices()]
                value /= 1.0 * len(streamline)

                result[i,j,k] = value

    return result
