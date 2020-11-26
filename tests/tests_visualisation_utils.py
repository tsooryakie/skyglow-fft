import unittest
import numpy as np
import numpy.testing as np_tests
import visualisation_utils


class VisualisationUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_array = np.array(
            [
                [0.0, 13.86294361, 21.97224577],
                [0.0, 13.86294361, 21.97224577],
                [0.0, 13.86294361, 21.97224577],
            ]
        )

        return

    def test_log_transformation_happy(self) -> None:
        """
        This test checks that the log transformation is applied to the ndarray correctly
        when an appropriate argument is supplied to the log_transformation function.
        :return:
        """
        self.array_to_test = visualisation_utils.log_transformation(
            np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        )
        np_tests.assert_almost_equal(
            actual=self.array_to_test, desired=self.expected_array
        )

        return

    def test_log_transformation_sad(self) -> None:
        """
        This test checks that an expected exception is raised when the log_transformation function
        is not supplied with a correct argument.
        :return:
        """
        with self.assertRaises(TypeError) as cm:
            self.array_to_test = visualisation_utils.log_transformation(None)
            self.assertEqual(
                str(cm.exception), "TypeError: bad operand type for abs(): 'NoneType'"
            )

        return

    def test_log_transformation_lazy(self) -> None:
        """
        This function checks that an expected exception is raised when the log_transformation function
        is supplied with no argument.
        :return:
        """
        with self.assertRaises(TypeError) as cm:
            self.array_to_test = visualisation_utils.log_transformation()
            self.assertEqual(
                str(cm.exception),
                "TypeError: log_transformation() missing 1 required positional argument: 'frequency_domain_array'",
            )
        return

    def test_visualise_frequency_transforms_sad(self):
        """
        This test checks that the visualise_frequency_transforms function fails in an expected manner
        when supplied with incorrect arguments.
        :return:
        """
        with self.assertRaises(TypeError) as cm:
            visualisation_utils.visualise_transforms(None, None, None, None, None)
            self.assertEqual(
                str(cm.exception),
                "TypeError: Image data of dtype object cannot be converted to float",
            )
        return

    def test_visualise_frequency_transforms_lazy(self):
        """
        This test checks that the visualise_frequency_transforms function fails in an expected manner
        when supplied with no arguments.
        :return:
        """

        with self.assertRaises(TypeError) as cm:
            visualisation_utils.visualise_transforms()
            self.assertEqual(
                str(cm.exception),
                "TypeError: visualise_transforms() missing 5 required positional arguments: 'viirs_raster',"
                " 'viirs_log_spectrum', 'distance_kernel', 'distance_log_spectrum', and 'combined_frequency_shift'",
            )
        return

    def main(self) -> None:
        unittest.main()

    if __name__ == "__main__":
        main()
