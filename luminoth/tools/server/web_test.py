import numpy as np
import tensorflow as tf

from PIL import Image
from luminoth.models import get_model
from luminoth.tools.server.web import get_prediction


class WebTest(tf.test.TestCase):
    # TODO When the image size has big dimensions like (1024, 1024, 3),
    # Travis fails during this test, probably ran out of memory. Using an build
    # environment with more memory all works fine.
    def testFasterRCNN(self):
        """
        Tests the FasterRCNN's predict
        """
        model_class = get_model('fasterrcnn')

        image_resize = model_class.base_config.dataset.image_preprocessing
        image_resize_min = image_resize.min_size
        image_resize_max = image_resize.max_size

        # Does a prediction without resizing the image
        image = Image.fromarray(
            np.random.randint(
                low=0, high=255,
                size=(image_resize_min, image_resize_max, 3)
            ).astype(np.uint8)
        )

        results = get_prediction('fasterrcnn', image)

        # Check that scale_factor and inference_time are corrects values
        self.assertEqual(results['scale_factor'], 1.0)
        self.assertGreaterEqual(results['inference_time'], 0)

        # Check that objects, labels and probs aren't None
        self.assertIsNotNone(results['objects'])
        self.assertIsNotNone(results['objects_labels'])
        self.assertIsNotNone(results['objects_labels_prob'])

        # Does a prediction resizing the image
        image = Image.fromarray(
            np.random.randint(
                low=0, high=255,
                size=(image_resize_min, image_resize_max + 1, 3)
            ).astype(np.uint8)
        )

        results = get_prediction('fasterrcnn', image)

        # Check that scale_factor and inference_time are corrects values
        self.assertNotEqual(1.0, results['scale_factor'])
        self.assertGreaterEqual(results['inference_time'], 0)

        # Check that objects, labels and probs aren't None
        self.assertIsNotNone(results['objects'])
        self.assertIsNotNone(results['objects_labels'])
        self.assertIsNotNone(results['objects_labels_prob'])


if __name__ == '__main__':
    tf.test.main()