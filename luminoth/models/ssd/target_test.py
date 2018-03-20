import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.ssd.ssd_target import SSDTarget


class TargetTest(tf.test.TestCase):

    def setUp(self):
        super(TargetTest, self).setUp()
        self._config = EasyDict({
            'hard_negative_ratio': 2.,
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.2,
            'background_threshold_low': 0.0
        })
        self._equality_delta = 1e-03
        self._shared_model = SSDTarget(self._config, seed=0)
        tf.reset_default_graph()

    def _run_ssd_target(self, model, probs, all_anchors, gt_boxes):
        """Runs an instance of SSDTarget

        Args:
            model: an SSDTarget model.
            gt_boxes: a Tensor holding the ground truth boxes, with shape
                (num_gt, 5). The last value is the class label.
            proposed_boxes: a Tensor holding the proposed boxes. Its shape is
                (num_proposals, 4). The first value is the batch number.

        Returns:
            The tuple returned by SSDTarget._build().
        """
        ssd_target_net = model(probs, all_anchors, gt_boxes)
        with self.test_session() as sess:
            return sess.run(ssd_target_net)

    def _create_a_2_gt_box_sample(self):
        """Creates a testing sample

        Creates sample with 2 gt_boxes, with 3 true predictions and 7 false.
        Contains 3 classes.
        gt_box format = [xmin, ymin, xmax, ymax, label]
        anchors format = [xmin, ymin, xmax, ymax]
        """
        gt_boxes = np.array(
            [[10, 10, 50, 50, 0], [120, 200, 250, 280, 2]], dtype=np.float32
        )

        all_anchors = np.array([
            [40.341, 40.3, 80.0, 80.],  # Positive, overlap<tresh -> least worst
            [0., 0., 15., 15.],  # Negative, has some overlap with gt_box1
            [45., 45., 125., 220.],  # Negative, overlaps both gt_boxes
            [110., 190., 240., 270.],  # Positive, overlap>tresh with gt_box1
            [130., 210., 240., 270.],  # Positive, overlap>tresh with gt_box1
            [1., 1., 299., 299.],  # Negative, covers both gt_boxes
            [220., 50., 280., 200.],  # Negative, random
            [200., 210., 210., 220.],  # Negative, random
            [10., 20., 10., 25.],  # Negative, random
            [20., 10., 25., 10.],  # Negative, random
        ])

        probs = np.array([
            [.1, .2, .7],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4]
        ], dtype=np.float32)

        target_offsets = np.zeros_like(all_anchors)
        target_offsets[0, :] = self.get_offset(gt_boxes[0], all_anchors[0])
        target_offsets[3, :] = self.get_offset(gt_boxes[1], all_anchors[3])
        target_offsets[4, :] = self.get_offset(gt_boxes[1], all_anchors[4])

        target_classes = np.array([1., 0., 0., 3., 3., 0., 0., -1., 0., 0.],
                                  dtype=np.float32)
        return probs, all_anchors, gt_boxes, target_offsets, target_classes

    def _create_a_1_gt_box_sample(self):
        """Creates a testing sample

        Creates sample with 1 gt_box, with 2 true predictions and 8 false.
        Contains 3 classes.
        gt_box format = [xmin, ymin, xmax, ymax, label]
        anchors format = [xmin, ymin, xmax, ymax]
        """
        gt_boxes = np.array([[20, 20, 80, 80, 0]], dtype=np.float32)
        all_anchors = np.array([
            [31.341, 27.3, 75.0, 75.],  # Positive, overlap<tresh -> least worst
            [0., 0., 15., 15.],  # Negative, random
            [45., 45., 125., 220.],  # Negative, random
            [110., 190., 240., 270.],  # Negative, random
            [130., 210., 240., 270.],  # Negative, random
            [1., 1., 299., 299.],  # Negative, random
            [220., 50., 280., 200.],  # Negative, random
            [200., 210., 210., 220.],  # Negative, random
            [10., 20., 10., 25.],  # Negative, random
            [20., 10., 25., 10.],  # Negative, random
        ])
        probs = np.array([
            [.1, .2, .7],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4],
            [.8, .1, .1],
            [.3, .3, .4],
            [.1, .5, .4]
        ], dtype=np.float32)

        target_offsets = np.zeros_like(all_anchors)
        target_offsets[0, :] = self.get_offset(gt_boxes[0], all_anchors[0])

        target_classes = np.array(
            [1., -1., -1., 0., -1., -1., 0., -1., -1., -1.], dtype=np.float32
        )
        return probs, all_anchors, gt_boxes, target_offsets, target_classes

    def get_offset(self, gt_box, box):
        gt_box = self.convert_to_width_height_center(gt_box)
        box = self.convert_to_width_height_center(box)

        offset_x = (gt_box['x'] - box['x']) / box['w']
        offset_y = (gt_box['y'] - box['y']) / box['h']
        offset_w = np.log(gt_box['w'] / box['w'])
        offset_h = np.log(gt_box['h'] / box['h'])

        return offset_x, offset_y, offset_w, offset_h

    def convert_to_width_height_center(self, box):
        # The next two `+ 1`s are to keep this consistent with
        # `get_width_upright` in luminoth/utils/bbox_transform_tf.py, though
        # it does not make much sense
        width = box[2] - box[0] + 1
        height = box[3] - box[1] + 1
        center_x = box[0] + width / 2
        center_y = box[1] + height / 2
        return {'w': width, 'h': height, 'x': center_x, 'y': center_y}

    def testHardCase(self):
        """Tests a hard case with batch_size == 2"""

        # Disgusting code ahead, watch out!
        probs_1, all_anchors_1, gt_boxes_1, target_offsets_1, target_classes_1 \
            = self._create_a_1_gt_box_sample()
        gt_boxes_1 = np.pad(gt_boxes_1, [(0, 1), (0, 0)], 'constant')

        probs_2, all_anchors_2, gt_boxes_2, target_offsets_2, target_classes_2 \
            = self._create_a_2_gt_box_sample()

        probs = np.stack((probs_1, probs_2))
        all_anchors = np.stack((all_anchors_1, all_anchors_2))
        gt_boxes = np.stack((gt_boxes_1, gt_boxes_2))
        target_offsets = np.stack((target_offsets_1, target_offsets_2))
        target_classes = np.stack((target_classes_1, target_classes_2))

        class_targets, bbox_offsets_targets = self._run_ssd_target(
            self._shared_model, probs, all_anchors, gt_boxes
        )
        print("GOT THROUGH THE TENSORFLOW MODEL, YEAHHH !!!!")
        np.testing.assert_allclose(
            target_offsets, bbox_offsets_targets, rtol=self._equality_delta)
        np.testing.assert_allclose(
            target_classes, class_targets, rtol=self._equality_delta)


if __name__ == '__main__':
    tf.test.main()
