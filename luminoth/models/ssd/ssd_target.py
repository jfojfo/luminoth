import tensorflow as tf
import sonnet as snt

from luminoth.utils.bbox_transform_tf import encode
from luminoth.utils.bbox_overlap import bbox_overlap_tf_batch


class SSDTarget(snt.AbstractModule):
    """TODO
    """
    def __init__(self, config, seed=None, name='ssd_target'):
        """
        Args:
            num_classes: Number of possible classes.
            config: Configuration object for RCNNTarget.
        """
        super(SSDTarget, self).__init__(name=name)
        # Ratio of foreground vs background for the minibatch.
        self._foreground_fraction = config.hard_negative_ratio
        # IoU lower threshold with a ground truth box to be considered that
        # specific class.
        self._foreground_threshold = config.foreground_threshold
        # High and low treshold to be considered background.
        self._background_threshold_high = config.background_threshold_high
        self._background_threshold_low = config.background_threshold_low
        self._seed = seed

    def _build(self, probs, all_anchors, gt_boxes):
        """
        Args:
            all_anchors: A Tensor with anchors for all of SSD's features.
                The shape of the Tensor is (num_anchors, 4).
            gt_boxes: A Tensor with the ground truth boxes for the image.
                The shape of the Tensor is (num_gt, 5), having the truth label
                as the last value for each box.
        Returns:
            class_targets: Either a truth value of the anchor (a value
                between 0 and num_classes, with 0 being background), or -1 when
                the anchor is to be ignored in the minibatch.
                The shape of the Tensor is (num_anchors, 1).
            bbox_offsets_targets: A bounding box regression target for each of
                the anchors that have and greater than zero label. For every
                other anchors we return zeros.
                The shape of the Tensor is (num_anchors, 4).
        """

        all_anchors = tf.cast(all_anchors, tf.float32)
        gt_boxes = tf.cast(gt_boxes, tf.float32)

        overlaps = bbox_overlap_tf_batch(all_anchors, gt_boxes[:, :, :4])
        # overlaps now contains (num_anchors, num_gt_boxes) with the IoU of
        # anchor P and ground truth box G in overlaps[P, G]

        # For each overlap there is two possible outcomes for labelling by now:
        #  if max(iou) > config.foreground_threshold then we label with
        #      the highest IoU in overlap.
        #  elif (config.background_threshold_low <= max(iou) <=
        #       config.background_threshold_high) we label with background (0)
        #  else we label with -1.

        # max_overlaps gets, for each anchor, the index in which we can
        # find the gt_box with which it has the highest overlap.
        max_overlaps = tf.reduce_max(overlaps, axis=2)

        # Get the index of the best gt_box for each anchor.
        best_gtbox_for_anchors_idx = tf.argmax(overlaps, axis=2)
        # Having the index of the gt bbox with the best label we need to get
        # the label for each gt box and sum 1 to it because 0 is used for
        # background.
        best_fg_labels_for_anchors = tf.add(
            tf.gather(gt_boxes[:, :, 4], best_gtbox_for_anchors_idx),
            1.
        )
        # README:
        # WE BASICALLY HAVE 2 OPTIONS: REWRITE IT DIFFERENTLY SO WE DONT NEED TO GATHER
        #                              MAKE A HELPER FUNCTION TO CONVERT TO REAL INDEXES

        tf.InteractiveSession(); from IPython import embed; embed(display_banner=False)
        iou_is_fg = tf.greater_equal(
            max_overlaps, self._foreground_threshold
        )

        # We are going to label each anchor based on the IoU with
        # `gt_boxes`. Start by filling the labels with -1, marking them as
        # unknown.
        anchors_label_shape = tf.gather(tf.shape(all_anchors), [0, 1])
        anchors_label = tf.fill(
            dims=anchors_label_shape,
            value=-1.
        )
        # We update anchors_label with the value in
        # best_fg_labels_for_anchors only when the box is foreground.
        # TODO: Replace with a sparse_to_dense with -1 default_value
        anchors_label = tf.where(
            condition=iou_is_fg,
            x=best_fg_labels_for_anchors,
            y=anchors_label
        )

        # ############# Select the best anchor for each gt ###############
        best_anchor_idxs = tf.argmax(overlaps, axis=0)
        # Set the indices in best_anchor_idxs to True, and the rest to false.
        # tf.sparse_to_dense is used because we know the set of indices which
        # we want to set to True, and we know the rest of the indices
        # should be set to False. That's exactly the use case of
        # tf.sparse_to_dense.
        is_best_box = tf.sparse_to_dense(
            sparse_indices=best_anchor_idxs,
            sparse_values=True, default_value=False,
            output_shape=tf.cast(anchors_label_shape, tf.int64),
            validate_indices=False
        )
        # Now we need to find the anchors that are the best for each of the
        # gt_boxes. We overwrite the previous anchors_label with this
        # because setting the best anchor for each gt_box has priority.
        best_anchors_gt_labels = tf.sparse_to_dense(
            sparse_indices=best_anchor_idxs,
            sparse_values=gt_boxes[:, 4] + 1,
            default_value=-1,
            output_shape=tf.cast(anchors_label_shape, tf.int64),
            validate_indices=False,
            name="get_right_labels_for_bestboxes"
        )
        anchors_label = tf.where(
            condition=is_best_box,
            x=best_anchors_gt_labels,
            y=anchors_label,
            name="update_labels_for_bestbox_anchors"
        )
        # ################################################################

        # Use the worst backgrounds (the bgs whose probability of being fg is
        # the greatest).
        cls_probs = probs[:, :, 1:]
        max_cls_probs = tf.reduce_max(cls_probs, axis=2)
        import ipdb; ipdb.set_trace()

        # Exclude boxes with IOU > `background_threshold_high` with any GT.
        iou_less_than_bg_tresh_high_filter = tf.less_equal(
            max_overlaps, self._background_threshold_high
        )
        bg_anchors = tf.less_equal(anchors_label, 0)
        bg_overlaps_filter = tf.logical_and(
            iou_less_than_bg_tresh_high_filter, bg_anchors
        )

        max_cls_probs = tf.where(
            condition=bg_overlaps_filter,
            x=max_cls_probs,
            y=tf.fill(dims=anchors_label_shape, value=-1.),
        )

        # We calculate up to how many backgrounds we desire based on the
        # final number of foregrounds and the hard minning ratio.
        num_fg_mask = tf.greater(anchors_label, 0.0)
        num_fg = tf.cast(tf.count_nonzero(num_fg_mask), tf.float32)

        num_bg = tf.cast(num_fg * self._foreground_fraction, tf.int32)
        top_k_bg = tf.nn.top_k(max_cls_probs, k=num_bg)

        set_bg = tf.sparse_to_dense(
            sparse_indices=top_k_bg.indices,
            sparse_values=True, default_value=False,
            output_shape=anchors_label_shape,
            validate_indices=False
        )

        anchors_label = tf.where(
            condition=set_bg,
            x=tf.fill(dims=anchors_label_shape, value=0.),
            y=anchors_label
        )

        """
        Next step is to calculate the proper bbox targets for the labeled
        anchors based on the values of the ground-truth boxes.
        We have to use only the anchors labeled >= 1, each matching with
        the proper gt_boxes
        """

        # Get the ids of the anchors that mater for bbox_target comparison.
        is_anchor_with_target = tf.greater(
            anchors_label, 0
        )
        anchors_with_target_idx = tf.where(
            condition=is_anchor_with_target
        )
        # Get the corresponding ground truth box only for the anchors with
        # target.
        gt_boxes_idxs = tf.gather(
            best_gtbox_for_anchors_idx,
            anchors_with_target_idx
        )
        # Get the values of the ground truth boxes.
        anchors_gt_boxes = tf.gather_nd(
            gt_boxes[:, :4], gt_boxes_idxs
        )
        # We create the same array but with the anchors
        anchors_with_target = tf.gather_nd(
            all_anchors,
            anchors_with_target_idx
        )
        # We create our targets with bbox_transform
        bbox_targets = encode(
            anchors_with_target,
            anchors_gt_boxes,
        )

        # We unmap targets to anchor_labels (containing the length of
        # anchors)
        bbox_targets = tf.scatter_nd(
            indices=anchors_with_target_idx,
            updates=bbox_targets,
            shape=tf.cast(tf.shape(all_anchors), tf.int64)
        )

        return anchors_label, bbox_targets
