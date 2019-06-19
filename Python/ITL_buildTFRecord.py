import collections
import six
import math
import os.path

import sys
import build_data
import tensorflow as tf

itl_anodir = 'e:/temp/itl/models/research/deeplab/datasets/itl/segmentationclass/'
ITL_imageDir = 'e:/temp/itl/models/research/deeplab/datasets/itl/jpegimages/'
itl_listdir = 'e:/temp/itl/models/research/deeplab/datasets/itl/imagesets/'
ITL_outputDir = 'e:/temp/itl/models/research/deeplab/datasets/itl/tfrecord/'
itl_numshards = 4

itl_datasetsplits = tf.gfile.Glob(os.path.join(itl_listdir, '*.txt'))

for itl_datasetsplit in itl_datasetsplits:
	itl_dataset = os.path.basename(itl_datasetsplit)[:-4]
	print('Processing ' + itl_dataset)
	ITL_filenames = [x.strip('\n') for x in open(itl_datasetsplit, 'r')]
	itl_numimages = len(ITL_filenames)
	ITL_numPerShard = int(math.ceil(itl_numimages / float(itl_numshards)))

	ITL_imageReader = build_data.ImageReader('jpeg', channels=3)
	ITL_labelReader = build_data.ImageReader('png', channels=1)

	for ITL_shardId in range(itl_numshards):
		ITL_outputFilename = os.path.join(ITL_outputDir, '%s-%05d-of-%05d.tfrecord' % (itl_dataset, ITL_shardId, itl_numshards))
		with tf.python_io.TFRecordWriter(ITL_outputFilename) as tfrecord_writer:
			ITL_startIdx = ITL_shardId * ITL_numPerShard
			ITL_endIdx = min((ITL_shardId + 1) * ITL_numPerShard, itl_numimages)
			for i in range(ITL_startIdx, ITL_endIdx):
				print('\r>> Converting image %d/%d shard %d' % (i + 1, len(ITL_filenames), ITL_shardId))
				sys.stdout.flush()
				ITL_imageFilename = os.path.join(ITL_imageDir, ITL_filenames[i] + '.jpg')
				ITL_imageData = tf.gfile.FastGFile(ITL_imageFilename, 'rb').read()
				height, width = ITL_imageReader.read_image_dims(ITL_imageData)
				ITL_anoFilename = os.path.join(itl_anodir, ITL_filenames[i] + '.png')
				ITL_anoData = tf.gfile.FastGFile(ITL_anoFilename, 'rb').read()
				seg_height, seg_width = ITL_labelReader.read_image_dims(ITL_anoData)
				if height != seg_height or width != seg_width:
					raise RuntimeError('Shape mismatched between image and label.')
				ITL_example = build_data.image_seg_to_tfexample(ITL_imageData, ITL_filenames[i], height, width, ITL_anoData)
				tfrecord_writer.write(ITL_example.SerializeToString())
		print('\n')
		sys.stdout.flush()
