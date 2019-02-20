import os
import sys
sys.path.append('.')
import glob
import cv2
import numpy as np
caffe_root = 'lib/caffe-action/'
sys.path.insert(0,caffe_root + 'python')
from caffe.proto import caffe_pb2
from PIL import Image
from sklearn.metrics import accuracy_score
import caffe
import matplotlib.pyplot as plt
GLOG_minloglevel=2
from pyActionRecog.utils.video_funcs import default_aggregation_func
sys.path.append(os.path.join('./lib/caffe-action/', 'python'))

# Some params
NUM_FR_PER_VID = 25
BATCH_SIZE = 5
EPS = 8
ATTACK = True
ALPHA = 1
NUM_ITER = 0
ATTACK_RATE = 2
VIDEO_LIST = "/home/anhnguyen/action_recog/UCF-101/ucf101_split1_testVideos.txt"
VIDEO_PATH = "/media/anhnguyen/4C35-FA26/UCF101-split1"
VIDEO_AE_PATH = "/home/anhnguyen/action_recog/UCF-101/tsn_fgsm_iter_" + str(EPS) + "_rate_" + str(ATTACK_RATE)
NET_PROTOTXT = "/home/anhnguyen/action_recog/temporal-segment-networks/models/ucf101/tsn_bn_inception_rgb_train_val_ae_2.prototxt"
NET_WEIGHTS = "/home/anhnguyen/action_recog/temporal-segment-networks/models/ucf101_split_1_tsn_rgb_reference_bn_inception.caffemodel"
OUTPUT = "results/rgb_center_fgsm_iter_" + str(EPS) + "_rate_" + str(ATTACK_RATE) + ".txt"

f_out = open(OUTPUT,'w')

# Loading a network
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(NET_PROTOTXT, NET_WEIGHTS, caffe.TEST)

# Set up transformer
input_shape = net.blobs['data'].data.shape
transformer = caffe.io.Transformer({'data': (input_shape)})
if net.blobs['data'].data.shape[1] == 3:
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', np.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
else:
    pass # non RGB data need not use transformer

def display_info():	
	# Display information
	print("EPS = " + str(EPS))
	print("ALPHA = " + str(ALPHA))
	print("NUM_ITER = " + str(NUM_ITER))
	print("ATTACK_RATE = " + str(ATTACK_RATE))
	print("BATCH_SIZE = " + str(BATCH_SIZE))
	print("NUM_FR_PER_VID = " + str(NUM_FR_PER_VID))
	print("VIDEO_LIST = " + VIDEO_LIST)
	print("OUTPUT = " + OUTPUT)
	# print("AE_PATH = " + VIDEO_AE_PATH)
	print("------------------------------------------")

def predict_batch(data_in, labels_in, calculate_grad=False):
	global net

	# Set data blob and label blob
	net.blobs['data'].data[...] = data_in
	net.blobs['label'].data[...] = labels_in

	# Forward data
	out = net.forward()

	if calculate_grad:
		# Calculate backward gradient of loss layer w.r.t input
		data_diff = net.backward(diffs=['data'])
		grad_data = data_diff['data']
	
		return out['probs'].copy(), grad_data

	return out['probs'].copy()

def save_ae_frames(frame_paths, ae_batch, clean_batch):
	# Create output directory
	if os.path.isdir(VIDEO_AE_PATH) == False:
		os.system("mkdir " + VIDEO_AE_PATH)

	video = frame_paths[0].split('/')[-2]
	out_dir = os.path.join(VIDEO_AE_PATH, video)
	if os.path.isdir(out_dir) == False:
		os.system("mkdir " + out_dir)

	# Reverse effects of transformer preprocess
	ae_crops_deprocess = []
	for m in range(0, BATCH_SIZE):
	    ae_crop = ae_batch[m,:,:,:]
	    ae_crop += np.array([[[104]],[[117]],[[123]]])
	    # ae_crop = ae_crop[(2,1,0),:,:]
	    ae_crop = ae_crop.transpose((1,2,0))
	    ae_crop /= 255.0

	    # Clip exceeded values
	    ae_crop[ae_crop>1] = 1.
	    ae_crop[ae_crop<0] = 0.

	    # Add ae crop to clean frame
	    frame = clean_batch[m]
	    frame[16:240,58:282,:] = ae_crop[:,:,:]*255
	    frame = frame[:,:,(2,1,0)]

	    # Extract name of frame
	    fr_name = frame_paths[m].split('/')[-1]

	    img = Image.fromarray((frame*1).astype('uint8'))
	    img.save(os.path.join(out_dir,fr_name))

def display_blobs():
	global net

	print("Blobs:")
	for name, blob in net.blobs.iteritems():
		print("{:<5}:  {}".format(name, blob.data.shape))

display_info()

# Read video list
f = open(VIDEO_LIST, 'r')
f_lines = f.readlines()
f.close()

true_labels = []
pred_labels = []

for i_line, line in enumerate(f_lines):
	# Extract video name and its label in each line
	video = line.split(' ')[0].split('/')[1]
	label = int(line.split(' ')[1])

	true_labels.append(label)

	# Get all frames in video directory
	vid_dir = VIDEO_PATH + "/" + video
	# frames = glob.glob(vid_dir+"/img_*.jpg")
	frames = glob.glob(vid_dir+"/*.jpg")
	frames.sort()
	num_frames = len(frames)

	# Select 25 frames from each video
	selected_ids = np.round(np.linspace(0, num_frames-1, NUM_FR_PER_VID))
	assert(len(selected_ids) == NUM_FR_PER_VID)
	
	selected_frames = list(frames[int(i)] for i in selected_ids)

	# Attack rate
	if ATTACK_RATE == 1:
		frames_to_attack = selected_frames
	elif ATTACK_RATE == 2:
		frames_to_attack = list(selected_frames[i] for i in [0,2,4,6,8,10,12,14,16,18,20,22,24])
	elif ATTACK_RATE == 3:
		frames_to_attack = list(selected_frames[i] for i in [0,3,6,9,12,15,18,21,24])
	else:
		frames_to_attack = []

	# Create input array
	batch_data = []

	# Create array to store output probabilities
	probs = []

	# Read frame data 
	for idx,frame_path in enumerate(selected_frames):

		# Read frame data
		frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
		frame_resized = cv2.resize(frame, (340,256))
		
		# Add to batch array
		batch_data.append(frame_resized)

		# When number of read frame equals batch size
		# feed batch into network
		if(len(batch_data) == BATCH_SIZE):

			# Name of frames in batch currently being processed
			process_fr_names = selected_frames[idx-BATCH_SIZE+1:idx+1]
			
			# Oversample batch
			batch_os = caffe.io.oversample(batch_data, [224,224])

			# Extract center crops
			batch_crops = np.array(list(batch_os[i] for i in [4,14,24,34,44]))
			
			# Preprocess input batch
			batch_data_in = []
			for ix, crop in enumerate(batch_crops):
				crop = transformer.preprocess('data', crop)
				batch_data_in.append(crop)
			batch_data_in = np.array(batch_data_in)


			if ATTACK:
				# Iterative FGSM
				clip_min = batch_data_in - EPS
				clip_max = batch_data_in + EPS

				if NUM_ITER <= 0:
					num_iters = np.min([EPS + 4, 1.25*EPS])
					num_iters = int(np.max([np.ceil(num_iters), 1]))
				else:
					num_iters = NUM_ITER

				ae_batch = np.array(batch_data_in)
				label_batch = np.zeros((BATCH_SIZE,1,1,1))+label

				# Iterate to craft ae
				for i in range(num_iters):
					
					# Forward batch data
					score_batch, grad = predict_batch(ae_batch, label_batch, calculate_grad=True)

					# Get sign of gradient
					signed_grad = np.sign(grad)

					# Step perturbation 
					step_pertub = signed_grad*ALPHA

					# In each step add only small perturb
					ae_batch = np.clip(ae_batch+step_pertub, clip_min, clip_max)

					# # Total perturb w.r.t clean data
					# total_pertub = batch_data_in - ae_batch

					# if np.max(np.abs(total_pertub)) >= EPS:
					# 	break

				# Add pertubation to batch data
				ae_batch_final = np.array(batch_data_in)
				for i, frame in enumerate(process_fr_names): 
					if frame in frames_to_attack:
						ae_batch_final[i,:,:,:] = ae_batch[i,:,:,:]

				# Predict ae batch
				score_batch = predict_batch(ae_batch_final, label, calculate_grad=False)
				score_batch = np.squeeze(score_batch)

				if i_line == 0:
					save_ae_frames(process_fr_names, ae_batch_final, batch_data)	

			# Add batch score into probs array of all frames
			probs.extend(score_batch)

			# Erase batch_data array
			batch_data = []

	score_vid = np.mean(probs,0)
	pred_label = score_vid.argmax()
	pred_labels.append(pred_label)
	print(str(i_line)+" "+video+" "+str(label)+" "+str(pred_label))

	# Calculate partial accuracy
	if(i_line % 20 == 0 and i_line > 0):
		print('Accuracy: ' + str(accuracy_score(true_labels[0:i_line], pred_labels[0:i_line])*100) + '%')

	# Write video probabilities to file
	f_out.writelines("%s " %video)
	f_out.writelines("%s " %label)
	f_out.writelines("%.6f " %prob for prob in score_vid)
	f_out.writelines("\n")

print('Accuracy: ' + str(accuracy_score(true_labels, pred_labels)*100) + '%')

f_out.close()

display_info()